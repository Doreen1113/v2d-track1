"""Classical focal estimate from Manhattan line structure (no learned prior).

Line segments (OpenCV LSD) are back-projected to interpretation-plane normals n_s(K). For a
Manhattan scene there is a rotation R such that every structural segment satisfies n_s . R e_k = 0
for one axis k. We
  1. profile: for each focal f on a grid (principal point at the image centre) run RANSAC over
     Manhattan frames with common random triplets and record the robust alignment score;
  2. refine (f, R) — and in variants also (cx, cy) or a division-model k1 — by robust least squares
     on the inlier segments, starting from the best profile point;
  3. bootstrap the segments for an uncertainty.
Works on the person-free background plates (and optionally the sampled frames with the foreground
masked). Rows go to <out>/vp<tag>.jsonl, profiles to <out>/vp_profiles<tag>.npz.
--region upper/lower restricts the segments to the ceiling/upper walls or to the floor/lower part,
as a check that the result does not hinge on one family of lines (e.g. carpet texture).

    python fcal_vp.py --workers 12 [--frames-per-episode 8] [--region upper --tag _upper]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path("/work/doreen071/v2d")
F_GRID = np.arange(550.0, 1150.0 + 1e-6, 5.0)
THR = np.sin(np.deg2rad(1.0))


def detect(gray, invalid=None, min_len=30.0, border=6):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    L = lsd.detect(gray)[0]
    if L is None:
        return np.zeros((0, 4))
    L = L.reshape(-1, 4).astype(np.float64)
    le = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    h, w = gray.shape
    keep = le >= min_len
    for (x, y) in ((L[:, 0], L[:, 1]), (L[:, 2], L[:, 3])):
        keep &= (x > border) & (x < w - border) & (y > border) & (y < h - border)
    L = L[keep]
    if invalid is not None and len(L):
        # drop segments whose samples touch the invalid (foreground) region
        t = np.linspace(0, 1, 9)
        xs = (L[:, None, 0] * (1 - t) + L[:, None, 2] * t).round().astype(int).clip(0, w - 1)
        ys = (L[:, None, 1] * (1 - t) + L[:, None, 3] * t).round().astype(int).clip(0, h - 1)
        L = L[~invalid[ys, xs].any(1)]
    return L


def normals(L, f, cx, cy, k1=0.0):
    def ray(x, y):
        u, v = (x - cx) / f, (y - cy) / f
        if k1 != 0.0:
            s = 1.0 / (1.0 + k1 * (u * u + v * v))  # division model, distorted -> undistorted
            u, v = u * s, v * s
        return np.stack([u, v, np.ones_like(u)], -1)
    # use endpoints and midpoint (midpoint matters when k1 != 0): plane through 3 rays (fit by SVD)
    r1, r2 = ray(L[:, 0], L[:, 1]), ray(L[:, 2], L[:, 3])
    n = np.cross(r1, r2)
    if k1 != 0.0:
        rm = ray((L[:, 0] + L[:, 2]) / 2, (L[:, 1] + L[:, 3]) / 2)
        R3 = np.stack([r1, rm, r2], 1)
        R3 = R3 / np.linalg.norm(R3, axis=-1, keepdims=True)
        ns = np.linalg.svd(R3)[2][:, -1]
        n = ns * np.sign(np.einsum("sk,sk->s", ns, n) + 1e-12)[:, None]  # keep the cross-product orientation
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def frame_hyps(N, abc):
    a, b, c = abc
    d1 = np.cross(N[a], N[b])
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True) + 1e-12
    d2 = np.cross(d1, N[c])
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True) + 1e-12
    d3 = np.cross(d1, d2)
    return np.stack([d1, d2, d3], 1)  # H,3(axes),3


def score(N, w, D):
    r = np.abs(np.einsum("sk,hjk->hsj", N, D)).min(-1)  # H,S
    return (w[None] * np.clip(1 - (r / THR) ** 2, 0, None)).sum(-1)


def profile(L, w, cx, cy, H=3000, seed=0):
    rng = np.random.default_rng(seed)
    S = len(L)
    abc = rng.integers(0, S, (3, H))
    ok = (abc[0] != abc[1]) & (abc[0] != abc[2]) & (abc[1] != abc[2])
    abc = abc[:, ok]
    prof, best_D = [], []
    for f in F_GRID:
        N = normals(L, f, cx, cy)
        D = frame_hyps(N, abc)
        sc = score(N, w, D)
        i = int(np.argmax(sc))
        prof.append(sc[i])
        best_D.append(D[i])
    return np.array(prof), best_D


def refine(L, w, f0, R0, cx0, cy0, free_c=False, free_k=False, iters=4):
    """Robust LS over (f, rotvec[, cx, cy][, k1]) with assignments re-estimated each round."""
    rv0 = Rotation.from_matrix(R0.T).as_rotvec()  # D rows are axes: D = R^T rows -> R = D^T
    p = [f0, *rv0]
    if free_c:
        p += [cx0, cy0]
    if free_k:
        p += [0.0]
    p = np.array(p, float)

    def unpack(p):
        f, rv = p[0], p[1:4]
        j = 4
        cx, cy, k1 = cx0, cy0, 0.0
        if free_c:
            cx, cy = p[j], p[j + 1]
            j += 2
        if free_k:
            k1 = p[j]
        return f, Rotation.from_rotvec(rv).as_matrix().T, cx, cy, k1

    sw = np.sqrt(w / w.mean())
    inl = None
    for _ in range(iters):
        f, D, cx, cy, k1 = unpack(p)
        N = normals(L, f, cx, cy, k1)
        dots = np.abs(N @ D.T)
        k = dots.argmin(1)
        inl = dots.min(1) < 3 * THR
        if inl.sum() < 12:
            return None
        Li, ki, swi = L[inl], k[inl], sw[inl]

        def res(q):
            f, D, cx, cy, k1 = unpack(q)
            N = normals(Li, f, cx, cy, k1)
            return swi * np.einsum("sk,sk->s", N, D[ki])
        bounds_lo = [300.0, -np.inf, -np.inf, -np.inf] + ([0, 0] if free_c else []) + ([-1.0] if free_k else [])
        bounds_hi = [2500.0, np.inf, np.inf, np.inf] + ([2 * cx0, 2 * cy0] if free_c else []) + ([1.0] if free_k else [])
        p = least_squares(res, p, loss="cauchy", f_scale=THR, bounds=(bounds_lo, bounds_hi)).x
    f, D, cx, cy, k1 = unpack(p)
    N = normals(L, f, cx, cy, k1)
    dots = np.abs(N @ D.T)
    k = dots.argmin(1)
    inl = dots.min(1) < 2 * THR
    return {"f": float(f), "cx": float(cx), "cy": float(cy), "k1": float(k1), "D": D, "inl": inl, "k": k}


def vp_geometry(D, f, cx, cy):
    """Image position of the three vanishing points and their angle to the optical axis."""
    out = []
    for d in D:
        ang = float(np.rad2deg(np.arccos(min(1.0, abs(d[2])))))
        if abs(d[2]) > 1e-6:
            out.append({"x": float(f * d[0] / d[2] + cx), "y": float(f * d[1] / d[2] + cy), "axis_angle_deg": ang})
        else:
            out.append({"x": None, "y": None, "axis_angle_deg": ang})
    return out


def run_one(job):
    ep, cam, fid, ip, mp, boot, region = job
    img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    cx, cy = w / 2, h / 2
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    invalid = (m == 0) if fid == "bg" else (m > 0)
    L = detect(img, invalid)
    if region == "upper":  # ceiling and upper walls only (both endpoints above 0.40 H)
        L = L[(L[:, 1] < 0.40 * h) & (L[:, 3] < 0.40 * h)]
    elif region == "lower":  # floor, furniture and lower walls only (both endpoints below 0.50 H)
        L = L[(L[:, 1] > 0.50 * h) & (L[:, 3] > 0.50 * h)]
    wts = np.minimum(np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]), 300.0)
    rows = []
    if len(L) < 30:
        return rows, None
    prof, Ds = profile(L, wts, cx, cy)
    i = int(np.argmax(prof))
    f0 = float(F_GRID[i])
    base = {"episode": ep, "camera": cam, "frame": fid, "n_seg": int(len(L)), "f_profile_argmax": f0,
            "region": region}
    rng = np.random.default_rng(1)
    for name, fc, fk in (("vp_f", False, False), ("vp_fc", True, False), ("vp_fk", False, True)):
        r = refine(L, wts, f0, Ds[i], cx, cy, free_c=fc, free_k=fk)
        if r is None:
            rows.append({"method": name, **base, "error": "refine failed"})
            continue
        bs = []
        for _ in range(boot):
            j = rng.integers(0, len(L), len(L))
            rb = refine(L[j], wts[j], r["f"], r["D"], cx, cy, free_c=fc, free_k=fk, iters=2)
            if rb is not None:
                bs.append([rb["f"], rb["cx"], rb["cy"], rb["k1"]])
        bs = np.array(bs) if bs else np.full((1, 4), np.nan)
        counts = [int(((r["k"] == a) & r["inl"]).sum()) for a in range(3)]
        rows.append({"method": name, **base, "fx": r["f"], "fy": r["f"], "cx": r["cx"], "cy": r["cy"],
                     "k1": r["k1"], "boot_std_f": float(np.nanstd(bs[:, 0])),
                     "boot_std_cx": float(np.nanstd(bs[:, 1])), "boot_std_cy": float(np.nanstd(bs[:, 2])),
                     "inliers_per_axis": counts, "inlier_frac": float(r["inl"].mean()),
                     "vps": vp_geometry(r["D"], r["f"], r["cx"], r["cy"])})
    return rows, (ep, cam, fid, prof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, default=ROOT / "work/fcal_frames")
    ap.add_argument("--out", type=Path, default=ROOT / "work/fcal_results")
    ap.add_argument("--frames-per-episode", type=int, default=8)
    ap.add_argument("--boot", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--region", default="all", choices=["all", "upper", "lower"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in sorted(a.frames_dir.glob("episode_*")):
        m = json.loads((d / "meta.json").read_text())
        if a.episodes is not None and m["episode"] not in a.episodes:
            continue
        jobs.append((m["episode"], m["camera"], "bg", d / "bg.png", d / "bg_valid.png", a.boot, a.region))
        fr = m["frames"]
        if a.frames_per_episode > 0:
            pick = [fr[int(i)] for i in np.linspace(0, len(fr) - 1, a.frames_per_episode).round()]
            for i in pick:
                jobs.append((m["episode"], m["camera"], str(i), d / f"f_{i:06d}.png", d / f"m_{i:06d}.png", 0, a.region))
    profs = []
    with Pool(a.workers) as pool, open(a.out / f"vp{a.tag}.jsonl", "a") as f:
        for rows, pr in pool.imap_unordered(run_one, jobs):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if pr is not None:
                profs.append(pr)
            if rows:
                r0 = rows[0]
                print(r0["episode"], r0["camera"], r0["frame"],
                      [(r["method"], round(r.get("fx", np.nan), 1), round(r.get("boot_std_f", np.nan), 1)) for r in rows],
                      flush=True)
    np.savez_compressed(a.out / f"vp_profiles{a.tag}.npz", f_grid=F_GRID,
                        episode=np.array([p[0] for p in profs]), camera=np.array([p[1] for p in profs]),
                        frame=np.array([p[2] for p in profs]), profile=np.stack([p[3] for p in profs]))


if __name__ == "__main__":
    main()
