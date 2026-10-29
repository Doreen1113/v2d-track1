"""2D proxy evaluation of refined body runs against three detectors, after the assembly smoothing.

For each run directory (episode_XXXXXX.npz with pose [T,136], scales, shape; optional holdout [T]) the pose is
smoothed as assemble_episode.py does (--sigma-h), the MHR 308 keypoints are projected and compared on the
evaluation frames (held-out frames if the run has a holdout mask, else all valid frames):
  vit_px                 ViTPose-H COCO-17 (score > 0.5)                         [used as a fitting target]
  sap_<group>_px         Sapiens2 groups body / feet / wrist / hand / extra / face (score > 0.5)  [target in v6]
  rtmw_<group>_px        RTMW-x whole-body (independent): body (COCO-17), feet (toes / heels), wrist, hand
                         (score > --rtmw-thr; scored frames only)
  rtm_finger_px / rtm_wrist_px / rtm_finger_rel   RTMPose hand crops (work/hand_kp2d, score > 0.25; scored frames)
Per-episode means are averaged over episodes (episodes missing a group are skipped for that group).
With --ref DIR also the first-scored-frame change w.r.t. that run (first_pa_v2v_cm, first_mpjpe_cm).

  python eval_sap_proxies.py DEV/ho_base DEV/ho_feet DEV/ho_all --sigma-h 3 --episodes 0 3 6 --out DEV/sap_eval.json
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_body_proxies import assembled_pose, sim3  # noqa: E402
from eval_hand_proxy import L70, R70  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402

WORK = "/work/doreen071/v2d/work"
J22 = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
COCO_TO_70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
GROUPS = dict(body=list(range(0, 15)), feet=list(range(15, 21)), wrist=[41, 62],
              hand=list(range(21, 41)) + list(range(42, 62)), extra=list(range(63, 70)), face=list(range(70, 308)))
# RTMW (COCO-WholeBody 133) -> MHR308 index
RTMW_FEET = {17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20}
RTMW_LH = list(range(91, 112))   # wrist, thumb1..4, index1..4, ... (COCO-WholeBody hand order)
RTMW_RH = list(range(112, 133))


def regress(kmap, V, J):
    """MHR308 keypoints [T,308,3] from verts [T,V,3] and joints [T,127,3] (sparse regressor over [verts | joints])."""
    vj = np.concatenate([V, J], 1)
    T, N, _ = vj.shape
    return np.asarray(kmap @ vj.transpose(1, 0, 2).reshape(N, T * 3)).reshape(-1, T, 3).transpose(1, 0, 2)


def project(kp, K):
    return np.stack([K[0] * kp[..., 0] / kp[..., 2] + K[2], K[1] * kp[..., 1] / kp[..., 2] + K[3]], -1)


def mean_or_nan(e, m):
    return float(e[m].mean()) if m.any() else float("nan")


def evaluate(d, ep, mhr, kmap, sigma_h, scored, rtmw_thr, ref=None, intr_dir=os.path.join(WORK, "body")):
    b = np.load(os.path.join(d, f"episode_{ep:06d}.npz"))
    raw = np.load(os.path.join(intr_dir, f"episode_{ep:06d}.npz"))
    valid = raw["valid"].astype(bool)
    src = b if "fx" in b.files else raw  # refine_body_v6 runs carry the intrinsics of their input
    K = np.array([float(src["fx"]), float(src["fy"]), float(src["cx"]), float(src["cy"])])
    pose = assembled_pose(b, sigma_h)
    T = len(pose)
    V, J = mhr(pose, b["scales"], b["shape"])
    kp = regress(kmap, V, J)
    uv = project(kp, K)
    ho = b["holdout"].astype(bool) if "holdout" in b.files else np.zeros(T, bool)
    ev = (valid & ho) if ho.any() else valid.copy()
    sc = np.zeros(T, bool)
    sc[[f for f in scored if f < T]] = True
    out = {}
    # ViTPose
    vit = torch.load(os.path.join(WORK, "body_gvhmr/preproc", f"episode_{ep:06d}_feat.pt"), weights_only=False)
    kv = vit["kp2d"].numpy().astype(np.float64)
    kv[~vit["detected"].numpy().astype(bool), :, 2] = 0
    e = np.linalg.norm(uv[:, COCO_TO_70] - kv[..., :2], axis=-1)
    out["vit_px"] = mean_or_nan(e, (kv[..., 2] > 0.5) & ev[:, None])
    out["vit_px_scored"] = mean_or_nan(e, (kv[..., 2] > 0.5) & (ev & sc)[:, None])
    # Sapiens
    p = os.path.join(WORK, "sap_kp", f"episode_{ep:06d}.npz")
    if os.path.exists(p):
        s = np.load(p)
        skp, ssc = s["kp"].astype(np.float64), s["score"].astype(np.float64)
        es = np.linalg.norm(uv - skp, axis=-1)
        for g, ids in GROUPS.items():
            out[f"sap_{g}_px"] = mean_or_nan(es[:, ids], (ssc[:, ids] > 0.5) & ev[:, None])
    # RTMW whole-body (scored frames)
    p = os.path.join(WORK, "sap_rtmw", f"episode_{ep:06d}.npz")
    if os.path.exists(p):
        r = np.load(p)
        fr = r["frames"].astype(int)
        keep = ev[fr]
        fr, rk, rs = fr[keep], r["kp"][keep].astype(np.float64), r["score"][keep].astype(np.float64)
        if len(fr):
            u = uv[fr]
            e = np.linalg.norm(u[:, COCO_TO_70] - rk[:, :17], axis=-1)
            out["rtmw_body_px"] = mean_or_nan(e, rs[:, :17] > rtmw_thr)
            src, dst = list(RTMW_FEET), list(RTMW_FEET.values())
            e = np.linalg.norm(u[:, dst] - rk[:, src], axis=-1)
            out["rtmw_feet_px"] = mean_or_nan(e, rs[:, src] > rtmw_thr)
            src = RTMW_RH + RTMW_LH
            dst = R70 + L70
            e = np.linalg.norm(u[:, dst] - rk[:, src], axis=-1)
            m = rs[:, src] > rtmw_thr
            wr = np.zeros(42, bool)
            wr[[0, 21]] = True
            out["rtmw_wrist_px"] = mean_or_nan(e[:, wr], m[:, wr])
            out["rtmw_hand_px"] = mean_or_nan(e[:, ~wr], m[:, ~wr])
    # RTMPose hand crops (scored frames)
    p = os.path.join(WORK, "hand_kp2d", f"episode_{ep:06d}.npz")
    if os.path.exists(p):
        h = np.load(p)
        fr = h["frames"].astype(int)
        if len(fr):
            keep = ev[fr]
            det = np.stack([h["right"], h["left"]], 1)[keep]
            fr = fr[keep]
            if len(fr):
                pred = np.stack([uv[fr][:, R70], uv[fr][:, L70]], 1)
                conf = det[..., 2] > 0.25
                size = np.linalg.norm(det[:, :, 9, :2] - det[:, :, 0, :2], axis=-1)
                e = np.linalg.norm(pred - det[..., :2], axis=-1)
                fing, wr = conf.copy(), conf.copy()
                fing[..., 0] = False
                wr[..., 1:] = False
                out["rtm_finger_px"] = mean_or_nan(e, fing)
                out["rtm_wrist_px"] = mean_or_nan(e, wr)
                out["rtm_finger_rel"] = mean_or_nan(e / np.maximum(size[..., None], 5.0) * 100, fing)
    if ref is not None:
        rr = np.load(os.path.join(ref, f"episode_{ep:06d}.npz"))
        first = scored[0] if scored else int(np.flatnonzero(valid)[0])
        Vr, Jr = mhr(assembled_pose(rr, sigma_h)[[first]], rr["scales"], rr["shape"])
        out["first_mpjpe_cm"] = float(np.linalg.norm(Jr[0, J22] - J[first, J22], axis=-1).mean() * 100)
        out["first_pa_v2v_cm"] = float(np.linalg.norm(sim3(V[first], Vr[0]) - Vr[0], axis=-1).mean() * 100)
        # change on the scored frames after one similarity fitted on the first scored frame (as the CD-H alignment)
        fr = np.array([f for f in scored if f < T]) if scored else np.flatnonzero(valid)
        Vr, _ = mhr(assembled_pose(rr, sigma_h)[fr], rr["scales"], rr["shape"])
        a0, b0 = V[fr[0]], Vr[0]
        ma, mb = a0.mean(0), b0.mean(0)
        u, sv, vt = np.linalg.svd((a0 - ma).T @ (b0 - mb))
        c = np.ones(3)
        c[-1] = -1 if np.linalg.det(u @ vt) < 0 else 1
        R = (u * c) @ vt
        sc = (sv * c).sum() / ((a0 - ma) ** 2).sum()
        Va = sc * (V[fr] - ma) @ R + mb
        out["v2v_change_cm"] = float(np.linalg.norm(Va - Vr, axis=-1).mean() * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sigma-h", type=float, default=3.0)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--rtmw-thr", type=float, default=3.0, help="RTMW SimCC score threshold")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--intr-dir", default=os.path.join(WORK, "body"),
                    help="body dir with fx/fy/cx/cy and valid, for runs that do not carry their own intrinsics")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHRForward(device=dev)
    kmap = sp.load_npz(os.path.join(WORK, "sap_kp/kp308_mapping.npz")).tocsr()
    scored = {int(k): v for k, v in json.load(open(os.path.join(WORK, "body_v2/scored_frames.json"))).items()}
    res = {d: {} for d in a.dirs}
    for ep in a.episodes:
        for d in a.dirs:
            if not os.path.exists(os.path.join(d, f"episode_{ep:06d}.npz")):
                continue
            res[d][ep] = evaluate(d, ep, mhr, kmap, a.sigma_h, scored.get(ep, []), a.rtmw_thr, a.ref, a.intr_dir)
        print(ep, " | ".join(f"{res[d][ep].get('vit_px', np.nan):.2f}/{res[d][ep].get('rtmw_feet_px', np.nan):.2f}/"
                             f"{res[d][ep].get('rtm_finger_px', np.nan):.2f}" for d in a.dirs if ep in res[d]), flush=True)
    eps = sorted(set.intersection(*[set(r) for r in res.values()]))
    keys = sorted({k for r in res.values() for e in r.values() for k in e})
    summ = {}
    for d in a.dirs:
        summ[d] = {}
        for k in keys:
            v = [res[d][e][k] for e in eps if k in res[d][e] and np.isfinite(res[d][e][k])]
            summ[d][k] = float(np.mean(v)) if v else float("nan")
    print(f"{len(eps)} episodes, sigma_h {a.sigma_h}")
    order = ["vit_px", "vit_px_scored", "rtmw_body_px", "rtmw_feet_px", "rtmw_wrist_px", "rtmw_hand_px",
             "rtm_finger_px", "rtm_wrist_px", "rtm_finger_rel", "sap_body_px", "sap_feet_px", "sap_wrist_px",
             "sap_hand_px", "sap_extra_px", "sap_face_px", "first_pa_v2v_cm", "first_mpjpe_cm", "v2v_change_cm"]
    order = [k for k in order if k in keys]
    print(f"{'':>40s} " + " ".join(f"{k.replace('_px', ''):>13s}" for k in order))
    for d in a.dirs:
        print(f"{d[-40:]:>40s} " + " ".join(f"{summ[d][k]:13.3f}" for k in order))
    # paired comparison with the first directory: mean difference and number of episodes that got better
    base = a.dirs[0]
    print(f"paired vs {base[-40:]} (mean diff / episodes better of n)")
    for d in a.dirs[1:]:
        cells = []
        for k in order:
            dd = [res[d][e][k] - res[base][e][k] for e in eps
                  if k in res[d][e] and k in res[base][e] and np.isfinite(res[d][e][k]) and np.isfinite(res[base][e][k])]
            cells.append(f"{np.mean(dd):+7.3f} {sum(x < 0 for x in dd):2d}/{len(dd):<2d}" if dd else f"{'':>13s}")
        print(f"{d[-40:]:>40s} " + " ".join(cells))
    if a.out:
        json.dump(dict(per_episode=res, summary=summ, episodes=eps, sigma_h=a.sigma_h), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
