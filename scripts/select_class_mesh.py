"""Pick one shared mesh per object class by multi-frame silhouette agreement over its 3 episodes.

Candidates per class: the v1 meshes of its three episodes (WORK/objects), every SAM 3D Objects
candidate of WORK/objects_v2/candidates (object_candidates.py), and for the hula hoop a torus
(a hoop is a torus; only its size and tube thickness are fitted, from the video).

A candidate C is scored on an episode e through the v1 trajectory of e (mesh M_e, per-frame
pose s_e, R_f, t_f): a single similarity G maps C into M_e's local frame,
    point_cam(f) = s_e R_f (g_s g_R c + g_t) + t_f,
found by a rotation search (octahedral group + random rotations, silhouette IoU on a few
frames) followed by a joint refinement over ~24 well-tracked frames (silhouette + MoGe depth,
so the scale stays consistent with that episode's depth), then a short per-frame refinement.
Score = median per-frame hard IoU (actor pixels ignored). The class mesh maximises the mean
score over the class's episodes; its implied size (body frame) is also reported.

Outputs: WORK/objects_v2/class_mesh/<class>.ply, selection.json (all scores)
         WORK/objects_v2/init/episode_XXXXXX.npz (R [T,3,3], t [T,3], s, mesh): the v1 trajectory
         mapped onto the class mesh, the starting point of track_object_v2.py

Usage: python select_class_mesh.py --classes hula_hoop black_pan   (default: all)
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import objlib  # noqa: E402
from common import WORK, load_meta  # noqa: E402
from objlib import V2, Episode, FrameTargets, Renderer, data_loss, so3_exp  # noqa: E402

THIN = {"hula_hoop"}


def torus_mesh(path, minor=0.015, n_major=96, n_minor=12):
    """Unit-diameter torus (major radius 0.5) in the x-y plane."""
    import trimesh
    u = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    r = 0.5 + minor * np.cos(vv)
    V = np.stack([r * np.cos(uu), r * np.sin(uu), minor * np.sin(vv)], -1).reshape(-1, 3)
    F = []
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = ((i + 1) % n_major) * n_minor + j
            c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            d = i * n_minor + (j + 1) % n_minor
            F += [[a, b, c], [a, c, d]]
    trimesh.Trimesh(V, np.array(F), process=False).export(path)
    return path


def rotation_hypotheses(n_random=72, seed=0):
    from scipy.spatial.transform import Rotation
    rots = list(Rotation.create_group("O").as_matrix())
    rots += list(Rotation.random(n_random, random_state=seed).as_matrix())
    return np.stack(rots)


def bbox_centre(V):
    return 0.5 * (V.min(0) + V.max(0))


class EpisodeRef:
    """v1 trajectory of an episode + evaluation frames."""

    def __init__(self, ep, n_eval=24):
        self.ep = ep
        self.E = Episode(ep)
        tr = np.load(os.path.join(WORK, "objects", f"episode_{ep:06d}_track.npz"))
        self.R, self.t, self.s = tr["R"], tr["t"], float(tr["scale"])
        self.iou_v1 = tr["iou"]
        self.mesh = os.path.join(WORK, "objects", f"episode_{ep:06d}_object.ply")
        Vm, _ = objlib.load_mesh(self.mesh, 3000)
        self.centre = bbox_centre(Vm)
        self.rms = float(np.sqrt(((Vm - self.centre) ** 2).sum(1).mean()))
        # evaluation frames: the best-tracked visible frames of each time bin (also low-IoU ones
        # are kept in proportion, so a mesh that explains all views is preferred)
        T = self.E.T
        vis = np.flatnonzero(self.E.visible & (self.E.area > 30))
        bins = np.array_split(vis, n_eval)
        fr = []
        for b in bins:
            if len(b):
                fr.append(int(b[np.argmax(self.iou_v1[b])]))
        self.frames = np.array(fr)
        self.tg = FrameTargets(self.E, self.frames)
        self.ren = Renderer(self.E.K, self.E.H, self.E.W)


def score_candidate(ref, cpath, device="cuda", thin=False, verbose=False):
    Vc_np, Fc_np = objlib.load_mesh(cpath, 2000)
    cc = bbox_centre(Vc_np)
    rms_c = float(np.sqrt(((Vc_np - cc) ** 2).sum(1).mean()))
    sg0 = ref.rms / rms_c
    V = torch.tensor(Vc_np, dtype=torch.float32, device=device)
    F = torch.tensor(Fc_np, dtype=torch.int64, device=device)
    fr = ref.frames
    Rf = torch.tensor(ref.R[fr], dtype=torch.float32, device=device)
    tf = torch.tensor(ref.t[fr], dtype=torch.float32, device=device)
    cen_e = torch.tensor(ref.centre, dtype=torch.float32, device=device)
    cen_c = torch.tensor(cc, dtype=torch.float32, device=device)
    s_e = ref.s

    def posed(Rg, sg, tg_off, idx):
        """Rg [H,3,3], sg [H], tg_off [H,3] -> verts [H*len(idx), V, 3]."""
        local = sg[:, None, None] * torch.einsum("vj,hij->hvi", V - cen_c, Rg) + (cen_e + tg_off)[:, None]
        cam = s_e * torch.einsum("hvj,fij->hfvi", local, Rf[idx]) + tf[idx][None, :, None]
        cz = (s_e * torch.einsum("hj,fij->hfi", (cen_e + tg_off).expand(local.shape[0], 3), Rf[idx])
              + tf[idx][None])[..., 2]
        return cam.reshape(-1, V.shape[0], 3), cz.reshape(-1)

    # 1) rotation search on 8 frames
    hyp = torch.tensor(rotation_hypotheses(), dtype=torch.float32, device=device)
    H = len(hyp)
    sub = torch.arange(0, len(fr), max(1, len(fr) // 8), device=device)[:8]
    losses = torch.zeros(H, device=device)
    with torch.no_grad():
        for a in range(0, H, 8):
            hh = hyp[a:a + 8]
            verts, cz = posed(hh, torch.full((len(hh),), sg0, device=device), torch.zeros(len(hh), 3, device=device), sub)
            sil, z = ref.ren(verts, F)
            tg = ref.tg.subset(sub.repeat(len(hh)))
            l = data_loss(sil, z, tg, zc=cz, thin=thin).reshape(len(hh), len(sub)).mean(1)
            losses[a:a + 8] = l
    top = torch.argsort(losses)[:4]
    if verbose:
        print("  hyp losses: identity", float(losses[0]), "top", losses[top].tolist(), top.tolist())
    # 2) joint refinement of G over all eval frames, for the top hypotheses
    allf = torch.arange(len(fr), device=device)
    best = None
    for h in top:
        w = torch.zeros(3, device=device, requires_grad=True)
        dt = torch.zeros(3, device=device, requires_grad=True)
        ls = torch.zeros((), device=device, requires_grad=True)
        prm = [{"params": [w], "lr": 0.01}, {"params": [dt], "lr": 0.005 / s_e}]
        if not thin:
            prm.append({"params": [ls], "lr": 0.005})
        opt = torch.optim.Adam(prm)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 80, eta_min=1e-4)
        for it in range(80):
            Rg = (so3_exp(w) @ hyp[h])[None]
            verts, cz = posed(Rg, (sg0 * torch.exp(ls))[None], dt[None], allf)
            sil, z = ref.ren(verts, F)
            l = data_loss(sil, z, ref.tg, zc=cz, thin=thin).mean()
            opt.zero_grad()
            l.backward()
            opt.step()
            sch.step()
        if best is None or float(l) < best[0]:
            best = (float(l), (so3_exp(w) @ hyp[h]).detach(), float(sg0 * torch.exp(ls)), dt.detach().clone())
    _, Rg, sg, dt = best
    if verbose:
        loc = (sg * (V - cen_c) @ Rg.T + cen_e + dt).detach().cpu().numpy()
        i_g = [objlib.frame_iou(ref.E, int(f), s_e * loc @ ref.R[f].T + ref.t[f], Fc_np) for f in fr]
        print("  after G refine: loss", best[0], "median IoU", np.median(i_g), "sg/sg0", sg / sg0)
    # 3) per-frame refinement + hard IoU
    wf = torch.zeros(len(fr), 3, device=device, requires_grad=True)
    tfd = torch.zeros(len(fr), 3, device=device, requires_grad=True)
    opt = torch.optim.Adam([{"params": [wf], "lr": 0.005}, {"params": [tfd], "lr": 0.002}])
    local = sg * (V - cen_c) @ Rg.T + cen_e + dt
    for it in range(30):
        Rp = so3_exp(wf) @ Rf
        verts = s_e * torch.einsum("vj,fij->fvi", local, Rp) + (tf + tfd)[:, None]
        cz = (s_e * torch.einsum("j,fij->fi", cen_e + dt, Rp) + tf + tfd)[:, 2]
        sil, z = ref.ren(verts, F)
        l = data_loss(sil, z, ref.tg, zc=cz, thin=thin)
        opt.zero_grad()
        l.sum().backward()
        opt.step()
    with torch.no_grad():
        Rp = (so3_exp(wf) @ Rf).cpu().numpy()
        tp = (tf + tfd).cpu().numpy()
        loc = local.cpu().numpy()
    ious = np.array([objlib.frame_iou(ref.E, int(f), s_e * loc[:, :] @ Rp[k].T + tp[k], Fc_np)
                     for k, f in enumerate(fr)])
    # similarity: c -> M_e local: x = sg Rg (c - cc) + centre_e + dt  =>  g_R = Rg, g_s = sg,
    # g_t = centre_e + dt - sg Rg cc
    gR = Rg.cpu().numpy().astype(np.float64)
    gt = ref.centre + dt.cpu().numpy() - sg * gR @ cc
    size = s_e * sg * float(np.linalg.norm(Vc_np.max(0) - Vc_np.min(0)))  # bbox diagonal, MoGe metres
    return dict(median_iou=float(np.median(ious)), mean_iou=float(ious.mean()), g_R=gR.tolist(),
                g_s=float(sg), g_t=gt.tolist(), diag_moge=size)


def candidates_for(eps, cls):
    out = []
    for ep in eps:
        out.append((f"v1_ep{ep}", os.path.join(WORK, "objects", f"episode_{ep:06d}_object.ply")))
        for p in sorted(glob.glob(os.path.join(V2, "candidates", f"episode_{ep:06d}", "f*.ply"))):
            out.append((f"ep{ep}_{os.path.basename(p)[:-4]}", p))
    if cls == "hula_hoop":
        d = os.path.join(V2, "class_mesh")
        os.makedirs(d, exist_ok=True)
        out.append(("torus", torus_mesh(os.path.join(d, "torus_hoop.ply"))))
    return out


def run_class(cls, eps, ratio):
    t0 = time.time()
    thin = cls in THIN
    refs = [EpisodeRef(ep) for ep in eps]
    res = {}
    for name, path in candidates_for(eps, cls):
        r = {}
        for ref in refs:
            try:
                r[ref.ep] = score_candidate(ref, path, thin=thin)
            except Exception as e:  # broken candidate mesh
                print(f"[{cls}] {name} on ep {ref.ep} failed: {e}", flush=True)
                r[ref.ep] = None
        if any(v is None for v in r.values()):
            continue
        body_diag = [r[ep]["diag_moge"] / ratio[str(ep)] for ep in eps]
        score = float(np.mean([r[ep]["median_iou"] for ep in eps]))
        res[name] = dict(path=path, score=score, per_ep=r, body_diag=body_diag)
        print(f"[{cls}] {name:22s} score {score:.3f} " +
              " ".join(f"ep{ep}:{r[ep]['median_iou']:.3f}" for ep in eps) +
              f" diag(body) {np.round(body_diag, 3)}  ({time.time() - t0:.0f}s)", flush=True)
    best = max(res, key=lambda k: res[k]["score"])
    print(f"[{cls}] BEST {best} score {res[best]['score']:.3f}", flush=True)
    return res, best


def write_init(cls, eps, res, best):
    import trimesh
    d = os.path.join(V2, "class_mesh")
    os.makedirs(d, exist_ok=True)
    src = res[best]["path"]
    dst = os.path.join(d, f"{cls}.ply")
    m = trimesh.load(src, process=False)
    m.export(dst)
    os.makedirs(os.path.join(V2, "init"), exist_ok=True)
    # The same physical object: an episode whose implied (body-frame) scale is > 15 % off the
    # class median is rescaled about the camera centre (silhouettes unchanged), e.g. the
    # oversized hoop of episode 2.
    ratio = objlib.load_ratio()
    body_s = {ep: float(np.load(os.path.join(WORK, "objects", f"episode_{ep:06d}_track.npz"))["scale"])
              * res[best]["per_ep"][ep]["g_s"] / ratio[str(ep)] for ep in eps}
    med = float(np.median(list(body_s.values())))
    for ep in eps:
        tr = np.load(os.path.join(WORK, "objects", f"episode_{ep:06d}_track.npz"))
        g = res[best]["per_ep"][ep]
        gR, gs, gt = np.array(g["g_R"]), g["g_s"], np.array(g["g_t"])
        s_e = float(tr["scale"])
        R = tr["R"] @ gR
        t = s_e * np.einsum("tij,j->ti", tr["R"], gt) + tr["t"]
        k = med / body_s[ep] if abs(np.log(med / body_s[ep])) > np.log(1.15) else 1.0
        if k != 1.0:
            print(f"[{cls}] ep {ep}: scale {body_s[ep]:.3f} vs class median {med:.3f} -> rescaled by {k:.3f}",
                  flush=True)
        np.savez(os.path.join(V2, "init", f"episode_{ep:06d}.npz"), R=R, t=t * k, s=s_e * gs * k, mesh=dst,
                 source=src, size_k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*")
    ap.add_argument("--init-only", action="store_true", help="re-write init files from selection/<class>.json")
    ap.add_argument("--override", nargs="*", default=[],
                    help="class=candidate: use this (scored) candidate instead of the top score, e.g. when the "
                         "top one is incomplete (black_pan=ep13_f000042: the best-scoring pan has no handle)")
    a = ap.parse_args()
    meta = load_meta()
    by_cls = {}
    for ep, m in sorted(meta.items()):
        by_cls.setdefault(m["object"], []).append(ep)
    ratio = objlib.load_ratio()
    os.makedirs(os.path.join(V2, "selection"), exist_ok=True)
    for cls in (a.classes or sorted(by_cls)):
        eps = by_cls[cls]
        if a.init_only:
            sp = os.path.join(V2, "selection", f"{cls}.json")
            d = json.load(open(sp))
            res = {k: dict(v, per_ep={int(e): r for e, r in v["per_ep"].items()}) for k, v in d["results"].items()}
            ov = dict(o.split("=") for o in a.override)
            if cls in ov:
                d["best_by_score"] = d.get("best_by_score", d["best"])
                d["best"] = ov[cls]
                json.dump(d, open(sp, "w"), indent=1)
                print(f"[{cls}] using {d['best']} (top score: {d['best_by_score']})", flush=True)
            write_init(cls, eps, res, d["best"])
            continue
        res, best = run_class(cls, eps, ratio)
        json.dump(dict(best=best, results=res), open(os.path.join(V2, "selection", f"{cls}.json"), "w"), indent=1)
        write_init(cls, eps, res, best)


if __name__ == "__main__":
    main()
