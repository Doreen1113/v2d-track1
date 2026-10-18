"""Remove hand-object interpenetration by opening the fingers instead of moving the object.

Only the MHR pose parameters that drive finger joints are changed (parameters whose joints all lie
strictly inside a wrist subtree, wrists 41 / 77 excluded). They do not move any of the scorer's 22
body joints (the wrist and the joints 42 / 78 keep their positions), so ACC-H is unchanged and
CD-H changes only on finger vertices. Per penetrating scored frame the finger parameters are
optimised against the signed distance of the submission-budgeted object mesh (grid, exact near
the surface) (penetration with
a small margin + a quadratic pull to the original pose); the result is verified with the scorer's
exact inside test and frames that still penetrate are reported (fix_penetration.py can then move
the object for those).

    python fix_penetration_fingers.py --episodes-dir EPS [--episodes ...]     (edits EPS in place)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contactlib as cl  # noqa: E402
from mhr_utils import DEFAULT_MHR, FLIP, MHRForward  # noqa: E402

WRISTS = (41, 77)


def finger_params(model):
    b = {k: v.cpu().numpy() for k, v in model.named_buffers()}
    P = b["character_torch.parameter_transform.parameter_transform"]
    par = b["character_torch.skeleton.joint_parents"]
    J = len(par)
    A = np.abs(P[:J * 7].reshape(J, 7, -1)).sum(1)
    out = []
    for w in WRISTS:
        sub = set()
        for j in range(J):
            p = par[j]
            while p >= 0:
                if p == w:
                    sub.add(j)
                    break
                p = par[p]
        out += [p for p in range(136) if A[:, p].sum() > 0 and set(np.flatnonzero(A[:, p])) <= sub]
    return np.array(sorted(out))


class Forward(torch.nn.Module):
    def __init__(self, mhr):
        super().__init__()
        self.m = mhr.model
        self.flip = torch.tensor(FLIP, dtype=torch.float32, device=mhr.device)

    def forward(self, pose, scales, shape):
        p = torch.cat([pose, scales.expand(len(pose), -1)], 1)
        expr = torch.zeros(len(p), 72, device=p.device)
        v, _ = self.m(shape.expand(len(p), -1).contiguous(), p, expr, True)
        return v * self.flip / 100.0


def fix_episode(path, mhr, fwd, hand_ids, fidx, frames, iters, margin, w_reg, rounds=4):
    data = dict(np.load(path))
    mesh = next(p for s in (".ply", ".obj", ".glb") if (p := path.with_name(path.stem + "_object" + s)).exists())
    mv, mf = cl.budget(mesh)
    sdf = cl.SDFGrid(mv, mf, res=128)
    esdf = cl.ExactSDF(mv, mf, device=str(mhr.device))
    pose = data["pose"]
    R, t, s = data["object_rotation"], data["object_translation"], float(data["object_scale"])
    verts, _ = mhr(pose[frames], data["scales"], data["shape"])
    hands = verts[:, hand_ids]

    def exact(h, fr):
        local = np.einsum("tpi,tij->tpj", h - t[fr, None], R[fr]) / s
        return cl.penetration_depth(local.reshape(-1, 3), mv, mf).reshape(local.shape[:2]) * s

    d0 = exact(hands, frames)
    bad = frames[(d0 > 0).any(1)]
    before = float(d0.mean() * 100)
    if len(bad) == 0:
        return before, before, 0, 0
    dev = mhr.device
    sc = torch.tensor(data["scales"], dtype=torch.float32, device=dev)[None]
    sh = torch.tensor(data["shape"], dtype=torch.float32, device=dev)[None]
    base = torch.tensor(pose[bad], dtype=torch.float32, device=dev)
    Rt = torch.tensor(R[bad], dtype=torch.float32, device=dev)
    tt = torch.tensor(t[bad], dtype=torch.float32, device=dev)
    hid = torch.tensor(hand_ids, device=dev)
    fi = torch.tensor(fidx, device=dev)
    delta = torch.zeros(len(bad), len(fidx), device=dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.03)
    for rnd in range(rounds):
        for it in range(iters):
            p = base.clone()
            p[:, fi] = p[:, fi] + delta
            h = fwd(p, sc, sh)[:, hid]
            local = torch.einsum("tpi,tij->tpj", h - tt[:, None], Rt) / s
            sd = sdf.query_hybrid(local, esdf, 0.01 / s) * s * 100.0
            pen = torch.relu(margin * (rnd + 1) - sd)
            loss = 10.0 * (pen + pen ** 2).sum() + w_reg * (delta ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = base.clone()
            p[:, fi] = p[:, fi] + delta
            h = fwd(p, sc, sh)[:, hid].cpu().numpy()
        d1 = exact(h, bad)
        if not (d1 > 0).any():
            break
    pose = pose.copy()
    pose[bad] = p.cpu().numpy()
    data["pose"] = pose
    np.savez(path, **data)
    verts, _ = mhr(pose[frames], data["scales"], data["shape"])
    d2 = exact(verts[:, hand_ids], frames)
    return before, float(d2.mean() * 100), int((d2 > 0).any(1).sum()), len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--margin", type=float, default=0.2, help="clearance target (cm) in the SDF loss")
    ap.add_argument("--w-reg", type=float, default=0.01)
    a = ap.parse_args()
    mhr = MHRForward(DEFAULT_MHR, device="cuda" if torch.cuda.is_available() else "cpu")
    for prm in mhr.model.parameters():
        prm.requires_grad_(False)
    fwd = Forward(mhr)
    hand_ids = cl.hand_ids_from(mhr.model)
    fidx = finger_params(mhr.model)
    scored = cl.scored()
    eps = a.episodes if a.episodes is not None else sorted(
        int(p.stem.split("_")[1]) for p in a.episodes_dir.glob("episode_??????.npz"))
    for ep in eps:
        b, af, n, nb = fix_episode(a.episodes_dir / f"episode_{ep:06d}.npz", mhr, fwd, hand_ids, fidx,
                                   scored[ep], a.iters, a.margin, a.w_reg)
        print(f"episode {ep}: hand depth {b:.5f} -> {af:.5f} cm (body frame), penetrating frames {nb} -> {n}",
              flush=True)


if __name__ == "__main__":
    main()
