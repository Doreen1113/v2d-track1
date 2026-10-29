"""Oracle variants of a validation run (diagnostic only): replace one component of the prediction with the ground
truth, write the episodes to runs/oracle_<x>_<name>/episodes and score them with val_score.py as any run.

MHR parametrisation used here (checked on the official model): joint 1 is the body root; its position is
offset_1 + 10 cm * pose[0:3] (MHR native frame, y / z flipped w.r.t. the camera) and its orientation is
euler_xyz(pose[3:6]) (extrinsic x-y-z, identity pre-rotation); joint 0 is a fixed world joint. So the root
translation moves the whole body rigidly, and the root rotation turns it about joint 1 without moving joint 1.
"Pelvis" below is joint 1 (the first of the scorer's 22 joints).

The scorer fits one Sim(3) (s, R, t; rows: x_gt = s x_pred R + t) on the alignment vertices of the first scored
frame f0. The trajectory / orientation oracles keep frame f0 as predicted (so that Sim(3) does not change) and replace
the motion relative to f0 with the GT motion mapped into the prediction's frame by that Sim(3):
  a     pelvis trajectory:  p'(t) = p(f0) + (p_gt(t) - p_gt(f0)) R^T / s      (whole body shifted rigidly)
  b     as a, camera z only (the depth axis of the prediction's camera)
  c     as a, camera x / y only
  d     global (torso) orientation relative to f0: torso joints = root, hips, spine 1-3, neck, clavicles, upper arms
        (root-relative; the GT's mapped into the prediction's frame by R). A_p(t), A_g(t) = Kabsch rotation of the
        predicted / GT torso from f0 to t; the whole body is turned about the pelvis by A_g(t) A_p(t)^T, so the
        predicted torso turns since f0 as the GT torso does
  dabs  the whole body turned about the pelvis by C(t) = Kabsch(predicted torso -> GT torso) in every frame, f0 too
        (the Sim(3) refits)
  ad    a and d together (the root's rigid motion relative to f0)
  e     body articulation: GT pose[6:136] except the finger columns 68:122 (assemble_episode.FINGER_COLS), with
        the GT root rotation, the predicted root translation and identity, then the whole body turned about the pelvis
        so its torso joints have the predicted torso orientation (Kabsch)
  f     identity: GT shape and scales
  g     global rigid motion (mesh level): per frame the rigid transform that best maps the predicted alignment
        vertices onto the GT's (mapped into the prediction's frame by the inverse Sim(3)) is applied to the whole
        predicted body (root rotation + translation), f0 included; what is left is articulation + identity error
  grot  the rotation part of g only, about the pelvis (pelvis path as predicted)
  h     the complement of g: the GT body (articulation + identity) placed per frame by the rigid transform that best
        maps it onto the aligned prediction, i.e. only the predicted global rigid motion is wrong
  all   every parameter from the GT (sanity check: CD-H 0)
d / dabs / e turn the body about the pelvis by a torso-based rotation while the predicted articulation (fitted to
the 2D keypoints together with the predicted root) stays, so they can break compensations between the two; g / grot
/ h use the whole mesh and are the better-posed split of global motion vs articulation + identity.
Parameter-level variants, reported to show why they are ill-defined here: the predicted MHR (SAM 3D Body) and the GT
MHR (fitted to SMPL vertices) split the trunk orientation differently between the root rotation and the spine
joints, so swapping the root rotation alone (droot: as d but G'(t) = R dG(t) R^T G(f0) with dG(t) = G_gt(t) G_gt(f0)^T
the GT root-rotation change; drootabs: the GT root rotation as is) or the articulation alone with the predicted root
(eparam) produces a trunk that is bent twice or not at all.
For e, f and dabs frame f0 changes too, so their effect includes the change of the frame-0 alignment.

  python val_oracle.py --pred /work/doreen071/v2d_val/runs/moge/episodes --name moge
  bash val_score_all.sh oracle_a_moge oracle_b_moge ...
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import val_score as vs  # noqa: E402
from assemble_episode import FINGER_COLS  # noqa: E402
from mhr_utils import FLIP, MHRForward, cam_t_to_root_trans  # noqa: E402

VAL = "/work/doreen071/v2d_val"
ORACLES = ("a", "b", "c", "d", "dabs", "ad", "e", "f", "g", "grot", "h", "all", "droot", "drootabs", "eparam")
TORSO = [1, 2, 18, 35, 36, 37, 110, 74, 38, 75, 39]  # root, up-legs, spine 1-3, neck, clavicles, up-arms (MHR ids)
F = np.diag(FLIP)


def root_frame(pose):
    """Orientation of joint 1 in the camera (Track 1) frame: F @ euler_xyz(pose[3:6])."""
    return F @ Rotation.from_euler("xyz", pose[:, 3:6]).as_matrix()


def set_root_frame(pose, G):
    pose = pose.copy()
    pose[:, 3:6] = Rotation.from_matrix(F @ G).as_euler("xyz")
    return pose


def kabsch(A, B):
    """Per frame rotation C (column form) minimising sum |C a_i - b_i|^2; A, B [T, N, 3] (centred by the caller)."""
    H = np.einsum("tni,tnj->tij", A, B)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", U, Vt)))
    D = np.repeat(np.eye(3)[None], len(A), 0)
    D[:, 2, 2] = d
    return np.einsum("tji,tjk,tkl->til", Vt, D, U.transpose(0, 2, 1)).transpose(0, 1, 2)


def ident(z):
    sc, sh = np.asarray(z["scales"], np.float64), np.asarray(z["shape"], np.float64)
    return (np.median(sc, 0) if sc.ndim == 2 else sc), (np.median(sh, 0) if sh.ndim == 2 else sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="assembled run (runs/<name>/episodes)")
    ap.add_argument("--name", required=True, help="suffix of the output runs: runs/oracle_<x>_<name>")
    ap.add_argument("--gt", default=f"{VAL}/gt")
    ap.add_argument("--oracles", nargs="*", default=list(ORACLES))
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    a = ap.parse_args()
    kit = vs.load_kit("CD-H")
    body = vs.build_body(kit, vs.DEFAULT_MHR, "all")
    align = body.roles["alignment"]
    fw = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    scored = {int(k): np.asarray(v, int) for k, v in json.load(open(os.path.join(a.gt, "scored_frames.json"))).items()}
    eps = a.episodes if a.episodes is not None else sorted(scored)
    log = {}
    for ep in eps:
        zp = dict(np.load(os.path.join(a.pred, f"episode_{ep:06d}.npz")))
        zg = np.load(os.path.join(a.gt, f"episode_{ep:06d}.npz"))
        pose = np.asarray(zp["pose"], np.float64)
        gpose = np.asarray(zg["pose"], np.float64)[: len(pose)]
        sc, sh = ident(zp)
        gsc, gsh = ident(zg)
        f0 = int(scored[ep][0])
        Vp, Jp = fw(pose, sc, sh)
        Vg, Jg = fw(gpose, gsc, gsh)
        s, R, _t = kit.fit_similarity_transform(Vp[f0, align], Vg[f0, align])
        s, R, _t = float(s), np.asarray(R), np.asarray(_t)
        pp, pg = Jp[:, 1], Jg[:, 1]
        target = pp[f0] + (pg - pg[f0]) @ R.T / s
        Gp, Gg = root_frame(pose), root_frame(gpose)
        Gd = R @ (Gg @ Gg[f0].T) @ R.T @ Gp[f0]  # droot
        # torso orientation: GT torso (root-relative) mapped into the prediction's frame vs the predicted torso
        tp = Jp[:, TORSO] - pp[:, None]
        tg = (Jg[:, TORSO] - pg[:, None]) @ R.T
        C = kabsch(tp, tg)
        rep = lambda x: np.repeat(x[f0][None], len(x), 0)  # noqa: E731
        Ap, Ag = kabsch(rep(tp), tp), kabsch(rep(tg), tg)  # torso rotation since f0, prediction / GT
        M = np.einsum("tij,tkj->tik", Ag, Ap)  # Ag Ap^T: the GT's torso rotation since f0 replaces the prediction's
        Gt = np.einsum("tij,tjk->tik", M, Gp)  # d
        Gtabs = np.einsum("tij,tjk->tik", C, Gp)  # dabs
        # e: GT articulation and root rotation on the predicted root position / identity, re-oriented to the
        # predicted torso
        # g / grot: per-frame rigid map of the predicted mesh onto the GT mesh in the prediction's frame
        Gs = (Vg[:, align] - _t) @ R.T / s
        Pv = Vp[:, align]
        cp, cg = Pv.mean(1), Gs.mean(1)
        Qg = kabsch(Pv - cp[:, None], Gs - cg[:, None])  # column form: x' = Qg (x - cp) + cg
        Gg_ = np.einsum("tij,tjk->tik", Qg, Gp)
        pg_ = np.einsum("tij,tj->ti", Qg, pp - cp) + cg
        # h: the GT body placed by the rigid map of the GT mesh onto the aligned prediction (in the GT's frame)
        Pa = s * Vp[:, align] @ R + _t
        ca, cgt = Pa.mean(1), Vg[:, align].mean(1)
        Qh = kabsch(Vg[:, align] - cgt[:, None], Pa - ca[:, None])
        Gh = np.einsum("tij,tjk->tik", Qh, Gg)
        ph = np.einsum("tij,tj->ti", Qh, pg - cgt) + ca
        PH = set_root_frame(gpose, Gh)
        PH[:, 0:3] += cam_t_to_root_trans(ph - pg)
        art = [c for c in range(6, 136) if c not in set(FINGER_COLS)]
        PB = pose.copy()
        PB[:, 3:6] = gpose[:, 3:6]
        PB[:, art] = gpose[:, art]
        _, JB = fw(PB, sc, sh)
        C2 = kabsch(JB[:, TORSO] - JB[:, 1:2], tp)
        PB = set_root_frame(PB, np.einsum("tij,tjk->tik", C2, root_frame(PB)))
        log[ep] = {"sim3_scale": s, "sim3_rot_deg": float(np.degrees(Rotation.from_matrix(R).magnitude())),
                   "pelvis_traj_err_cm": float(np.linalg.norm(target - pp, axis=1)[scored[ep]].mean() * 100)}
        for x in a.oracles:
            P, S, H = pose.copy(), sc.copy(), sh.copy()
            if x in ("a", "b", "c", "ad"):
                new = pp.copy()
                cols = {"a": [0, 1, 2], "ad": [0, 1, 2], "b": [2], "c": [0, 1]}[x]
                new[:, cols] = target[:, cols]
                P[:, 0:3] += cam_t_to_root_trans(new - pp)
            if x in ("d", "ad"):
                P = set_root_frame(P, Gt)
            if x == "dabs":
                P = set_root_frame(P, Gtabs)
            if x == "droot":
                P = set_root_frame(P, Gd)
            if x == "drootabs":
                P[:, 3:6] = gpose[:, 3:6]
            if x == "e":
                P = PB.copy()
            if x == "eparam":
                P[:, art] = gpose[:, art]
            if x in ("g", "grot"):
                P = set_root_frame(P, Gg_)
                if x == "g":
                    P[:, 0:3] += cam_t_to_root_trans(pg_ - pp)
            if x == "h":
                P, S, H = PH.copy(), gsc.copy(), gsh.copy()
            if x == "f":
                S, H = gsc.copy(), gsh.copy()
            if x == "all":
                P, S, H = gpose.copy(), gsc.copy(), gsh.copy()
            out = os.path.join(VAL, "runs", f"oracle_{x}_{a.name}", "episodes")
            os.makedirs(out, exist_ok=True)
            rec = dict(zp)
            rec.update(pose=P, scales=S, shape=H)
            np.savez(os.path.join(out, f"episode_{ep:06d}.npz"), **rec)
            for ext in (".ply", ".obj", ".glb", ".off", ".stl"):
                m = os.path.join(a.pred, f"episode_{ep:06d}_object{ext}")
                if os.path.exists(m):
                    shutil.copy(m, out)
        # self-check on ad: pelvis path as intended (mm), root frames as intended, torso rotation since f0 = the GT's (deg)
        if "ad" in a.oracles:
            q = np.load(os.path.join(VAL, "runs", f"oracle_ad_{a.name}", "episodes", f"episode_{ep:06d}.npz"))
            _, Jq = fw(q["pose"], sc, sh)
            log[ep]["check_pelvis_mm"] = float(np.abs(Jq[:, 1] - target).max() * 1000)
            log[ep]["check_rot"] = float(np.abs(root_frame(q["pose"]) - Gt).max())
            tq = Jq[:, TORSO] - Jq[:, 1:2]
            dq = np.einsum("tij,tkj->tik", kabsch(rep(tq), tq), Ag)
            log[ep]["check_torso_rot_deg"] = float(np.degrees(Rotation.from_matrix(dq).magnitude()).max())
        log[ep]["rigid_rot_err_deg"] = float(np.degrees(Rotation.from_matrix(Qg).magnitude())[scored[ep]].mean())
        log[ep]["torso_rot_err_deg"] = float(np.degrees(Rotation.from_matrix(M).magnitude())[scored[ep]].mean())
        print(ep, {k: round(v, 4) for k, v in log[ep].items()}, flush=True)
    json.dump({str(k): v for k, v in log.items()}, open(os.path.join(VAL, "results", f"oracle_{a.name}_build.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
