"""Build the local 3D validation set from BEHAVE (Bhatnagar et al., CVPR 2022): Track 1-style videos plus MHR ground truth.

Per sequence, one Kinect colour view is turned into a Track 1 episode:
  video   the Kinect colour stream (2048x1536, 30 fps), undistorted with the Kinect calibration onto a pinhole camera
          with square pixels and the principal point at the image centre, resized to 1536x1152, a window of
          --length frames, H.264 (val root data/track_1/videos/chunk-000/observation.images.exo_camera/episode_XXXXXX.mp4)
  human   BEHAVE's 30 fps registrations (SMPL-H parameters) -> SMPL vertices -> camera frame of that Kinect ->
          MHR lod1 topology (barycentric map of MHR's SMPL conversion tool, smpl2mhr_mapping.npz) -> MHR parameters with
          one identity per episode (the kit's tools/track1/mesh_to_mhr_params.py, the converter the challenge names
          for MHR-topology meshes). The per-frame conversion residual is kept.
  object  the registration template (BEHAVE objects/<name>/<name>_f*.ply, centred as in BEHAVE's parse_obj_pose.py)
          and its per-frame rotation / translation in the camera frame, scale 1.
GT frames: video frames whose timestamp has a registration within 20 ms; the others are not scored.

SMPL-H is approximated by SMPL (gendered, 10 betas): the SMPL-H model is not available here. The body pose of the 22
SMPL-H body joints is used as is, SMPL's two hand joints are left at zero (flat hands; BEHAVE's own SMPL-H hand pose is
one constant pose anyway). SMPL-H betas are not SMPL betas (38-96 mm vertex error if used directly), so per subject
the SMPL betas and a root offset are fitted to BEHAVE's SMPL-H keyframe meshes (1 fps, PLY), with the SMPL-H body pose
(--stage smplfit; held-out error 8-10 mm on non-hand vertices, 20-45 mm on the hands).

Stages (run in the gvhmr env: smplx + torch + cv2):
  plan      choose the Kinect / window (whole body inside the image, few dropped frames); writes <gt>/plan.json
  intr      GT intrinsics (+ focal sweep) in the pipeline format (<root>/work/intrinsics_gt*/), scored_frames*.json
  video     write the episode videos (needs the extracted BEHAVE colour mp4s)
  smplfit   per-subject SMPL fit to the SMPL-H keyframe meshes (needs <keyframes>/<seq>/t*.000/person/fit02)
  gt        SMPL -> MHR conversion and object poses; writes <gt>/episode_XXXXXX.npz (+ _object.ply), gt_info.json
  meta      episodes_metadata.jsonl for the pipeline copies
  check-keyframes   SMPL (without the fit) vs the SMPL-H keyframe meshes

  python val_build_behave.py --stage plan intr video meta smplfit gt
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

BEHAVE = "/work/doreen071/v2d_val/data/behave"
VAL = "/work/doreen071/v2d_val"
SMPL_DIR = "/work/doreen071/spatial/lhmpp/repo/pretrained_models/human_model_files"
SMPL2MHR = "/work/doreen071/v2d/third_party/MHR/tools/mhr_smpl_conversion/assets/smpl2mhr_mapping.npz"
KIT_TOOLS = "/work/doreen071/v2d/kit/v2d_submission_kit/tools/track1"
MHR_MODEL = "/work/doreen071/v2d/assets/mhr_model.pt"
FFMPEG = "/home/doreen071/.conda/envs/lyra2/bin/ffmpeg"
W_OUT, H_OUT = 1536, 1152
W_SRC, H_SRC = 2048, 1536

# (sequence, Kinect, first frame of the window, object prompt); Kinect and window from a full-body visibility scan
# (all SMPL joints inside the image on >= 99% of the window's frames), views mixed on purpose
SEQUENCES = [
    ("Date01_Sub01_suitcase_lift", 1, 240, "suitcase"),
    ("Date01_Sub01_monitor_hand", 3, 30, "computer monitor"),
    ("Date01_Sub01_backpack_back", 0, 270, "backpack"),
    ("Date02_Sub02_boxlarge_hand", 0, 600, "cardboard box"),
    ("Date02_Sub02_trashbin", 3, 30, "trash bin"),
    ("Date02_Sub02_tablesmall_move", 0, 60, "small table"),
    ("Date04_Sub05_boxsmall", 0, 390, "cardboard box"),
    ("Date04_Sub05_suitcase", 2, 630, "suitcase"),
    ("Date05_Sub06_boxlarge", 0, 780, "cardboard box"),
    ("Date05_Sub06_monitor_hand", 2, 0, "computer monitor"),
    ("Date06_Sub07_boxlarge", 0, 30, "cardboard box"),
    ("Date06_Sub07_monitor_move", 1, 450, "computer monitor"),
    # second batch: windows whose Kinect colour stream drops no frames (<= 2 jumps > 40 ms in 600 frames)
    ("Date01_Sub01_trashbin", 1, 240, "trash bin"),
    ("Date01_Sub01_boxlong_hand", 2, 60, "cardboard box"),
    ("Date02_Sub02_monitor_hand", 1, 60, "computer monitor"),
    ("Date02_Sub02_suitcase_ground", 1, 120, "suitcase"),
    ("Date04_Sub05_toolbox", 2, 510, "toolbox"),
    ("Date05_Sub06_backpack_hand", 1, 270, "backpack"),
    ("Date06_Sub07_tablesmall_move", 1, 570, "small table"),
]
# Episodes whose 600-frame window has many dropped Kinect frames (time jumps of 66-500 ms between consecutive video
# frames: 83-206 jumps): the video is not uniform in time, which the pipeline's temporal terms assume, and the GT
# acceleration has spikes. Built and run, but scored separately (scored_frames_framedrop.json).
FRAME_DROP = {1, 2, 3, 4, 5, 6, 9}
SIMPLIFIED = {"backpack": "backpack_f1000.ply", "basketball": "basketball_f1000.ply", "boxlarge": "boxlarge_f1000.ply",
              "boxtiny": "boxtiny_f1000.ply", "boxlong": "boxlong_f1000.ply", "boxsmall": "boxsmall_f1000.ply",
              "boxmedium": "boxmedium_f1000.ply", "chairblack": "chairblack_f2500.ply",
              "chairwood": "chairwood_f2500.ply", "monitor": "monitor_closed_f1000.ply",
              "keyboard": "keyboard_f1000.ply", "plasticcontainer": "plasticcontainer_f1000.ply",
              "stool": "stool_f1000.ply", "tablesquare": "tablesquare_f2000.ply", "toolbox": "toolbox_f1000.ply",
              "suitcase": "suitcase_f1000.ply", "tablesmall": "tablesmall_f1000.ply", "yogamat": "yogamat_f1000.ply",
              "yogaball": "yogaball_f1000.ply", "trashbin": "trashbin_f1000.ply"}


def seq_date(seq):
    return seq.split("_")[0]


def seq_object(seq):
    return seq.split("_")[2]


def kinect_calib(seq, k):
    c = json.load(open(f"{BEHAVE}/calibs/calibs/intrinsics/{k}/calibration.json"))["color"]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
    dist = np.array(c["opencv"][4:], np.float64)  # k1 k2 p1 p2 k3 k4 k5 k6
    e = json.load(open(f"{BEHAVE}/calibs/calibs/{seq_date(seq)}/config/{k}/config.json"))
    R = np.array(e["rotation"], np.float64).reshape(3, 3)  # local -> world (world = Kinect 1 colour camera)
    t = np.array(e["translation"], np.float64)
    return K, dist, R, t


def target_K():
    """Pinhole camera of the output video (1536x1152), principal point at (W/2, H/2) like the pipeline assumes."""
    return np.array([[0, 0, W_OUT / 2], [0, 0, H_OUT / 2], [0, 0, 1.0]])


def out_focal(K_src):
    return 0.5 * (K_src[0, 0] + K_src[1, 1]) * W_OUT / W_SRC


def world_to_local(X, R, t):
    return (X - t) @ R


def aa_to_mat(aa):
    import torch
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, dtype=aa.dtype, device=aa.device) + th.sin() * K + (1 - th.cos()) * K @ K


def subject_of(seq):
    return seq.split("_")[1]


def subject_fit(seq):
    path = os.path.join(VAL, "gt", f"smpl_fit_{subject_of(seq)}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run --stage smplfit first (or --no-fit)")
    return json.load(open(path))


def smpl_world(seq, device="cpu", use_fit=True):
    """SMPL vertices / joints in the BEHAVE world frame for all 30 fps registrations of a sequence."""
    import smplx
    import torch
    p = np.load(f"{BEHAVE}/params/{seq}/smpl_fit_all.npz")
    gender = str(p["gender"])
    model = smplx.create(SMPL_DIR, model_type="smpl", gender=gender, num_betas=10).to(device)
    poses, betas, trans = (torch.as_tensor(p[k], dtype=torch.float32, device=device) for k in ("poses", "betas", "trans"))
    T = len(poses)
    delta = torch.zeros(3, device=device)
    fit = subject_fit(seq) if use_fit else None
    if fit is not None:  # SMPL shape + root offset fitted to this subject's SMPL-H keyframe meshes
        betas = torch.as_tensor(fit["betas"], dtype=torch.float32, device=device)[None].expand(T, -1)
        delta = torch.as_tensor(fit["delta"], dtype=torch.float32, device=device)
    body_pose = torch.cat([poses[:, 3:66], torch.zeros(T, 6, device=device)], 1)  # SMPL hand joints 22, 23 at rest
    vs, js = [], []
    with torch.no_grad():
        for a in range(0, T, 256):
            o = model(betas=betas[a:a + 256], body_pose=body_pose[a:a + 256], global_orient=poses[a:a + 256, :3])
            # BEHAVE's SMPL layer (smplpytorch, center_idx=0) centres on the root joint only when trans is zero;
            # with a translation it is the usual verts + trans (checked against the keyframe meshes)
            off = trans[a:a + 256] + aa_to_mat(poses[a:a + 256, :3]) @ delta
            vs.append((o.vertices + off[:, None]).cpu().numpy())
            js.append((o.joints[:, :24] + off[:, None]).cpu().numpy())
    times = np.array([float(s[1:]) for s in p["frame_times"]])
    return np.concatenate(vs), np.concatenate(js), times, gender, np.asarray(model.faces, np.int64)


def color_times(seq, k):
    return np.asarray(json.load(open(f"{BEHAVE}/time/{seq}.{k}.time.json"))["color"], np.float64) / 1e6


def match_frames(vt, gt_times, tol=0.02):
    """For each video frame: index of the registration within tol seconds (or -1)."""
    out = np.full(len(vt), -1)
    j = np.searchsorted(gt_times, vt)
    for i, (t, jj) in enumerate(zip(vt, j)):
        best = None
        for c in (jj - 1, jj):
            if 0 <= c < len(gt_times) and abs(gt_times[c] - t) < tol and (best is None or
                                                                         abs(gt_times[c] - t) < abs(gt_times[best] - t)):
                best = c
        if best is not None:
            out[i] = best
    return out


def stage_plan(args):
    import cv2
    path = os.path.join(args.gt, "plan.json")
    plan = {int(k): v for k, v in json.load(open(path)).items()} if os.path.exists(path) else {}
    for ep, (seq, kpref, spref, prompt) in enumerate(SEQUENCES):
        old = plan.get(ep)
        if old and old["sequence"] == seq and old["kinect"] == kpref and old["start"] == spref and not args.overwrite:
            continue
        _, J, gtt, gender, _ = smpl_world(seq, use_fit=False)
        cand = []
        for k in range(4):
            K, dist, R, t = kinect_calib(seq, k)
            vt = color_times(seq, k)
            m = match_frames(vt, gtt)
            # window with the most registered frames (ties: closest to the middle)
            L = min(args.length, len(vt))
            cs = np.concatenate([[0], np.cumsum(m >= 0)])
            counts = cs[L:] - cs[:-L]
            mid = (len(vt) - L) / 2
            starts = np.flatnonzero(counts == counts.max())
            s = int(starts[np.argmin(np.abs(starts - mid))]) if spref is None else int(spref)
            sel = m[s:s + L]
            gi = sel[sel >= 0]
            Jl = world_to_local(J[gi].reshape(-1, 3), R, t)
            uv = cv2.projectPoints(Jl, np.zeros(3), np.zeros(3), K, dist)[0].reshape(len(gi), -1, 2)
            z = Jl[:, 2].reshape(len(gi), -1)
            mg = 30
            inside = ((uv[..., 0] > mg) & (uv[..., 0] < W_SRC - mg) & (uv[..., 1] > mg) & (uv[..., 1] < H_SRC - mg)
                      & (z > 0.3)).all(1)
            height = np.median(uv[..., 1].max(1) - uv[..., 1].min(1))
            jumps = int((np.diff(vt[s:s + L]) > 0.040).sum())
            cand.append(dict(kinect=k, start=s, length=L, n_gt=int(len(gi)), vis=float(inside.mean()), time_jumps=jumps,
                             height_px=float(height), depth_m=float(np.median(z[:, 0]))))
        ok = [c for c in cand if c["vis"] >= 0.99 and c["n_gt"] >= 0.9 * args.length]
        if kpref is not None:
            pick = next(c for c in cand if c["kinect"] == kpref)
        elif ok:
            # rotate the preferred view over episodes so different viewpoints appear
            order = [(ep + i) % 4 for i in range(4)]
            pick = min(ok, key=lambda c: order.index(c["kinect"]))
        else:
            pick = max(cand, key=lambda c: (c["vis"], c["n_gt"]))
        plan[ep] = dict(sequence=seq, gender=gender, prompt=prompt, object=seq_object(seq), **pick, candidates=cand)
        print(ep, seq, "->", {k: pick[k] for k in ("kinect", "start", "n_gt", "vis", "time_jumps", "height_px", "depth_m")},
              "| vis per kinect", [round(c["vis"], 2) for c in cand], flush=True)
    os.makedirs(args.gt, exist_ok=True)
    json.dump({str(k): plan[k] for k in sorted(plan)}, open(path, "w"), indent=1)


def load_plan(args):
    return {int(k): v for k, v in json.load(open(os.path.join(args.gt, "plan.json"))).items()}


def video_dir(args):
    return os.path.join(args.root, "data/track_1/videos/chunk-000/observation.images.exo_camera")


def stage_video(args):
    import cv2
    plan = load_plan(args)
    os.makedirs(video_dir(args), exist_ok=True)
    for ep in args.episodes or sorted(plan):
        p = plan[ep]
        seq, k = p["sequence"], p["kinect"]
        out = os.path.join(video_dir(args), f"episode_{ep:06d}.mp4")
        if os.path.exists(out) and not args.overwrite:
            print(f"ep {ep}: video exists", flush=True)
            continue
        src = os.path.join(args.videos, f"{seq}.{k}.color.mp4")
        K, dist, _, _ = kinect_calib(seq, k)
        f = out_focal(K)
        s = W_SRC / W_OUT  # output pixel centre (x + 0.5) maps to source (x + 0.5) * s
        Kn = np.array([[f * s, 0, (W_OUT / 2 + 0.5) * s - 0.5], [0, f * s, (H_OUT / 2 + 0.5) * s - 0.5], [0, 0, 1]])
        mx, my = cv2.initUndistortRectifyMap(K, dist, None, Kn, (W_SRC, H_SRC), cv2.CV_32FC1)
        cap = cv2.VideoCapture(src)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        nt = len(color_times(seq, k))
        if abs(n - nt) > 1:
            print(f"ep {ep}: WARNING video has {n} frames, timestamps {nt}", flush=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, p["start"])
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W_OUT}x{H_OUT}",
               "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               out + ".tmp.mp4"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        t0, written = time.time(), 0
        for i in range(p["length"]):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"ep {ep}: read failed at frame {p['start'] + i}")
            und = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            proc.stdin.write(cv2.resize(und, (W_OUT, H_OUT), interpolation=cv2.INTER_AREA).tobytes())
            written += 1
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ep {ep}: ffmpeg failed")
        os.replace(out + ".tmp.mp4", out)
        print(f"ep {ep}: {seq} k{k} frames {p['start']}..{p['start'] + written - 1} -> {out} "
              f"({time.time() - t0:.0f}s)", flush=True)


def write_intrinsics(args, ep, f, tag="intrinsics_gt", extra=None):
    d = os.path.join(args.root, "work", tag)
    os.makedirs(d, exist_ok=True)
    rec = {"episode": ep, "width": W_OUT, "height": H_OUT, "fx": f, "fy": f, "cx": W_OUT / 2, "cy": H_OUT / 2,
           **(extra or {})}
    json.dump(rec, open(os.path.join(d, f"episode_{ep:06d}.json"), "w"), indent=1)


def stage_gt(args):
    import torch
    import trimesh
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, KIT_TOOLS)
    from mesh_to_mhr_params import convert

    plan = load_plan(args)
    mp = np.load(SMPL2MHR)
    info_path = os.path.join(args.gt, "gt_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    sc_path = os.path.join(args.gt, "scored_frames.json")
    scored = json.load(open(sc_path)) if os.path.exists(sc_path) else {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ep in args.episodes or sorted(plan):
        p = plan[ep]
        seq, k, s0, L = p["sequence"], p["kinect"], p["start"], p["length"]
        out = os.path.join(args.gt, f"episode_{ep:06d}.npz")
        if os.path.exists(out) and not args.overwrite:
            print(f"ep {ep}: gt exists", flush=True)
            continue
        t0 = time.time()
        V, J, gtt, gender, faces = smpl_world(seq, dev)
        K, dist, R, t = kinect_calib(seq, k)
        m = match_frames(color_times(seq, k)[s0:s0 + L], gtt)
        valid = m >= 0
        Vl = np.full((L, V.shape[1], 3), np.nan)
        Vl[valid] = world_to_local(V[m[valid]].reshape(-1, 3), R, t).reshape(-1, V.shape[1], 3)
        tri = faces[mp["triangle_ids"]]
        tgt = (Vl[:, tri] * mp["baryc_coords"][None, :, :, None]).sum(2)  # [L, 18439, 3], NaN on invalid frames
        res = convert(tgt, MHR_MODEL, device=dev, precision=args.precision, log=lambda *a, **k: None)
        err = np.asarray(res["per_frame_vertex_error_mm"], np.float64)
        # residual in SMPL topology too: re-posed MHR vs the SMPL vertices, through the inverse map
        o = np.load(f"{BEHAVE}/params/{seq}/object_fit_all.npz")
        Ro = Rotation.from_rotvec(o["angles"]).as_matrix()
        to = o["trans"]
        og = m.copy()
        og[~valid] = 0
        R_cam = np.einsum("ji,tjk->tik", R, Ro[og])  # R_k^T R_obj
        t_cam = world_to_local(to[og], R, t)
        # hold the object pose on unregistered frames at the nearest registered one (not scored anyway)
        idx = np.flatnonzero(valid)
        near = idx[np.abs(np.arange(L)[:, None] - idx[None]).argmin(1)]
        R_cam, t_cam = R_cam[near], t_cam[near]
        mesh = trimesh.load(f"{BEHAVE}/objects/objects/{p['object']}/{SIMPLIFIED[p['object']]}", process=False)
        mv = np.asarray(mesh.vertices, np.float64)
        mv = mv - mv.mean(0)
        trimesh.Trimesh(mv, np.asarray(mesh.faces), process=False).export(
            os.path.join(args.gt, f"episode_{ep:06d}_object.ply"))
        f = out_focal(K)
        np.savez_compressed(out, pose=res["pose"].astype(np.float64), scales=res["scales"].astype(np.float64),
                            shape=res["shape"].astype(np.float64), object_rotation=R_cam, object_translation=t_cam,
                            object_scale=np.float64(1.0), valid=valid, conv_err_mm=err, gt_index=m,
                            fx=f, fy=f, cx=W_OUT / 2, cy=H_OUT / 2)
        frames = np.flatnonzero(valid)
        scored[str(ep)] = frames.tolist()
        info[str(ep)] = dict(sequence=seq, kinect=k, start=s0, length=L, gender=gender, object=p["object"],
                             n_scored=int(len(frames)), fx=f, conv_err_mm_mean=float(err[valid].mean()),
                             conv_err_mm_median=float(np.median(err[valid])), conv_err_mm_max=float(err[valid].max()),
                             depth_m=p["depth_m"], height_px_src=p["height_px"])
        # intrinsics_gt is written by the intr stage (pipeline jobs may be reading it now)
        json.dump(scored, open(sc_path, "w"))
        json.dump(info, open(info_path, "w"), indent=1)
        print(f"ep {ep}: {seq} k{k} {valid.sum()}/{L} GT frames, SMPL->MHR residual mean {err[valid].mean():.2f} mm "
              f"median {np.median(err[valid]):.2f} max {err[valid].max():.2f} ({time.time() - t0:.0f}s)", flush=True)


def stage_intr(args):
    """GT intrinsics (and the focal sweep around them) in the pipeline format, and the scored frames."""
    plan = load_plan(args)
    sc_path = os.path.join(args.gt, "scored_frames.json")
    scored = json.load(open(sc_path)) if os.path.exists(sc_path) else {}
    for ep in sorted(plan):
        p = plan[ep]
        K, _, _, _ = kinect_calib(p["sequence"], p["kinect"])
        f = out_focal(K)
        write_intrinsics(args, ep, f, "intrinsics_gt", {"source": "BEHAVE Kinect calibration"})
        for k in args.sweep:
            write_intrinsics(args, ep, f * k, f"intrinsics_gt{round(k * 100):03d}",
                             {"source": f"BEHAVE Kinect calibration x {k}"})
        m = match_frames(color_times(p["sequence"], p["kinect"])[p["start"]:p["start"] + p["length"]],
                         smpl_frame_times(p["sequence"]))
        scored[str(ep)] = np.flatnonzero(m >= 0).tolist()
    os.makedirs(args.gt, exist_ok=True)
    keep = {k: v for k, v in scored.items() if int(k) not in FRAME_DROP}
    drop = {k: v for k, v in scored.items() if int(k) in FRAME_DROP}
    json.dump(keep, open(sc_path, "w"))
    json.dump(drop, open(os.path.join(args.gt, "scored_frames_framedrop.json"), "w"))
    d = os.path.join(args.root, "work/body_v2")
    os.makedirs(d, exist_ok=True)
    json.dump(scored, open(os.path.join(d, "scored_frames.json"), "w"))  # the pipeline sees every episode
    print("intrinsics and scored frames written", {k: len(v) for k, v in scored.items()}, flush=True)


def smpl_frame_times(seq):
    return np.array([float(s[1:]) for s in np.load(f"{BEHAVE}/params/{seq}/smpl_fit_all.npz")["frame_times"]])


def stage_meta(args):
    plan = load_plan(args)
    d = os.path.join(args.root, "data/track_1/meta")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "episodes_metadata.jsonl"), "w") as fh:
        for ep in sorted(plan):
            p = plan[ep]
            fh.write(json.dumps({"episode_index": ep, "object": p["object"], "object_prompt": p["prompt"],
                                 "sequence": p["sequence"], "kinect": p["kinect"]}) + "\n")
    print("meta written", flush=True)


def load_keyframes(seq_dir, stride=1):
    import glob
    import pickle
    import trimesh
    out = []
    for fd in sorted(glob.glob(os.path.join(seq_dir, "t*.000")))[::stride]:
        pk = os.path.join(fd, "person/fit02/person_fit.pkl")
        if os.path.exists(pk) and os.path.exists(pk.replace(".pkl", ".ply")):
            d = pickle.load(open(pk, "rb"), encoding="latin1")
            v = np.asarray(trimesh.load(pk.replace(".pkl", ".ply"), process=False).vertices, np.float32)
            out.append((np.asarray(d["pose"], np.float32), np.asarray(d["betas"], np.float32)[:10],
                        np.asarray(d["trans"], np.float32), v, os.path.basename(fd)))
    return out


def stage_smplfit(args):
    """Per subject: SMPL betas and a root offset (in the root frame) fitted to BEHAVE's SMPL-H keyframe meshes with
    the SMPL-H body pose, so that SMPL stands in for SMPL-H. Half of the keyframes fit, the other half report."""
    import glob
    import smplx
    import torch
    by_subject = {}
    for seq_dir in sorted(glob.glob(os.path.join(args.keyframes, "*"))):
        by_subject.setdefault(subject_of(os.path.basename(seq_dir)), []).append(seq_dir)
    for sub, dirs in by_subject.items():
        gender = json.load(open(os.path.join(dirs[0], "info.json")))["gender"]
        kf = [k for d in dirs for k in load_keyframes(d)]
        model = smplx.create(SMPL_DIR, model_type="smpl", gender=gender, num_betas=10)
        hand = model.lbs_weights[:, [20, 21, 22, 23]].sum(1).detach().numpy() > 0.5
        P = torch.as_tensor(np.stack([k[0] for k in kf]))
        tr = torch.as_tensor(np.stack([k[2] for k in kf]))
        V = torch.as_tensor(np.stack([k[3] for k in kf]))
        b0 = torch.as_tensor(np.stack([k[1] for k in kf])).mean(0)
        n = len(kf)
        body_pose = torch.cat([P[:, 3:66], torch.zeros(n, 6)], 1)
        Rg = aa_to_mat(P[:, :3])

        def pred(b, dl):
            o = model(betas=b[None].expand(n, -1), body_pose=body_pose, global_orient=P[:, :3])
            return o.vertices + (tr + Rg @ dl)[:, None]

        def err(v, idx):
            e = (v[idx] - V[idx]).norm(dim=-1)
            return e.mean().item() * 1000, e[:, ~hand].mean().item() * 1000, e[:, hand].mean().item() * 1000

        tr_idx, te_idx = np.arange(0, n, 2), np.arange(1, n, 2)
        with torch.no_grad():
            before = err(pred(b0, torch.zeros(3)), te_idx)
        b = b0.clone().requires_grad_(True)
        dl = torch.zeros(3, requires_grad=True)
        opt = torch.optim.Adam([b, dl], lr=0.02)
        for it in range(600):
            opt.zero_grad()
            v = pred(b, dl)
            loss = ((v[tr_idx] - V[tr_idx]) ** 2).sum(-1)[:, ~hand].mean()  # body vertices (hand poses differ)
            loss.backward()
            opt.step()
        with torch.no_grad():
            after = err(pred(b, dl), te_idx)
        rec = {"subject": sub, "gender": gender, "betas": b.detach().numpy().tolist(),
               "delta": dl.detach().numpy().tolist(), "n_keyframes": n, "sequences": [os.path.basename(d) for d in dirs],
               "heldout_err_mm_before": before, "heldout_err_mm_after": after}
        json.dump(rec, open(os.path.join(args.gt, f"smpl_fit_{sub}.json"), "w"), indent=1)
        print(f"{sub} ({gender}, {n} keyframes): held-out vertex error all/body/hands before {before[0]:.1f}/"
              f"{before[1]:.1f}/{before[2]:.1f} mm -> after {after[0]:.1f}/{after[1]:.1f}/{after[2]:.1f} mm, "
              f"delta {np.round(rec['delta'], 3)}", flush=True)


def stage_check_keyframes(args):
    """SMPL (this script) vs the SMPL-H keyframe meshes BEHAVE ships (sequences/<seq>/t*.000/person/fit02)."""
    import glob
    import pickle
    import smplx
    import torch
    import trimesh
    rows = []
    for seq_dir in sorted(glob.glob(os.path.join(args.keyframes, "*"))):
        seq = os.path.basename(seq_dir)
        info = json.load(open(os.path.join(seq_dir, "info.json")))
        model = smplx.create(SMPL_DIR, model_type="smpl", gender=info["gender"], num_betas=10)
        hand = model.lbs_weights[:, [20, 21, 22, 23]].sum(1).numpy() > 0.5  # wrists and hands
        for fd in sorted(glob.glob(os.path.join(seq_dir, "t*.000")))[:: args.kf_stride]:
            pk = os.path.join(fd, "person/fit02/person_fit.pkl")
            if not os.path.exists(pk):
                continue
            d = pickle.load(open(pk, "rb"), encoding="latin1")
            gt = np.asarray(trimesh.load(pk.replace(".pkl", ".ply"), process=False).vertices)
            po = torch.as_tensor(np.asarray(d["pose"], np.float32))[None]
            with torch.no_grad():
                o = model(betas=torch.as_tensor(np.asarray(d["betas"], np.float32)[None, :10]),
                          body_pose=torch.cat([po[:, 3:66], torch.zeros(1, 6)], 1), global_orient=po[:, :3])
            v = o.vertices[0].numpy() + np.asarray(d["trans"])
            e = np.linalg.norm(v - gt, axis=1)
            rows.append((seq, os.path.basename(fd), e.mean() * 1000, np.median(e) * 1000, e.max() * 1000,
                         e[~hand].mean() * 1000, e[hand].mean() * 1000))
    for r in rows:
        print(f"{r[0]} {r[1]} mean {r[2]:.1f} mm median {r[3]:.1f} max {r[4]:.1f} | body {r[5]:.1f} hands {r[6]:.1f}")
    if rows:
        a = np.array([r[2:] for r in rows])
        print(f"ALL {len(rows)} keyframes: mean {a[:, 0].mean():.2f} mm, median-of-medians {np.median(a[:, 1]):.2f}, "
              f"max {a[:, 2].max():.1f}; non-hand vertices {a[:, 3].mean():.2f} mm, hand vertices {a[:, 4].mean():.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["plan", "intr", "video", "smplfit", "gt", "meta", "check-keyframes"])
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--length", type=int, default=600, help="frames per episode")
    ap.add_argument("--gt", default=f"{VAL}/gt")
    ap.add_argument("--root", default=f"{VAL}/root", help="pipeline mirror root (videos, meta, work)")
    ap.add_argument("--videos", default=f"{BEHAVE}/videos", help="extracted BEHAVE colour mp4s")
    ap.add_argument("--keyframes", default=f"{BEHAVE}/keyframes", help="for check-keyframes: BEHAVE sequences dir")
    ap.add_argument("--kf-stride", type=int, default=5)
    ap.add_argument("--precision", default="float64", choices=["float32", "float64"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sweep", type=float, nargs="*", default=[0.9, 0.95, 1.05, 1.1], help="focal factors for intr")
    args = ap.parse_args()
    for st in args.stage:
        {"plan": stage_plan, "intr": stage_intr, "video": stage_video, "gt": stage_gt, "meta": stage_meta,
         "check-keyframes": stage_check_keyframes, "smplfit": stage_smplfit}[st](args)


if __name__ == "__main__":
    main()
