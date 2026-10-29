"""Write an intrinsics directory (same format as work/intrinsics) with a new focal per episode.

Either one focal for every episode (--fx) or one focal per physical camera (--per-camera JSON
{"back_stereo_camera_left": 850.0, ...}). The principal point stays at the image centre. The
per-frame MoGe lists are rescaled by the same factor (as scale_intrinsics.py does) and the
low-resolution point maps are copied unchanged.

    python fcal_write_intrinsics.py --out WORK/fcal_intrinsics_global --fx 849.9 --note "..."
    python fcal_write_intrinsics.py --out WORK/fcal_intrinsics_percam --per-camera percam.json --note "..."
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path("/work/doreen071/v2d")
META = ROOT / "data/track_1/meta/episodes_metadata.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "work/intrinsics")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--per-camera", type=Path, default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    assert (a.fx is None) != (a.per_camera is None), "give exactly one of --fx / --per-camera"
    cam_of = {json.loads(l)["episode_index"]: json.loads(l)["camera"] for l in open(META)}
    per_cam = json.loads(a.per_camera.read_text()) if a.per_camera else None
    a.out.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.src.glob("episode_*")):
        if f.suffix != ".json":
            shutil.copy2(f, a.out / f.name)
            continue
        d = json.loads(f.read_text())
        new = a.fx if a.fx is not None else float(per_cam[cam_of[d["episode"]]])
        s = new / d["fx"]
        d["fx"], d["fy"] = new, new
        d["cx"], d["cy"] = d["width"] / 2, d["height"] / 2
        for k in ("fx_frames", "fy_frames"):
            if k in d:
                d[k] = [v * s for v in d[k]]
        d["note"] = f"{a.note} (camera {cam_of[d['episode']]}; x{s:.4f} of {a.src.name}/{f.name})"
        (a.out / f.name).write_text(json.dumps(d, indent=1))
        print(f.name, cam_of[d["episode"]], round(new, 2))


if __name__ == "__main__":
    main()
