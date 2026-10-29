"""Copy an intrinsics directory with the focal length multiplied by a constant (focal sweep).

    python scale_intrinsics.py --src WORK/intrinsics --out WORK/intrinsics_f095 --scale 0.95

The body is then re-run with run_sam3d_body.py --intr-dir OUT, refined with refine_body_v5.py, and
assembled with assemble_episode.py --object-depth-scale SCALE so the objects follow the body in depth.
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.src.glob("episode_*")):
        if f.suffix != ".json":
            shutil.copy2(f, a.out / f.name)
            continue
        d = json.loads(f.read_text())
        for k in ("fx", "fy"):
            d[k] = d[k] * a.scale
        if "fx_frames" in d:
            d["fx_frames"] = [v * a.scale for v in d["fx_frames"]]
        d["note"] = f"focal x{a.scale} of {f.name}"
        (a.out / f.name).write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
