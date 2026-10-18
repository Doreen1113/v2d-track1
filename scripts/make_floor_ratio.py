"""Per-episode MoGe/body scale from the floor plane.

refine_body_sequence.py fits the floor in MoGe depth and in the body frame (from foot contacts) and
stores their offset ratio as floor_moge_over_body in work/body_v2/metrics/episode_XXXXXX.json.
This collects it into the ratio JSON read by finalize_objects_v2.py and assemble_episode.py.

    python make_floor_ratio.py
"""
import argparse
import json
from pathlib import Path

WORK = Path("/work/doreen071/v2d/work")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", type=Path, default=WORK / "body_v2")
    ap.add_argument("--out", type=Path, default=WORK / "moge_to_body_ratio_floor.json")
    args = ap.parse_args()
    ratio = {str(e): json.loads((args.body / f"metrics/episode_{e:06d}.json").read_text())
             ["floor_moge_over_body"] for e in range(30)}
    out = args.out
    out.write_text(json.dumps({"note": f"MoGe/body scale from the floor plane ({args.body.name} metrics "
                                       "floor_moge_over_body)", "ratio": ratio}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
