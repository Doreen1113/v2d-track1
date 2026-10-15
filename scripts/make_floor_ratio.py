"""Per-episode MoGe/body scale from the floor plane.

refine_body_sequence.py fits the floor in MoGe depth and in the body frame (from foot contacts) and
stores their offset ratio as floor_moge_over_body in work/body_v2/metrics/episode_XXXXXX.json.
This collects it into the ratio JSON read by finalize_objects_v2.py and assemble_episode.py.

    python make_floor_ratio.py
"""
import json
from pathlib import Path

WORK = Path("/work/doreen071/v2d/work")


def main():
    ratio = {str(e): json.loads((WORK / f"body_v2/metrics/episode_{e:06d}.json").read_text())
             ["floor_moge_over_body"] for e in range(30)}
    out = WORK / "moge_to_body_ratio_floor.json"
    out.write_text(json.dumps({"note": "MoGe/body scale from the floor plane (body_v2 metrics "
                                       "floor_moge_over_body)", "ratio": ratio}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
