"""Summarise the per-frame focal estimates (fcal_moge.py, fcal_learned.py, fcal_vp.py) per physical camera.

For each method variant and camera it reports, over the sampled frames (person present) and over the
person-free background plates separately:
  median fx, robust spread (1.4826 * MAD) across frames, 16-84 percentile range,
  spread of the per-episode medians (scene dependence), and the scale vs the global 807.66 px focal.
Writes <out>/summary.json and prints markdown tables.

    python fcal_summarize.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/doreen071/v2d")
F_GLOBAL = 807.6573429107666
CAMS = ["back_stereo_camera_left", "front_stereo_camera_left", "left_stereo_camera_left", "right_stereo_camera_left"]
SHORT = {c: c.split("_")[0] for c in CAMS}


def rows(path):
    if not path.exists():
        return []
    out = []
    for l in path.read_text().splitlines():
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out


def variants(res_dir):
    """Yield (variant_name, list of dict(episode, camera, frame, fx, cx, cy, k1))."""
    V = defaultdict(list)
    for r in rows(res_dir / "moge.jsonl"):
        for key, suffix in (("fx", ""), ("fx_masked", "_masked")):
            V[r["method"] + suffix].append({**r, "fx": r[key], "cx": None, "cy": None})
        if r.get("pp_fit"):
            f, cx, cy = r["pp_fit"]
            V[r["method"] + "_ppfit"].append({**r, "fx": f, "cx": cx, "cy": cy})
    for name in ("unidepth", "depthpro", "geocalib", "geocalib_d", "anycalib", "anycalib_g"):
        for r in rows(res_dir / f"{name}.jsonl"):
            if "fx" in r and np.isfinite(r["fx"]):
                V[name].append(r)
    for tag in ("", "_upper", "_lower"):
        for r in rows(res_dir / f"vp{tag}.jsonl"):
            if "fx" in r:
                V[r["method"] + tag].append(r)
    # first VP pass (8 frames/episode): its vp_f rows are valid (only its k1 variant had a sign bug)
    for r in rows(res_dir / "vp_prefix.jsonl"):
        if "fx" in r and r["method"] == "vp_f":
            V["vp_f"].append(r)
    # de-duplicate (episode, frame) in case a job was appended twice
    for k in V:
        seen, keep = set(), []
        for r in V[k]:
            key = (r["episode"], str(r["frame"]))
            if key not in seen:
                seen.add(key)
                keep.append(r)
        V[k] = keep
    return V


def stats(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    med = float(np.median(v))
    return {"n": int(len(v)), "median": med, "mad_std": float(1.4826 * np.median(np.abs(v - med))),
            "p16": float(np.percentile(v, 16)), "p84": float(np.percentile(v, 84)), "scale": med / F_GLOBAL}


def summarise(V):
    S = {}
    for name, rs in sorted(V.items()):
        S[name] = {}
        for cam in CAMS + ["ALL"]:
            sel = [r for r in rs if cam == "ALL" or r["camera"] == cam]
            fr = [r for r in sel if str(r["frame"]) != "bg"]
            bg = [r for r in sel if str(r["frame"]) == "bg"]
            ep_med = defaultdict(list)
            for r in fr:
                ep_med[r["episode"]].append(r["fx"])
            ep_m = {e: float(np.median(v)) for e, v in ep_med.items()}
            S[name][cam] = {
                "frames": stats([r["fx"] for r in fr]),
                "plates": stats([r["fx"] for r in bg]),
                "episode_medians": ep_m,
                "between_episode_std": float(np.std(list(ep_m.values()))) if len(ep_m) > 1 else None,
                "cx": stats([r.get("cx") for r in fr + bg]),
                "cy": stats([r.get("cy") for r in fr + bg]),
                "k1": stats([r.get("k1") for r in fr + bg]) if any("k1" in r for r in sel) else None,
                "fy_over_fx": stats([r["fy"] / r["fx"] for r in fr + bg if r.get("fy")]),
            }
    return S


# one representative per independent method family (person masked where the method allows)
FAMILIES = {"VP (lines, geometric)": "vp_f", "VP ceiling/upper lines only": "vp_f_upper", "MoGe-2 full res": "moge_t9027_masked",
            "MoGe-2 3600 tok (as work/intrinsics)": "moge_t3600_masked", "AnyCalib": "anycalib",
            "GeoCalib": "geocalib", "Depth Pro": "depthpro", "UniDepthV2": "unidepth"}


def consensus(S):
    out = {}
    for cam in CAMS + ["ALL"]:
        vals = {}
        for fam, key in FAMILIES.items():
            if key in S and S[key][cam]["frames"] is not None:
                vals[fam] = S[key][cam]["frames"]["median"]
        fam_vals = dict(vals)
        if "VP (lines, geometric)" in fam_vals and "VP ceiling/upper lines only" in fam_vals:
            # the two VP variants are one family: use their mean in the family median
            vp = (fam_vals.pop("VP (lines, geometric)") + fam_vals.pop("VP ceiling/upper lines only")) / 2
            fam_vals["VP (mean of all/upper)"] = vp
        v = np.array(list(fam_vals.values()))
        out[cam] = {"per_family": vals, "median_of_families": float(np.median(v)) if len(v) else None,
                    "scale": float(np.median(v) / F_GLOBAL) if len(v) else None,
                    "min_scale": float(v.min() / F_GLOBAL) if len(v) else None,
                    "max_scale": float(v.max() / F_GLOBAL) if len(v) else None}
    return out


def fmt(s, key="median"):
    return "-" if s is None else f"{s[key]:.1f}"


def print_tables(S):
    print("\n## fx per camera: frames median [robust std] (plates median) | scale vs 807.66\n")
    print("| method | " + " | ".join(SHORT[c] for c in CAMS) + " | ALL |")
    print("|---|" + "---|" * (len(CAMS) + 1))
    for name, d in S.items():
        cells = []
        for cam in CAMS + ["ALL"]:
            fr, bg = d[cam]["frames"], d[cam]["plates"]
            ref = fr or bg
            if ref is None:
                cells.append("-")
                continue
            cells.append(f"{fmt(fr)} [{fmt(fr, 'mad_std')}] ({fmt(bg)}) x{ref['scale']:.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n## between-episode std of per-episode medians (px)\n")
    for name, d in S.items():
        print(name, {SHORT.get(c, c): (None if d[c]["between_episode_std"] is None else round(d[c]["between_episode_std"], 1)) for c in CAMS})
    print("\n## principal point (median over frames+plates) where estimated\n")
    for name, d in S.items():
        if d["ALL"]["cx"] is not None:
            print(name, {SHORT.get(c, c): (fmt(d[c]["cx"]), fmt(d[c]["cy"]), fmt(d[c]["cx"], "mad_std"), fmt(d[c]["cy"], "mad_std")) for c in CAMS + ["ALL"]})
    print("\n## distortion k1 where estimated\n")
    for name, d in S.items():
        if d["ALL"]["k1"] is not None:
            print(name, {SHORT.get(c, c): round(d[c]["k1"]["median"], 4) for c in CAMS + ["ALL"] if d[c]["k1"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, default=ROOT / "work/fcal_results")
    a = ap.parse_args()
    V = variants(a.res)
    S = summarise(V)
    C = consensus(S)
    (a.res / "summary.json").write_text(json.dumps({"variants": S, "consensus": C}, indent=1))
    print_tables(S)
    print("\n## consensus (median of method families, frames)\n")
    for cam, c in C.items():
        if c["median_of_families"] is None:
            continue
        print(SHORT.get(cam, cam), f"{c['median_of_families']:.1f} px  x{c['scale']:.3f}  range x{c['min_scale']:.3f}-x{c['max_scale']:.3f}",
              {k: round(v, 1) for k, v in c["per_family"].items()})


if __name__ == "__main__":
    main()
