#!/bin/bash
# One table for every finished run: what the FoM believed, and what the device did.
#
# The two match columns are the point. Both are the SAME functional -- the KL
# geometric mean the optimizer scores -- evaluated on two different observables:
#
#   FoM match   from the normal-incidence probe's momentum decomposition at the
#               EML plane, which is what the optimizer actually maximizes
#   PP match    the same target re-scored on the MEASURED far field, produced by
#               OLED_angular_target.npz + optimization_target_match
#
# Nothing was writing that npz until 2026-08-20, so the postprocess had never once
# run the comparison and every manifest carried an empty optimization_target_match.
# With it in place the two numbers sit side by side automatically, which is the
# whole question this project has been circling: does the score track the device.
#
# CONVERGED is not decoration. Before the bottom-air margin was fixed, every
# forward and adjoint run diverged -- fields growing to ~1e5 instead of decaying --
# while the driver logged nothing at all. Two optimizations and a day of analysis
# were built on those fields. A run with div>0 here is not a result.
#
# usage: summarize.sh [tag ...]      (default: every directory under results/)
set -u
PIPE_DIR=/home/eidl/Junseo/oled_pipeline
PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python

tags=("$@")
[ ${#tags[@]} -eq 0 ] && mapfile -t tags < <(find "$PIPE_DIR/results" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | sort)
[ ${#tags[@]} -eq 0 ] && { echo "no runs under $PIPE_DIR/results"; exit 0; }

printf "  %-18s %-9s %8s %8s %8s %8s %8s %7s  %s\n" \
       TAG STACK FoM_match PP_match LEE 45/0 shape iters CONVERGED
for t in "${tags[@]}"; do
    out="$PIPE_DIR/results/$t"
    log="$out/${t}.log"
    # run_one.sh records the path while the run is still under Ongoing/; a
    # finished run has since been moved to Done/ or Failed/, so follow it by name
    # rather than reporting "unknown" for every run that actually completed.
    rundir=$(cat "$PIPE_DIR/state/${t}.rundir" 2>/dev/null)
    if [ -n "${rundir:-}" ] && [ ! -d "$rundir" ]; then
        base=$(basename "$rundir")
        for where in Done Failed Ongoing; do
            [ -d "/home/eidl/Lumerical_data/$where/$base" ] && {
                rundir="/home/eidl/Lumerical_data/$where/$base"; break; }
        done
    fi
    man="$out/OLED_postprocess_manifest.json"
    [ -f "$man" ] || man="$rundir/A/OLED_postprocess_manifest.json"
    # The optimizer's own last score, and how many iterations it got.
    fm=$(grep -ohE "match=[0-9.]+" "$log" "$rundir/OLED_rec_output.log" 2>/dev/null | tail -1 | cut -d= -f2)
    it=$(grep -ohcE "^\[rec\]" "$rundir/OLED_rec_output.log" 2>/dev/null | tail -1)
    kind=$(grep -ohE "\[stack\] microcavity/[a-z]+" "$log" "$rundir/OLED_rec_output.log" 2>/dev/null |
           tail -1 | sed 's|.*/||')
    conv=$(grep -c "\[FDTD\] converged" "$rundir/OLED_rec_output.log" 2>/dev/null)
    div=$(grep -c "\[FDTD\] DIVERGED" "$rundir/OLED_rec_output.log" 2>/dev/null)
    "$PY" - "$man" "$t" "${kind:-?}" "${fm:-}" "${it:-0}" "${conv:-0}" "${div:-0}" <<'PYEOF'
import json, sys
man, tag, kind, fm, it, conv, div = sys.argv[1:8]
def g(d, *p):
    for k in p:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None: return None
    return d
try:
    m = json.load(open(man))
except Exception:
    m = {}
pp   = g(m, "optimization_target_match", "match")
lee  = g(m, "ensemble_lee")
r45  = (g(m, "angular_metrics", "ratios_to_zero") or [None, None])[-1]
shp  = g(m, "angular_metrics", "shape_score")
f = lambda v, w=8, p=4: (f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}")
flag = "DIVERGED" if int(div) else (f"{conv} ok" if int(conv) else "unknown")
print(f"  {tag:<18} {kind:<9} {f(float(fm) if fm else None)} {f(pp)} {f(lee)} "
      f"{f(r45)} {f(shp)} {int(it):>7}  {flag}")
PYEOF
done
echo
echo "  FoM_match = probe momentum decomposition;  PP_match = same target on the MEASURED far field."
echo "  They should track each other. Where they do not, the FoM is not describing the device."
