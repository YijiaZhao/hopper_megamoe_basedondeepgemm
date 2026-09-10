#!/usr/bin/env bash
# Where do the spills land? For each cubin: SASS line numbers of SETMAXREG (role
# boundaries), the GMMA line range (math loop), and every STL/LDL (spill) line.
#   regbudget_sass.sh <cubin> [<cubin> ...]
set -u
for c in "$@"; do
  s="${c%.cubin}.sass"
  cuobjdump -sass "$c" > "$s" || continue
  echo "== $(basename "$c")  lines=$(wc -l < "$s") HGMMA=$(grep -c HGMMA "$s") QGMMA=$(grep -c QGMMA "$s") IGMMA=$(grep -c IGMMA "$s")"
  echo "   SETMAXREG: $(grep -n -E 'SETMAXREG' "$s" | sed -E 's/^([0-9]+):.*SETMAXREG[^ ]* *(.*);.*/\1(\2)/' | tr '\n' ' ')"
  echo "   GMMA lines: $(grep -n -E 'HGMMA|QGMMA|IGMMA' "$s" | cut -d: -f1 | sed -n '1p;$p' | tr '\n' ' ')"
  echo "   STL/LDL lines: $(grep -n -E ' (STL|LDL)' "$s" | cut -d: -f1 | tr '\n' ' ')"
  # spill lines between the first and last GMMA => in/around the math loop
  first=$(grep -n -E 'HGMMA|QGMMA|IGMMA' "$s" | head -1 | cut -d: -f1); last=$(grep -n -E 'HGMMA|QGMMA|IGMMA' "$s" | tail -1 | cut -d: -f1)
  if [ -n "$first" ]; then
    echo "   spills inside GMMA range: $(grep -n -E ' (STL|LDL)' "$s" | cut -d: -f1 | awk -v a="$first" -v b="$last" '$1>=a && $1<=b' | wc -l)"
  fi
done
