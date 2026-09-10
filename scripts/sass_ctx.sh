#!/usr/bin/env bash
# Print the SASS instructions around given line numbers (encoding comments stripped).
#   sass_ctx.sh <file.sass> <line> [<line> ...]      (CTX=3 lines each side by default)
set -u
f=$1; shift
ctx=${CTX:-3}
for l in "$@"; do
  echo "== $l"
  sed -n "$((l - ctx)),$((l + ctx))p" "$f" | sed -E 's#/\*[^*]*\*/##g; s/ +/ /g' | grep -v '^ *$'
done
