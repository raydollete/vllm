#!/usr/bin/env bash
# Export the current patch stack as a numbered series of .patch files.
#
# Usage:
#   scripts/fork/export-patches.sh [output-dir]   (default: patches/)
#
# Produces a portable, reviewable snapshot of every commit between
# base/current and the stack tip. Useful for backup, diff review, or applying
# the stack elsewhere with `git am`.
#
# See FORK.md for the full workflow.
set -euo pipefail

BASE_REF="${BASE_REF:-base/current}"
STACK_TIP="${STACK_TIP:-deploy}"
OUT_DIR="${1:-patches}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; exit 1; }

git rev-parse --verify --quiet "${BASE_REF}" >/dev/null \
  || die "${BASE_REF} does not exist."
git rev-parse --verify --quiet "refs/heads/${STACK_TIP}" >/dev/null \
  || die "Stack tip branch '${STACK_TIP}' does not exist."

count="$(git rev-list --count "${BASE_REF}..${STACK_TIP}")"
[[ "${count}" -gt 0 ]] || die "No commits between ${BASE_REF} and ${STACK_TIP}."

# Start clean so dropped patches don't linger as stale files.
if [[ -d "${OUT_DIR}" ]]; then
  rm -f "${OUT_DIR}"/*.patch
fi
mkdir -p "${OUT_DIR}"

say "Exporting ${count} patch(es) from ${BASE_REF}..${STACK_TIP} into ${OUT_DIR}/"
git format-patch --output-directory "${OUT_DIR}" --numbered \
  "${BASE_REF}..${STACK_TIP}" >/dev/null

say "Done. Re-apply elsewhere with:  git am ${OUT_DIR}/*.patch"
