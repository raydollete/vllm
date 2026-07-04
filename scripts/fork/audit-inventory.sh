#!/usr/bin/env bash
# Audit the FORK.md patch inventory against live upstream PR state.
#
# For every `#NNNN` PR referenced in a `patch/NN-*` inventory row, query
# `gh pr view N --json state` and flag any carried patch whose tracked PR has
# left the OPEN state:
#   - MERGED  -> DROP   (upstream absorbed it; delete branch + row, then rebase)
#   - CLOSED  -> REVIEW (closed unmerged; verify the capability landed elsewhere,
#                        or update/drop the row)
# Exits non-zero if any offender is found, printing the offending rows — so CI
# catches a tracked PR merging before it silently rots in the stack.
#
# A row may list more than one PR (a stacked pair like #40898+#43200, or a rival
# impl). This audit reports per-PR state; the AND/OR drop timing in the row's
# note is the maintainer's call — the script only surfaces the state change.
#
# Env:
#   FORK_MD          path to the inventory file (default: <repo-root>/FORK.md)
#   UPSTREAM_REPO    GitHub repo for `gh pr view` (default: vllm-project/vllm)
#
# See FORK.md ("Donor-branch triage") for the surrounding procedure.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_MD="${FORK_MD:-${here}/../../FORK.md}"
UPSTREAM_REPO="${UPSTREAM_REPO:-vllm-project/vllm}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
bad() { printf '\033[1;31mXX\033[0m %s\n' "$*" >&2; }

if [[ ! -f "${FORK_MD}" ]]; then
  warn "Inventory file not found: ${FORK_MD}"
  exit 2
fi
if ! command -v gh >/dev/null 2>&1; then
  warn "GitHub CLI 'gh' not found; cannot query PR state."
  exit 2
fi

say "Auditing inventory in ${FORK_MD} against ${UPSTREAM_REPO}"

declare -A STATE_CACHE=()  # pr number -> state (query each PR once)
offenders=0
checked=0

# Inventory rows are exactly the table lines whose first cell is a `patch/NN-*`
# branch in backticks.
while IFS= read -r row; do
  branch="$(sed -E 's/^\| *`([^`]+)`.*/\1/' <<<"${row}")"
  # All `#NNNN` tokens in the row are tracked-PR references (only the tracking
  # cell carries them; Purpose never does).
  mapfile -t prs < <(grep -oE '#[0-9]+' <<<"${row}" | tr -d '#' | sort -u)
  [[ ${#prs[@]} -eq 0 ]] && continue  # e.g. patch/00 (never upstreamed)

  for pr in "${prs[@]}"; do
    state="${STATE_CACHE[$pr]:-}"
    if [[ -z "${state}" ]]; then
      if ! state="$(gh pr view "${pr}" --repo "${UPSTREAM_REPO}" \
                      --json state -q .state 2>/dev/null)"; then
        warn "  ${branch}: #${pr} — could not query (network/auth?); skipped"
        continue
      fi
      STATE_CACHE[$pr]="${state}"
    fi
    checked=$((checked + 1))
    case "${state}" in
      OPEN)
        ;;
      MERGED)
        bad "  ${branch}: #${pr} is MERGED -> DROP (delete branch + row, rebase)"
        offenders=$((offenders + 1))
        ;;
      CLOSED)
        bad "  ${branch}: #${pr} is CLOSED -> REVIEW (verify covered elsewhere / update row)"
        offenders=$((offenders + 1))
        ;;
      *)
        warn "  ${branch}: #${pr} has unexpected state '${state}'"
        offenders=$((offenders + 1))
        ;;
    esac
  done
done < <(grep -E '^\| *`patch/[0-9]' "${FORK_MD}")

if [[ ${offenders} -gt 0 ]]; then
  bad "Inventory audit FAILED: ${offenders} tracked PR(s) no longer OPEN (of ${checked} checked)."
  exit 1
fi

say "Inventory audit OK: all ${checked} tracked PR(s) still OPEN."
