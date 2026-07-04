#!/usr/bin/env bash
# Build a drop-in replacement for the official vllm/vllm-openai image,
# overlaying this fork's Python-only patch stack onto it.
#
# Usage:
#   scripts/fork/build-image.sh <vllm-version> [image-tag]
#
#   <vllm-version>  release the OFFICIAL image exists for, e.g. v0.24.0.
#                   The stack base (base/current) must sit exactly on this
#                   tag — rebase first if not:
#                     STACK_TIP=<tip> scripts/fork/rebase.sh <vllm-version>
#   [image-tag]     output tag (default: vllm:<build-date>, e.g. vllm:2026-07-03;
#                   the stack tip SHA stays traceable via the image label
#                   org.opencontainers.image.revision)
#
# Env:
#   STACK_TIP   stack tip branch (default: deploy)
#   BASE_REF    base pointer (default: base/current)
#   DRY_RUN=1   stage + print the docker command without building
#
# The overlay file list is derived from git (base..tip), never hand-kept.
# Refuses to build if the stack touches anything outside pure-Python vllm/
# sources — that requires the full source build (docker/Dockerfile) instead.
#
# See FORK.md for the full workflow.
set -euo pipefail

BASE_REF="${BASE_REF:-base/current}"
STACK_TIP="${STACK_TIP:-deploy}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { warn "$*"; exit 1; }

[[ $# -ge 1 && $# -le 2 ]] || die "Usage: $0 <vllm-version> [image-tag]"
version="$1"

repo_root="$(git rev-parse --show-toplevel)"
git rev-parse --verify --quiet "${BASE_REF}" >/dev/null \
  || die "${BASE_REF} does not exist."
git rev-parse --verify --quiet "refs/heads/${STACK_TIP}" >/dev/null \
  || die "Stack tip branch '${STACK_TIP}' does not exist (set STACK_TIP=...)."

image_tag="${2:-vllm:$(date +%F)}"

# --- Base alignment: the overlay diff must be against the SAME version as the
# official base image, or we'd silently mix versions.
git rev-parse --verify --quiet "${version}^{commit}" >/dev/null \
  || die "'${version}' is not a known tag/ref. Run scripts/fork/sync.sh first."
if [[ "$(git rev-parse "${BASE_REF}^{commit}")" != "$(git rev-parse "${version}^{commit}")" ]]; then
  warn "Stack base ${BASE_REF} ($(git rev-parse --short "${BASE_REF}")) is NOT ${version}."
  warn "The overlay would mix versions. Rebase the stack first:"
  die "  STACK_TIP=${STACK_TIP} scripts/fork/rebase.sh ${version}"
fi

# --- Derive the overlay file list from git; refuse anything the overlay
# can't faithfully represent.
mapfile -t changed < <(git diff --name-only "${BASE_REF}" "refs/heads/${STACK_TIP}")
[[ ${#changed[@]} -gt 0 ]] || die "No changes between ${BASE_REF} and ${STACK_TIP}."

overlay_files=()
for f in "${changed[@]}"; do
  case "$f" in
    vllm/*.py) overlay_files+=("$f") ;;
    tests/*|docs/*|FORK.md|scripts/fork/*|docker/Dockerfile.fork) ;;  # not shipped
    csrc/*|*.cu|*.cpp|*.c|*.h|*.cuh|CMakeLists.txt|setup.py|cmake/*)
      die "Stack changes '$f' — compiled code. Overlay image is invalid; use the full source build (docker/Dockerfile)." ;;
    *) die "Stack changes '$f' — outside vllm/*.py and not a known non-shipped path. Extend build-image.sh deliberately if this should be overlaid." ;;
  esac
done
[[ ${#overlay_files[@]} -gt 0 ]] || die "Stack has no vllm/*.py changes to overlay."

say "Overlaying ${#overlay_files[@]} file(s) from ${BASE_REF}..${STACK_TIP}:"
printf '    %s\n' "${overlay_files[@]}"

# --- Stage a clean build context from the TIP COMMIT (not the working tree,
# which may be on another branch or dirty).
ctx="$(mktemp -d)"
trap 'rm -rf "${ctx}"' EXIT
mkdir -p "${ctx}/overlay"
git archive "refs/heads/${STACK_TIP}" "${overlay_files[@]}" | tar -x -C "${ctx}/overlay"

expected_version="${version#v}"

build_cmd=(docker build
  -f "${repo_root}/docker/Dockerfile.fork"
  --build-arg "VLLM_VERSION=${version}"
  --build-arg "EXPECTED_VLLM_VERSION=${expected_version}"
  --build-arg "STACK_TIP_SHA=$(git rev-parse "refs/heads/${STACK_TIP}")"
  -t "${image_tag}"
  "${ctx}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  say "DRY_RUN — staged context at ${ctx} (removed on exit). Would run:"
  printf '    %q ' "${build_cmd[@]}"; echo
  find "${ctx}" -type f | sed 's/^/    /'
  exit 0
fi

say "Building ${image_tag} on vllm/vllm-openai:${version}"
"${build_cmd[@]}"
say "Done. Drop-in replacement for vllm/vllm-openai:${version}:"
say "  docker run --gpus all ... ${image_tag} <same args as official image>"
