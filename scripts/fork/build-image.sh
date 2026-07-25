#!/usr/bin/env bash
# Build a drop-in replacement for the official vllm/vllm-openai image,
# overlaying this fork's Python-only patch stack onto it.
#
# Usage:
#   scripts/fork/build-image.sh <base> [image-tag]
#
#   <base>          which OFFICIAL image to overlay onto. Two forms:
#                     v0.24.0                     a release tag
#                     nightly-<40-hex> | <40-hex> a nightly, published daily as
#                                                 vllm/vllm-openai:nightly-<sha>
#                   Either way the stack base (base/current) must sit exactly on
#                   that commit — rebase first if not:
#                     STACK_TIP=<tip> scripts/fork/rebase.sh <base>
#   [image-tag]     output tag (default: vllm:<build-date>, e.g. vllm:2026-07-03;
#                   the stack tip SHA stays traceable via the image label
#                   org.opencontainers.image.revision)
#
# Env:
#   STACK_TIP          stack tip branch (default: deploy)
#   BASE_REF           base pointer (default: base/current)
#   DRY_RUN=1          stage + print the docker command without building
#   SKIP_IMPORT_CHECK=1  skip the post-build `docker run --gpus all` import check
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

[[ $# -ge 1 && $# -le 2 ]] || die "Usage: $0 <base> [image-tag]"
base_arg="$1"

repo_root="$(git rev-parse --show-toplevel)"
git rev-parse --verify --quiet "${BASE_REF}" >/dev/null \
  || die "${BASE_REF} does not exist."
git rev-parse --verify --quiet "refs/heads/${STACK_TIP}" >/dev/null \
  || die "Stack tip branch '${STACK_TIP}' does not exist (set STACK_TIP=...)."

image_tag="${2:-vllm:$(date +%F)}"

# --- Resolve the base kind. A nightly is pinned by commit sha (vLLM publishes
# vllm/vllm-openai:nightly-<full-sha> daily); a release is pinned by tag.
if [[ "${base_arg}" =~ ^(nightly-)?([0-9a-f]{40})$ ]]; then
  base_kind="nightly"
  base_sha="${BASH_REMATCH[2]}"
  image_ref="nightly-${base_sha}"
  git_ref="${base_sha}"
else
  base_kind="release"
  image_ref="${base_arg}"
  git_ref="${base_arg}"
fi

# --- Base alignment: the overlay diff must be against the SAME commit as the
# official base image, or we'd silently mix versions.
git rev-parse --verify --quiet "${git_ref}^{commit}" >/dev/null \
  || die "'${base_arg}' is not a known tag/ref. Run scripts/fork/sync.sh first."
if [[ "$(git rev-parse "${BASE_REF}^{commit}")" != "$(git rev-parse "${git_ref}^{commit}")" ]]; then
  warn "Stack base ${BASE_REF} ($(git rev-parse --short "${BASE_REF}")) is NOT ${base_arg}."
  warn "The overlay would mix versions. Rebase the stack first:"
  die "  STACK_TIP=${STACK_TIP} scripts/fork/rebase.sh ${base_arg}"
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

# Files the stack *adds* are exempt from the "must already exist in the image"
# rail (that rail is the base-mismatch canary for *modified* files). They get
# the inverse check instead — see Dockerfile.fork.
mapfile -t added < <(git diff --diff-filter=A --name-only "${BASE_REF}" "refs/heads/${STACK_TIP}")
new_files=""
for f in "${overlay_files[@]}"; do
  for a in "${added[@]}"; do
    [[ "$f" == "$a" ]] && new_files+="${f} " && break
  done
done

say "Overlaying ${#overlay_files[@]} file(s) from ${BASE_REF}..${STACK_TIP}:"
for f in "${overlay_files[@]}"; do
  case " ${new_files} " in
    *" ${f} "*) printf '    %s  (new)\n' "$f" ;;
    *)          printf '    %s\n' "$f" ;;
  esac
done

# --- Stage a clean build context from the TIP COMMIT (not the working tree,
# which may be on another branch or dirty).
ctx="$(mktemp -d)"
trap 'rm -rf "${ctx}"' EXIT
mkdir -p "${ctx}/overlay"
git archive "refs/heads/${STACK_TIP}" "${overlay_files[@]}" | tar -x -C "${ctx}/overlay"

# A release image reports vllm.__version__ == "<version without v>...", a
# nightly reports e.g. "0.23.1rc1.dev748+g4080263bb" — assert on whichever
# substring actually identifies the base.
if [[ "${base_kind}" == "nightly" ]]; then
  expected_version="+g$(git rev-parse --short=9 "${base_sha}")"
  version_match="contains"
else
  expected_version="${base_arg#v}"
  version_match="prefix"
fi

build_cmd=(docker build
  -f "${repo_root}/docker/Dockerfile.fork"
  --build-arg "VLLM_VERSION=${image_ref}"
  --build-arg "EXPECTED_VLLM_VERSION=${expected_version}"
  --build-arg "VERSION_MATCH=${version_match}"
  --build-arg "NEW_FILES=${new_files% }"
  --build-arg "STACK_TIP_SHA=$(git rev-parse "refs/heads/${STACK_TIP}")"
  -t "${image_tag}"
  "${ctx}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  say "DRY_RUN — staged context at ${ctx} (removed on exit). Would run:"
  printf '    %q ' "${build_cmd[@]}"; echo
  find "${ctx}" -type f | sed 's/^/    /'
  exit 0
fi

say "Building ${image_tag} on vllm/vllm-openai:${image_ref}"
"${build_cmd[@]}"

# The build itself never imports vllm: `docker build` has no GPU, and nightly
# images fail `import vllm` without one. Do the real import check here, where a
# GPU is available.
if [[ "${SKIP_IMPORT_CHECK:-0}" != "1" ]]; then
  say "Verifying patched modules import (docker run --gpus all)"
  # --entrypoint: the official image's entrypoint is `vllm serve`, which would
  # swallow these args as CLI flags.
  # The module list is derived from the overlay, so the check actually exercises
  # what this build changed rather than a fixed pair of modules.
  import_probe="$(printf '%s\n' "${overlay_files[@]}" \
    | sed -n 's#^vllm/\(.*\)\.py$#\1#p' | sed 's#/__init__$##; s#/#.#g' \
    | sed 's/^/vllm./' | sort -u | paste -sd, -)"
  if ! docker run --rm --gpus all --entrypoint python3 "${image_tag}" -c \
      "import importlib, vllm; [importlib.import_module(m) for m in '${import_probe}'.split(',')]; print('import check OK:', vllm.__version__)"; then
    die "Post-build import check FAILED for ${image_tag}. Image left in place for inspection."
  fi
else
  warn "SKIP_IMPORT_CHECK=1 — patched modules were NOT import-checked."
fi

# Record the build in git: an annotated build/<tag-suffix> tag pinning the
# exact tip commit (the image's revision label dangles once the stack is
# rebased; the tag keeps it alive and diffable). Annotation = the manifest.
build_tag="build/${image_tag#*:}"
git tag -f -a "${build_tag}" "refs/heads/${STACK_TIP}" -m "$(
  printf 'image: %s\nbase:  vllm/vllm-openai:%s\nbase commit: %s\n\npatches:\n' \
    "${image_tag}" "${image_ref}" "$(git rev-parse "${BASE_REF}^{commit}")"
  git log --oneline --reverse "${BASE_REF}..refs/heads/${STACK_TIP}" | sed 's/^/  /'
)"
say "Tagged ${build_tag} -> $(git rev-parse --short "refs/heads/${STACK_TIP}") (git show ${build_tag} for the manifest)"

say "Done. Drop-in replacement for vllm/vllm-openai:${image_ref}:"
say "  docker run --gpus all ... ${image_tag} <same args as official image>"
