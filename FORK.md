# Fork maintenance — patch stack over upstream vLLM

This fork carries local changes as a **rebaseable patch stack** on top of
`vllm-project/vllm`. This document is the doctrine and the live index of what
we carry.

## Remotes

| Remote     | URL                                    | Role                    |
| ---------- | -------------------------------------- | ----------------------- |
| `origin`   | `git@github.com:raydollete/vllm.git`   | our fork (push here)    |
| `upstream` | `github.com/vllm-project/vllm.git`     | vLLM (never push)       |

## Branch model

```
base/current (tag → any upstream commit: release tag, upstream/main, SHA, or PR head)
  └─ patch/01-<topic>
       └─ patch/02-<topic>
            └─ patch/03-<topic>        # topmost patch = deployable tip
                 └─ deploy             # OPTIONAL: only if it has its OWN commit(s)
```

- **`main`** — a pristine mirror of `upstream/main`. It **never** holds local
  patches; it is only ever fast-forwarded. `git diff main...<tip>` is the full
  diff of everything we carry.
- **`base/current`** — a movable tag marking the exact upstream commit the
  stack is currently rebased onto. It can point at *anything* upstream: a
  release tag, an `upstream/main` commit, a bare SHA, or a fetched PR head.
- **`patch/NN-<topic>`** — one logical, self-contained change each. This is the
  unit that becomes an upstream PR. Numbered to make dependency order explicit;
  independent patches may sit in any order.
- **Stack tip** — the branch you build/run and pass to `rebase.sh` as
  `STACK_TIP`. Pick ONE of:
  - **the topmost `patch/NN` branch** (default, simplest) — use this when all
    your changes are patches. Set `STACK_TIP=patch/NN-top`.
  - **a separate `deploy` branch** — only if it carries deploy-only commits
    that never go upstream (private config, local defaults). It **must sit
    strictly above** the topmost patch (have at least one commit of its own).

> **Footgun — coincident tip.** `git rebase --update-refs` advances every
> branch strictly *inside* the rebased range but **silently skips** a branch
> that points at the *same commit* as the tip. So a `deploy` that merely
> duplicates the top patch (same SHA) would be left on stale commits after a
> rebase. `rebase.sh` refuses this case with a clear error. Either give
> `deploy` its own commit, or don't keep a separate `deploy` and set
> `STACK_TIP` to the topmost patch.

## Patch inventory

Keep this table current. When a patch merges upstream, delete its branch and
remove its row — the next rebase absorbs it via the new base.

| Branch                              | Purpose                                              | Intent    | Upstream tracking → drop when it merges |
| ----------------------------------- | ---------------------------------------------------- | --------- | --------------------------------------- |
| `patch/00-fork-tooling`             | Fork machinery (FORK.md + scripts)                   | permanent | — (never upstreamed) |
| `patch/01-precommit-grammar-filter` | Spec-decode × reasoning: drop maskless post-boundary bonus tokens before they enter the response | tracking | **#43424** — this *is* that PR (Adam Moisa); rival impl **#36138**. Do **not** open our own. Drop when either merges. |
| `patch/02-xgrammar-allow-propertynames` | Allow JSON schemas with `propertyNames` through the xgrammar backend | tracking | **#42904** — different (better) impl: adds real support vs. our blocklist removal. Local override only; drop & use native support when it merges. |
| `patch/03-mtp-prefix-mamba-accuracy` | Exclude final partially-accepted block from mamba cache-hit lookup under MTP/eagle | tracking | **#43650** (zack041). Drop when it merges. |
| `patch/04-uniform-batch-prefills` | `_is_uniform_decode` also verifies no request is still prefilling | tracking | **#39945** (Benjamin Chislett). Drop when it merges. |
| `patch/05-thinking-budget-reasoning-end` | Detect reasoning end via parser's intrinsic marker under `thinking_token_budget` | tracking | **#43210** (Sebastian Schoennenbeck). Conflict-resolved against main's incremental scan; re-check on rebase. Drop when it merges. |
| `patch/06-streaming-stop-tool-parsing` | Flush tool-parser-buffered text as content when generation stops mid-parse | tracking | **#42213** (dtnguyen). Classic parsers only; engine parsers no-op safely. Drop when it merges. |
| `patch/07-mamba-block-table-fusion` | Triton-fused mamba align-mode block-table gather | tracking | **#38020** (Jialin Ouyang) — approved upstream, likely merges soon. Drop when it merges. |
| `patch/08-gdn-inplace-ssm-state` | In-place SSM state access in GDN chunk prefill (Qwen3.5/Qwen3-Next/OLMo-Hybrid) | tracking | **#41824** (kermit). Drop when it merges. |

Provenance note: patches 03–08 were triaged out of `repne/vllm` branch
`v15-dflash` (2026-07-03). Everything DFlash-specific from that branch is
already in upstream main (#43081, #46104, #43733, #42692, …) — no DFlash
patches are carried.

Intent legend:
- **permanent** — carried indefinitely; no upstream equivalent expected.
- **pending** — our own change we intend to submit upstream; drops out once *our* PR merges.
- **tracking** — an upstream PR already covers this (ours or someone else's).
  We do **not** open a duplicate (AGENTS.md); we carry the patch only until that
  PR lands, then delete the branch + row and rebase so upstream's version takes
  over. The "Upstream tracking" cell names the PR to watch.

## One-time setup

Already applied in this clone (recorded here for new clones):

```bash
git config rerere.enabled true      # remember conflict resolutions
git config rerere.autoupdate true
git config rebase.updateRefs true   # keep the whole stack aligned on rebase

git fetch upstream --tags
git checkout main && git merge --ff-only upstream/main
git tag base/current <any-ref>      # v0.9.2, upstream/main, a SHA — your call

# Seed the first patch off the base, then chain further patches off each other:
git checkout -b patch/01-<topic> base/current
# ...commit...
git checkout -b patch/02-<topic> patch/01-<topic>
# ...commit...

# Then choose your stack tip (see "Branch model"):
#  (a) simplest — build from the topmost patch, no separate deploy branch:
export FORK_STACK_TIP=patch/02-<topic>        # pass to rebase.sh via STACK_TIP
#  (b) or a deploy branch WITH its own commit(s) (deploy-only config):
git checkout -b deploy patch/02-<topic>
# ...commit deploy-only changes...  (must be >=1 commit so deploy != top patch)
```

Requires git ≥ 2.38 for `--update-refs`. `rebase.sh` and `export-patches.sh`
default `STACK_TIP` to `deploy`; override per-invocation with
`STACK_TIP=patch/NN-top scripts/fork/rebase.sh ...` if you chose option (a).

## Everyday workflow

```bash
scripts/fork/sync.sh                # fetch upstream, ff main, report drift
```

Rebase the whole stack onto a new base (the base is an argument — pick any
target each time):

```bash
scripts/fork/rebase.sh v0.9.3           # onto a release tag …
scripts/fork/rebase.sh upstream/main    # … or bleeding upstream …
scripts/fork/rebase.sh pr/45723         # … or an unmerged upstream PR head
```

Resolve conflicts once per patch (`git rebase --continue` between each);
`rerere` auto-replays any resolution it has seen before. On success the script
retags `base/current` to the new base and prints the push commands:

```bash
git push origin --force-with-lease 'patch/*' deploy
git push origin -f base/current
```

### Trying an unmerged upstream PR without committing to it

Rebase onto the PR head, build, evaluate — then rebase back onto your real
base. The PR's commits simply fall out of the `--onto` range, leaving the patch
stack intact:

```bash
scripts/fork/rebase.sh pr/45723     # test against the PR
# ...build + evaluate...
scripts/fork/rebase.sh v0.9.3       # back to your permanent base
```

### Exporting the stack

```bash
scripts/fork/export-patches.sh      # writes patches/NNNN-*.patch (git am-able)
```

## Building the drop-in container

While every patch is pure Python, ship as an **overlay image**: official
`vllm/vllm-openai` base + only the stack's changed files copied over the
installed package. Seconds to build, byte-identical kernels, same
entrypoint/env/ports as the official image.

```bash
# 1. Official images exist per release — move the stack onto that release:
STACK_TIP=<tip> scripts/fork/rebase.sh v0.24.0

# 2. Build (file list derived from git, staged from the tip commit):
STACK_TIP=<tip> scripts/fork/build-image.sh v0.24.0
#    → vllm-openai-fork:v0.24.0-<tipsha>, labeled with the stack SHA
#    (org.opencontainers.image.revision). DRY_RUN=1 to preview.

# 3. Run anywhere the official image ran, same args:
docker run --gpus all ... vllm-openai-fork:v0.24.0-<tipsha> ...
```

Safety rails built into `build-image.sh` / `docker/Dockerfile.fork`:

- **Base alignment enforced** — refuses to build unless `base/current` sits
  exactly on the version tag matching the base image (no silent version mixing).
- **Pure-Python enforced** — refuses if the stack touches `csrc/`, `*.cu`,
  `setup.py`, etc.; that requires the full source build (`docker/Dockerfile`).
- Overlay is staged from the **tip commit** via `git archive`, never from the
  (possibly dirty / different-branch) working tree.
- In-image checks: installed `vllm.__version__` must match the expected base,
  every overlaid file must already exist in the image, stale `__pycache__` is
  purged, and the patched modules must import.

Day-to-day you can keep developing against `upstream/main`; hop to the release
tag only at packaging time (`rerere` makes the round-trip cheap).

## Build & verify after a rebase

Per `AGENTS.md`:

```bash
# Python-only patches:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
# If any patch touches C/C++/CUDA, do a full build instead:
uv pip install -e . --torch-backend=auto

# Smoke test:
.venv/bin/python -c "import vllm; print(vllm.__version__)"
# plus any test covering a patched area, e.g.:
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

## Invariants (things that should always hold)

- `git diff --stat main...upstream/main` is empty after `sync.sh` — the mirror
  has not drifted (no local patch leaked onto `main`).
- The stack tip is either the topmost `patch/NN` branch, or a `deploy` branch
  that sits strictly above it (own commit) — never a `deploy` that coincides
  with the top patch. `rebase.sh` enforces this.
- `base/current` points at a real upstream commit, never at one of our patches.
