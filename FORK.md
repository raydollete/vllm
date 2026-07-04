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

```text
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

| Branch | Purpose | Intent | Upstream tracking → drop when it merges |
| --- | --- | --- | --- |
| `patch/00-fork-tooling` | Fork machinery (FORK.md + scripts) | permanent | — (never upstreamed) |
| `patch/01-precommit-grammar-filter` | Spec-decode × reasoning: drop maskless post-boundary bonus tokens before they enter the response | tracking | **#43424** — this *is* that PR (Adam Moisa); rival impl **#36138**. Do **not** open our own. Drop when either merges. |
| `patch/02-xgrammar-allow-propertynames` | Allow JSON schemas with `propertyNames` through the xgrammar backend | tracking | **#42904** — different (better) impl: adds real support vs. our blocklist removal. Local override only; drop & use native support when it merges. |
| `patch/03-mtp-prefix-mamba-accuracy` | Exclude final partially-accepted block from mamba cache-hit lookup under MTP/eagle | tracking | **#43650** (zack041). Drop when it merges. |
| `patch/04-uniform-batch-prefills` | `_is_uniform_decode` also verifies no request is still prefilling | tracking | **#39945** (Benjamin Chislett). Drop when it merges. |
| `patch/05-thinking-budget-reasoning-end` | Detect reasoning end via parser's intrinsic marker under `thinking_token_budget` | tracking | **#43210** (Sebastian Schoennenbeck). Conflict-resolved against main's incremental scan; re-check on rebase. Drop when it merges. |
| `patch/06-streaming-stop-tool-parsing` | Flush tool-parser-buffered text as content when generation stops mid-parse | tracking | **#42213** (dtnguyen). Classic parsers only; engine parsers (incl. prod `qwen3_xml`) no-op safely here — their end-of-stream flush is `_flush_engine_parsers` (`vllm/parser/abstract_parser.py:916-957`), not this patch. Drop when it merges. |
| `patch/07-mamba-block-table-fusion` | Triton-fused mamba align-mode block-table gather | tracking | **#38020** (Jialin Ouyang) — approved upstream, likely merges soon. Drop when it merges. |
| `patch/08-gdn-inplace-ssm-state` | In-place SSM state access in GDN chunk prefill (Qwen3.5/Qwen3-Next/OLMo-Hybrid) | tracking | **#41824** (kermit). Drop when it merges. |
| `patch/09-dflash-swa` | DFlash mixed sliding/full `layer_types` draft support + FlashInfer per-`window_left` metadata groups (restores v15 DFlash-8) | tracking | **#40898** (jianc99) + **#43200** (gq112, stacked on it). Ported as ONE patch from `refs/pull/43200/head`. Drop when both merge; if #40898 lands first, shrinks to the #43200 delta at the next rebase. |
| `patch/10-greedy-rejection-shortcut` | Rejection-sampler shortcut for all-greedy speculative sampling | tracking | **#38007** (zzaebok). Drop when it merges. |
| `patch/11-gdn-inproj-fusion` | Fuse Qwen3.5 GDN `in_proj_ba` into 6-way `in_proj` MergedColumnParallelLinear | tracking | **#41457** (jhsmith409). On the critical path — prod target Qwen3.6-27B-FP8 is itself GDN-hybrid. Drop when it merges. |
| `patch/12-gdn-qkv-compile` | Re-enable torch.compile for `rearrange_mixed_qkv` in GDN linear attention | tracking | **#42241** (tjtanaa). Drop when it merges. |
| `patch/13-mamba-postprocess-skip` | Skip mamba postprocess kernel when no block boundary can be crossed | tracking | donor commit `df2c23014` (Tony), adapts **#42574** (mamingyuan-nv) atop patch/07. Drop when #42574 merges. |

Provenance note (v15-dflash triage 2026-07-03; **corrected 2026-07-04**):
patches 03–13 were triaged out of `repne/vllm` branch `v15-dflash`. Capability
assertions (cite the SHA + the capability + the config that exercises it, and an
explicit "NOT covered" line for any sibling behind an open PR):

- **Uniform-`use_swa` DFlash SWA IS upstream** — #46104 (`9969466a5`, merged
  2026-07-01) adds `_resolve_layer_attention`, covering the MiMo shape
  (`layer_types=None` + `use_swa`, and all-`full_attention`). Exercised by
  MiMo-family drafters (e.g. `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash`).
- **Mixed per-layer `layer_types` DFlash SWA was NOT upstream** — it is gated
  behind the still-open **#40898** (+ FlashInfer **#43200**), which #46104's own
  raise at `qwen3_dflash.py:93` explicitly defers to. The original 2026-07-03
  triage read #46104's *title* as full coverage and dropped #40898, crashing the
  production DFlash-8 drafter (`z-lab/Qwen3.6-27B-DFlash`, `layer_types =
  4×sliding_attention@2048 + 1×full_attention`) at engine start. Restored as
  **`patch/09-dflash-swa`**.
- The four still-open v15 perf changes dropped in the same triage are restored as
  **patches 10–13** (see rows above).
- Full 22-PR + 3-donor-commit disposition — including the four "fixed on main"
  supersession claims **re-verified against this base** (#42875/#43074 N/A,
  #39615 covered by the parser engine, #43349 optimization present) and the
  open reasoning-count remainder (#45787/#45802, enforcement unaffected) — is
  recorded in `openspec/changes/restore-v15-dflash-swa-gaps/` (proposal.md
  §Blast-Radius 11 + tasks.md §7). See the **Donor-branch triage** procedure
  below for how this class of miss is now prevented.

Intent legend:

- **permanent** — carried indefinitely; no upstream equivalent expected.
- **pending** — our own change we intend to submit upstream; drops out once *our* PR merges.
- **tracking** — an upstream PR already covers this (ours or someone else's).
  We do **not** open a duplicate (AGENTS.md); we carry the patch only until that
  PR lands, then delete the branch + row and rebase so upstream's version takes
  over. The "Upstream tracking" cell names the PR to watch.

## Donor-branch triage

When adopting patches from a **donor branch** (a third-party fork such as
`repne/vllm:v15-dflash` that itself merged unmerged upstream PRs), classify every
carried change **mechanically**. Reading PR titles is exactly how the 2026-07-03
triage concluded "DFlash is already upstream" and dropped the open #40898 (see the
provenance note above). These steps are the guardrail; none may be skipped.

1. **Enumerate every PR-head merge** the donor branch carries:
   ```bash
   git log --grep 'refs/pull' <donor>/v15-dflash
   ```
2. **Get each PR's real state — never infer it from the title:**
   ```bash
   gh pr view N --repo vllm-project/vllm --json state,mergedAt
   ```
   - `MERGED` → absorbed once the base is new enough; drop.
   - `OPEN` / `CLOSED` → requires an explicit **carry-or-drop** disposition with a
     one-line rationale. A `CLOSED` PR is **not** automatically "covered" — it may
     be superseded, or closed unfixed. Verify via step 3.
3. **Supersession = read the superseding diff, not the title.** "Fixed on main" /
   "covered by #X" holds only if #X's *diff* delivers the same capability for
   *our* config shape. Record the merged SHA, the capability, and the config that
   exercises it. If a sibling capability is still behind an open PR, write an
   explicit "NOT covered" line (the #46104-covers-uniform-SWA / #40898-still-open
   trap).
4. **Sweep for donor-authored commits** not attached to any PR head:
   ```bash
   git rev-list --no-merges <donor>/v15-dflash ^upstream/main \
     $(for n in <every-PR-number>; do echo "^refs/remotes/upstream/pr-$n"; done)
   ```
   Each survivor is a donor-original change — carry it as a tracking patch against
   the upstream PR it adapts (e.g. `patch/13` ← `df2c23014` → #42574) or drop it
   with a reason.

Every PR must end in exactly one bucket — merged-and-dropped, carried (with the
PR to watch), or dropped-with-reason. None left unclassified. The inventory-audit
script (`scripts/fork/audit-inventory.sh`) then fails CI if a carried patch's
tracked PR later leaves the OPEN state.

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
#    → vllm:<build-date> (e.g. vllm:2026-07-03), labeled with the stack SHA
#    (org.opencontainers.image.revision). DRY_RUN=1 to preview.
#    Pass a second arg to override the tag.

# 3. Run anywhere the official image ran, same args:
docker run --gpus all ... vllm:2026-07-03 ...
```

### Build provenance — what is in an image?

Every build is documented **in git and in the image itself**, never in a
hand-maintained ledger:

- `build-image.sh` creates an annotated tag `build/<image-tag-suffix>`
  (e.g. `build/2026-07-03`) at the exact tip commit. The annotation is the
  manifest: base image + the patch commits included.
  `git show build/2026-07-03` answers "what went into `vllm:2026-07-03`?"
  The tag also keeps the built commit alive after later rebases move the
  patch branches (otherwise the image's revision label would dangle).
- The image carries labels: `org.opencontainers.image.revision` (tip SHA)
  and `org.opencontainers.image.base.name` (exact official base image).
  `docker inspect <image>` recovers both without the repo.

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

### Post-rebase smoke-boot matrix (required before a base is declared good)

A rebase + overlay build is **not "good"** until **every production serve
profile** boots to a serving state on the new image. Load-only boots are cheap
and would have caught the DFlash-8 mixed-`layer_types` crash in seconds. Minimum
matrix (run after `build-image.sh`, before deploying):

| Profile | Config (delta from the common Qwen3.6-27B-FP8 target) | Boot gate |
| ------- | ----------------------------------------------------- | --------- |
| **P1 — MTP-3** | `speculative_config={method: mtp, num_speculative_tokens: 3}` | `/v1/models` responds |
| **P2 — DFlash-8** | `speculative_config={method: dflash, num_speculative_tokens: 8, model: z-lab/Qwen3.6-27B-DFlash, attention_backend: FLASHINFER, draft_sample_method: greedy}`, `kv_cache_dtype=fp8_e4m3`, `max_model_len=262144`, `block_size=32`, prefix caching, `cudagraph_mode=FULL_AND_PIECEWISE` | `/v1/models` responds; **no** `NotImplementedError`/causal-assert from the DFlash path |

If any profile fails to reach serving state, **reject the base (or the offending
patch)** and record the result against the build provenance tag (`build/<date>`).
Deploy only after every profile is green.

## Invariants (things that should always hold)

- `git diff --stat main...upstream/main` is empty after `sync.sh` — the mirror
  has not drifted (no local patch leaked onto `main`).
- The stack tip is either the topmost `patch/NN` branch, or a `deploy` branch
  that sits strictly above it (own commit) — never a `deploy` that coincides
  with the top patch. `rebase.sh` enforces this.
- `base/current` points at a real upstream commit, never at one of our patches.
