# reference/

Frozen, committed reference data for the DROID-leaderboard reproduction effort
(see `plans/BENCHMARK.md` and `docs/eval_reproduction.md`).

- `leaderboard_snapshot.csv` — a point-in-time export of the public MolmoSpaces leaderboard,
  filtered to the DROID embodiment. **Captured 2026-08-17 through 2026-08-19**, one policy
  block at a time as each integration landed -- so `snapshot_date` varies by row and is
  per-policy, not file-wide. The leaderboard page itself is a
  client-rendered JS app (not fetchable via plain HTTP), but the user found via browser
  DevTools that it loads static per-(task, policy) CSVs at
  `https://molmospaces.allen.ai/benchmark/data/<task_slug>/<policy_slug>.csv` (e.g.
  `mb_pick_classic/pi05.csv`) -- these ARE plain-fetchable. Task slugs found:
  `mb_pick_msproc`=Pick-v1.5, `mb_pick_classic`=Pick-v2-classic,
  `mb_pick_filament`=Pick-v2-filament, `mb_pick_rand_cam`=Pick-v2-RandCam, `mb_pnp`=PnP-v2,
  `mb_pnp_next_to`=PnP-NextTo-v2, `mb_pnp_color`=PnP-Color-v2, `ms_open`=Open-v1,
  `ms_close`=Close-v1; policy slugs `pi05`=pi05_droid, `tiptop`=tiptop. Each file's
  `OVERALL` row gives `oracle_rate_pct` (present in every file, unlike the plain/at-end
  success rate which only some files include) -- used here as `success_rate`/
  `metric=oracle`, consistent with BENCHMARK.md's working assumption. No dedicated
  combined-aggregate file was found (`mb_combined`, `all_combined` etc. all 404); the
  `MolmoBot Combined` row here is computed by pooling oracle_successes/total across the 7
  Group B task files, matching BENCHMARK.md's own methodology.
  **tiptop, captured 2026-08-18: `ms_open`/`ms_close` (the two Group A, bench-v1 tasks)
  return the SPA's HTML shell, not a CSV** -- confirmed by content inspection (`<!DOCTYPE
  html>`, not `policy,category,...`), same signature as a 404 on this site, unlike the 7
  Group B slugs which all returned real CSVs. TiPToP genuinely has no leaderboard entry for
  Open-v1/Close-v1; there is nothing to compare our results against for those two tasks for
  this policy, and that's expected, not a fetch bug.
  **cosmos, captured 2026-08-18: policy slug is `cosmos` (bare), not `cosmos_edge`/
  `cosmos_nano`/`cosmos3_edge`/`cosmos3_nano`** -- all of the latter 404 (HTML shell) at
  every task slug; only `cosmos` returns real CSVs, and it has entries for all 9 tasks
  (unlike tiptop, Open-v1/Close-v1 both present). **The leaderboard does not say which of
  the two DROID checkpoints (Cosmos3-Edge-Policy-DROID, 4B, or Cosmos3-Nano-Policy-DROID,
  16B) produced this row.** Rather than guess, the same numbers are duplicated here under
  both `cosmos_edge` and `cosmos_nano` (this repo's two registered policy names for the two
  checkpoints) so `compare_to_leaderboard.py` -- which looks up `runs/<policy>/<task>/` by
  exact policy-column string -- compares each of our two real runs against this one
  ambiguous reference, rather than silently matching neither. Treat a PASS/FAIL for either
  as weaker evidence than the other policies' unambiguous comparisons until/unless the
  variant is confirmed.
  **molmoact2, captured 2026-08-19**: slug `molmoact2`, entries for all 9 tasks.
  **dreamzero, captured 2026-08-19: slug `dreamzero`, and it has GROUP A ONLY.** `ms_open`
  (246/990 = 24.85%) and `ms_close` (552/915 = 60.33%) return real CSVs; all 7 Group B slugs
  return the SPA's HTML shell, the same 404 signature TiPToP shows on `ms_open`/`ms_close`.
  So DreamZero has no MolmoBot-Combined entry and no Group B cell to compare against -- that
  is a fact about the leaderboard, not a fetch failure, and it means DreamZero contributes
  nothing to the Group B campaign. Note this is the mirror image of TiPToP's gap: between the
  two of them, every one of the 9 tasks has at least one policy with no reference number.
  **pi0, captured 2026-08-28: slug `pi0`, and it has GROUP A ONLY** -- the same shape as
  DreamZero. `ms_open` (110/1000 = 11.00%) and `ms_close` (486/915 = 53.11%) return real CSVs
  with the usual `oracle_successes`/`oracle_rate_pct` columns, so these are directly
  comparable to every other row here; all 7 Group B slugs return the SPA's HTML shell. Added
  when pi0-DROID was brought into the campaign alongside pi0.5. Note a sibling slug `pi0_fast`
  also resolves, with the same Group-A-only coverage (`ms_open` 111/1001 = 11.09%, `ms_close`
  353/915 = 38.58%); it is NOT captured here because pi0-FAST is a different checkpoint
  (`pi0_fast_droid_jointpos`) and is not currently in `scripts/eval_common.py`'s POLICIES.
  Both `pi0`'s run_paths are `/home/orayyan/projects/molmospaces/eval_output/new_results/
  {open,close}/pi0` -- the same author and pipeline as the pi05 Group A rows, and a different
  one from the pi05 Group B rows, which is the heterogeneity noted further down.
  Every policy's rows are now captured; nothing further is outstanding here.

  **`n_episodes` is not measured the same way for every policy block, and it matters for any
  n-weighted comparison.** `pi05_droid`'s counts are the real per-file totals from each
  leaderboard CSV (997, 987, 985, 804, 541, 322, 961, 1000, 915; Combined 5597), so they vary
  per task the way real evaluation sets do. The `tiptop`/`molmoact2_droid`/`cosmos_*` blocks
  are uniformly 1000 for the 8 non-Close tasks, 915 for Close-v1 and exactly 7000 for
  Combined -- i.e. the "fall back to the benchmark JSON's own episode count" substitution this
  README requires be noted below, applied without being noted at the time. Recorded now.
  `dreamzero`'s two rows are real per-file totals (990, 915). Columns:
  `task,policy,success_rate,n_episodes,embodiment,metric,snapshot_date`
  - `task`: one of this repo's 9 task names (see `scripts/eval_common.py`'s `TASKS`), plus one
    row per policy named `MolmoBot Combined` for the Group B pooled aggregate.
  - `success_rate`: percentage (0-100), the leaderboard's point estimate.
  - `n_episodes`: episode count behind that point estimate. If the leaderboard doesn't expose
    this, fall back to the benchmark JSON's own episode count and note the substitution here.
  - `metric`: `at-end` or `oracle` — whichever success-condition definition the leaderboard
    entry used. See `docs/eval_reproduction.md` for why we default to assuming `oracle`.
  - `snapshot_date`: the date the export was taken (`YYYY-MM-DD`).

- `pinned_assets_<date>.json` — a frozen dump of `DATA_TYPE_TO_SOURCE_TO_VERSION` from
  `molmo_spaces.molmo_spaces_constants`, taken right after the first asset install. Referenced
  by `$MLSPACES_PINNED_ASSETS_FILE` so every eval run in this campaign uses the same asset
  versions even if upstream defaults drift later. **Captured: `pinned_assets_20260816.json`**
  (`molmospaces-bench-v1@20260408`, `molmospaces-bench-v2@20260415`). Every eval run must have
  `MLSPACES_PINNED_ASSETS_FILE` pointing at it and `MLSPACES_FORCE_INSTALL=False`;
  `scripts/run_full_matrix.sh` exports both, and `scripts/check_provenance.py` re-hashes the
  file against the sha256 each run recorded, so a silent edit fails the provenance check.
