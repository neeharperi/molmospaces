# reference/

Frozen, committed reference data for the DROID-leaderboard reproduction effort
(see `plans/BENCHMARK.md` and `docs/eval_reproduction.md`).

- `leaderboard_snapshot.csv` — a point-in-time export of the public MolmoSpaces leaderboard,
  filtered to the DROID embodiment. **Captured 2026-08-17.** The leaderboard page itself is a
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
  variant is confirmed. MolmoAct2/DreamZero rows still need adding once those policies'
  slugs are found (try `molmoact2`/`dreamzero` against the same task slugs). Columns:
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
  versions even if upstream defaults drift later. **Not yet captured** — created during the
  harness environment setup step in `docs/eval_reproduction.md`.
