# Subsampled benchmarks

One manifest per draw made by `scripts/benchmarks/subsample_benchmark.py`. The drawn
`benchmark.json` itself lives under `$MLSPACES_ASSETS_DIR/benchmarks/smoke/<name>/` with the
rest of the assets, not here -- it is derived data, and the manifest is what makes it
reproducible: source path, source `benchmark.json` hash, seed, and the category mix actually
achieved beside the source's.

Regenerate one with the seed and counts its manifest records:

```bash
python scripts/benchmarks/subsample_benchmark.py \
    --source $MLSPACES_ASSETS_DIR/benchmarks/<source path> \
    --episodes 50 --houses 10 --seed 42 --out-name Close-v1-smoke50
```

A small cell is not comparable to the leaderboard on its own -- 50 episodes carry a ±14 pp
interval, and the draw drops the rarest categories entirely. Read
`scripts/category_mix_check.py` beside any number produced from one: it reweights the
leaderboard's own per-category rates onto the mix that was actually evaluated, which is the
comparison a small cell can support.
