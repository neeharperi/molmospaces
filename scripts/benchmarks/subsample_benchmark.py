#!/usr/bin/env python
"""Draw a small, category-matched benchmark out of a full one.

    python scripts/benchmarks/subsample_benchmark.py \
        --source $MLSPACES_ASSETS_DIR/benchmarks/.../FrankaCloseDataGenConfig_20260123_json_benchmark \
        --episodes 50 --houses 10 --out-name Close-v1-smoke50

Why this exists rather than `--max_episodes`. That flag selects whole *houses* off the
front of the raw list, so a 50-episode Open-v1 run covers 5 of 13 object categories and
misses two of the highest-scoring ones -- which alone accounts for the entire apparent
leaderboard gap for two policies (`scripts/category_mix_check.py`). A truncated cell is
therefore not a small version of the benchmark, it is a different benchmark, and its rate
is not comparable to anything. This draws a subset whose category mix tracks the source's
instead, writes down what it actually achieved, and leaves the residual skew for
`category_mix_check.py` to reweight.

**Houses are the unit of parallelism, so they are a parameter.** `JsonEvalRunner` makes one
work item per house and each work item loads that house's scene once -- the dominant cost of
a short cell. Fewer houses means fewer scene loads and less parallelism; more houses means
the opposite. `--houses` is the knob, and the default is deliberately not "all of them".

The category of an episode is derived with the *same* two functions `eval_to_csv.py` uses to
label its per-category CSV rows, imported rather than reimplemented, so the manifest this
writes and the results CSV a run produces name their categories identically. Anything else
would make the reweighting compare two different partitions.

Output, beside the source's own layout so `eval.py` can point a `TaskSpec` straight at it:

    <out>/benchmark.json           the drawn episodes, in the source's schema
    <out>/benchmark_metadata.json  the source's, with the recomputed counts
    <out>/subsample_manifest.json  seed, source hash, and achieved vs source mix
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_to_csv import _simplify  # noqa: E402  -- the labels the results CSV will use


def category_of(episode: dict) -> str:
    """The episode's object category, as `eval_to_csv.py` will label it.

    `_extract_object_name` cannot be reused directly: it reads the `object_name` field out
    of a *recorded* trajectory's `obs_scene` blob, which does not exist until a run has
    happened. The benchmark carries the same string under `task.pickup_obj_name`, so the
    cleaning is repeated on that field and the simplification is imported.
    """
    raw = (episode.get("task") or {}).get("pickup_obj_name") or "unknown"
    cleaned = "".join(c if c.isalpha() else " " for c in raw).strip()
    return _simplify(cleaned.split()[0] if cleaned else "unknown")


def raw_category_of(episode: dict) -> str:
    """The object category keyed the way the SOURCE benchmark_metadata.json keys it.

    `category_of` above returns `_simplify`'d labels, because those are what the results CSV
    and category_mix_check.py use. The source metadata uses the raw lowercase name instead
    ('stand', 'chestofdrawers'), and writing our labels under the source's field name made
    one key mean two different partitions across the two files.
    """
    raw = (episode.get("task") or {}).get("pickup_obj_name") or "unknown"
    cleaned = "".join(c if c.isalpha() else " " for c in raw).strip()
    return (cleaned.split()[0] if cleaned else "unknown").lower()


def largest_remainder(counts: dict, total: int) -> dict:
    """Apportion ``total`` across ``counts`` in proportion, keeping the sum exact.

    Plain rounding overshoots or undershoots the total; largest-remainder does not, because
    the leftovers go to the categories rounded down hardest.

    **It does not rescue the rare categories, and the manifest says so.** On Close-v1 at
    total=50 the exact quotas run 17.1, 16.3, 11.5, 2.4, 2.3, 0.17, 0.11, 0.11, 0.055; the
    floors sum to 48, so two leftovers go to the two largest fractions and the four rarest
    of nine still draw zero -- see ``category_counts`` against ``category_share_source`` in
    reference/smoke_benchmarks/Close-v1-smoke50.json. What this buys over truncation is that
    the categories that *do* appear appear in the source's proportions, which is what
    category_mix_check.py reweights against. Covering the tail needs a larger draw, not a
    better apportionment.
    """
    pool = sum(counts.values())
    if pool == 0 or total <= 0:
        return {key: 0 for key in counts}
    exact = {key: total * value / pool for key, value in counts.items()}
    quota = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(quota.values())
    for key, _ in sorted(exact.items(), key=lambda kv: (kv[1] - int(kv[1]), counts[kv[0]]), reverse=True):
        if remainder <= 0:
            break
        quota[key] += 1
        remainder -= 1
    return quota


def choose_houses(episodes: list, wanted: int, rng: random.Random) -> list:
    """The ``wanted`` houses whose pooled category mix best tracks the whole benchmark.

    Greedy, because the exact version is a subset-selection problem and the greedy answer is
    already within a percentage point here. At each step it takes the house that most
    reduces total variation distance between the selected pool's mix and the source's.
    """
    by_house = collections.defaultdict(list)
    for episode in episodes:
        by_house[episode["house_index"]].append(episode)
    houses = sorted(by_house)
    if wanted >= len(houses):
        return houses

    source = collections.Counter(category_of(episode) for episode in episodes)
    source_share = {key: value / len(episodes) for key, value in source.items()}

    def distance(pool: collections.Counter) -> float:
        size = sum(pool.values()) or 1
        keys = set(source_share) | set(pool)
        return sum(abs(pool.get(key, 0) / size - source_share.get(key, 0.0)) for key in keys) / 2

    chosen: list = []
    pool: collections.Counter = collections.Counter()
    rng.shuffle(houses)  # so ties do not always break toward the lowest house index
    while len(chosen) < wanted:
        best = min(
            (house for house in houses if house not in chosen),
            key=lambda house: distance(pool + collections.Counter(category_of(e) for e in by_house[house])),
        )
        chosen.append(best)
        pool += collections.Counter(category_of(e) for e in by_house[best])
    return sorted(chosen)


def draw(episodes: list, total: int, houses: int, seed: int) -> tuple:
    """Pick ``total`` episodes from ``houses`` houses, tracking the source's category mix."""
    rng = random.Random(seed)
    keep_houses = set(choose_houses(episodes, houses, rng))
    pool = [episode for episode in episodes if episode["house_index"] in keep_houses]

    quota = largest_remainder(collections.Counter(category_of(e) for e in pool), min(total, len(pool)))

    by_category = collections.defaultdict(list)
    for episode in pool:
        by_category[category_of(episode)].append(episode)

    drawn = []
    for category, want in quota.items():
        candidates = by_category[category]
        # Round-robin over houses inside a category, so a quota of 6 spread over 4 houses
        # takes from four scenes rather than six episodes of one.
        per_house = collections.defaultdict(list)
        for episode in candidates:
            per_house[episode["house_index"]].append(episode)
        for bucket in per_house.values():
            rng.shuffle(bucket)
        order = sorted(per_house)
        rng.shuffle(order)
        while want > 0 and any(per_house[house] for house in order):
            for house in order:
                if want <= 0:
                    break
                if per_house[house]:
                    drawn.append(per_house[house].pop())
                    want -= 1

    # Stable output order: by house, then by the source's own ordering, so two runs of this
    # script produce byte-identical files and a cell's episode order is reproducible.
    position = {id(episode): index for index, episode in enumerate(episodes)}
    drawn.sort(key=lambda e: (e["house_index"], position[id(e)]))
    return drawn, sorted(keep_houses)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=pathlib.Path, required=True, help="a benchmark directory")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--houses",
        type=int,
        default=10,
        help="how many houses to draw from; the unit of parallelism and of scene-load cost",
    )
    parser.add_argument("--seed", type=int, default=42, help="eval.py also hardcodes 42")
    parser.add_argument("--out-name", required=True, help="directory name under --out-root")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a draw already at --out-name whose seed/size/source differs. Refused by "
        "default: any cell run against it recorded that benchmark.json's hash",
    )
    parser.add_argument(
        "--out-root",
        type=pathlib.Path,
        default=None,
        help="default $MLSPACES_ASSETS_DIR/benchmarks/smoke",
    )
    args = parser.parse_args()

    source_json = args.source / "benchmark.json"
    if not source_json.exists():
        raise SystemExit(f"{source_json} does not exist; --source wants a benchmark directory")
    raw = source_json.read_bytes()
    episodes = json.loads(raw)

    out_root = args.out_root
    if out_root is None:
        import os

        assets = os.environ.get("MLSPACES_ASSETS_DIR")
        if not assets:
            raise SystemExit("set MLSPACES_ASSETS_DIR, or pass --out-root")
        out_root = pathlib.Path(assets) / "benchmarks" / "smoke"
    out = out_root / args.out_name
    existing_manifest = out / "subsample_manifest.json"
    source_sha = hashlib.sha256(raw).hexdigest()
    if existing_manifest.exists() and not args.overwrite:
        prior = json.loads(existing_manifest.read_text())
        request = ("seed", "requested_episodes", "requested_houses", "source_sha256")
        mine = (args.seed, args.episodes, args.houses, source_sha)
        if tuple(prior.get(key) for key in request) != mine:
            # A cell's provenance records benchmark_sha256 of this directory's benchmark.json
            # (scripts/eval.py). Replacing the file under the same name leaves that hash
            # matching nothing on disk, so the cell's number can no longer be traced to an
            # episode list -- the anecdote eval.py's hash exists to prevent.
            raise SystemExit(
                f"{out} already holds a different draw.\n"
                + "".join(
                    f"  {key:<20} there {prior.get(key)!r:<24} here {value!r}\n"
                    for key, value in zip(request, mine, strict=True)
                )
                + "Any cell already run against it recorded that benchmark's hash. Use a new\n"
                "--out-name, or --overwrite if you are certain nothing depends on this one."
            )
    out.mkdir(parents=True, exist_ok=True)

    drawn, houses = draw(episodes, args.episodes, args.houses, args.seed)

    source_mix = collections.Counter(category_of(e) for e in episodes)
    drawn_mix = collections.Counter(category_of(e) for e in drawn)
    per_house = collections.Counter(e["house_index"] for e in drawn)

    (out / "benchmark.json").write_text(json.dumps(drawn, indent=1))

    metadata = {}
    source_metadata = args.source / "benchmark_metadata.json"
    if source_metadata.exists():
        metadata = json.loads(source_metadata.read_text())
    # Every per-episode count in the source describes the source, so leaving any of them
    # alongside num_episodes=50 makes the file contradict itself. Recompute the ones the
    # drawn episodes determine; drop the one they do not.
    raw_mix = collections.Counter(raw_category_of(e) for e in drawn)
    metadata.update(
        description=f"{args.episodes}-episode category-matched subsample of {args.source.name}",
        num_episodes=len(drawn),
        num_houses=len(houses),
        # Keyed as the source keys it. Our _simplify'd view is in subsample_manifest.json.
        object_category_counts={key: raw_mix[key] for key in sorted(raw_mix)},
        task_cls_counts=dict(sorted(collections.Counter(
            (e.get("task") or {}).get("task_cls", "unknown") for e in drawn).items())),
        robot_counts=dict(sorted(collections.Counter(
            (e.get("robot") or {}).get("robot_name", "unknown") for e in drawn).items())),
        house_counts=dict(sorted(per_house.items())),
    )
    # Episode length is a property of a recorded trajectory, not of the spec, so a draw
    # cannot recompute it. Carrying the source's would state 915 episodes' statistics over 50.
    metadata.pop("episode_length_stats", None)
    (out / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2))

    manifest = {
        "source": str(args.source),
        "source_sha256": source_sha,
        "source_episodes": len(episodes),
        "seed": args.seed,
        "requested_episodes": args.episodes,
        "requested_houses": args.houses,
        "drawn_episodes": len(drawn),
        "houses": houses,
        "episodes_per_house": {str(k): per_house[k] for k in sorted(per_house)},
        "max_episodes_in_one_house": max(per_house.values()) if per_house else 0,
        "category_counts": {key: drawn_mix[key] for key in sorted(drawn_mix)},
        "category_share_drawn": {
            key: round(drawn_mix[key] / len(drawn), 4) for key in sorted(drawn_mix)
        },
        "category_share_source": {
            key: round(source_mix[key] / len(episodes), 4) for key in sorted(source_mix)
        },
    }
    manifest["total_variation_distance"] = round(
        sum(
            abs(
                manifest["category_share_drawn"].get(key, 0.0)
                - manifest["category_share_source"].get(key, 0.0)
            )
            for key in set(source_mix) | set(drawn_mix)
        )
        / 2,
        4,
    )
    (out / "subsample_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {len(drawn)} episodes from {len(houses)} houses to {out}")
    print(f"  max episodes in one house: {manifest['max_episodes_in_one_house']}")
    print(f"  category mix distance from source: {manifest['total_variation_distance']}")
    print(f"  {'category':22s} {'drawn':>6s} {'share':>7s} {'source share':>13s}")
    for key in sorted(set(source_mix) | set(drawn_mix), key=lambda k: -source_mix[k]):
        print(
            f"  {key:22s} {drawn_mix[key]:6d} {manifest['category_share_drawn'].get(key, 0.0):7.3f} "
            f"{manifest['category_share_source'].get(key, 0.0):13.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
