"""Summarize matched-day experiments with common service metrics and paired intervals."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def families(payload: dict) -> dict:
    """Average seed outcomes within each independent day before comparing managers."""
    runs = defaultdict(list)

    for trial in payload["trials"]:
        days = sorted(trial["days"], key=lambda day: day["scenario_id"])
        ids = tuple(d["scenario_id"] for d in days)
        values = [
            [
                d["profit"],
                d["abandonment"] / d["periods"],
                100 * d["violations"] / d["periods"],
            ]
            for d in days
        ]
        runs[trial["family"]].append((ids, np.asarray(values)))

    averages = {}

    for name, seeds in runs.items():
        if any(ids != seeds[0][0] for ids, _ in seeds):
            raise ValueError(f"unmatched days for {name}")

        averages[name] = seeds[0][0], np.mean([v for _, v in seeds], axis=0)

    return averages


def interval(first: tuple, second: tuple) -> str:
    """Report mean paired daily profit difference and percentile interval in HKD."""
    if first[0] != second[0]:
        raise ValueError("paired comparisons require identical scenario IDs")

    differences = first[1][:, 0] - second[1][:, 0]
    draws = np.random.default_rng(518).choice(differences, (10000, len(differences)))
    low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])

    return f"{differences.mean():+.2f} [{low:+.2f}, {high:+.2f}]"


def main() -> None:
    """Write the tables and profit–service figure using saved daily outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    confirmed = families(json.loads(args.confirmation.read_text()))
    results = {}
    coordinates = defaultdict(list)

    for mode in ("2", "5", "10", "profit", "history", "schedule"):
        payload = json.loads((args.service_dir / mode / "results.json").read_text())
        results[mode] = families(payload)
        print(mode, payload["settings"])

        for family, (_, days) in results[mode].items():
            profit, abandonment, violations = days.mean(axis=0)
            print(f"{family}: {profit:.2f}, {abandonment:.3f}, {violations:.2f}%")

            if mode in ("2", "5", "10", "profit"):
                coordinates[family].append((violations, profit))

    print("Confirmation", {k: v[1].mean(axis=0).tolist() for k, v in confirmed.items()})
    print("RNN minus LSTM", interval(confirmed["rnn-0.001"], confirmed["lstm-0.001"]))

    for family in ("adam", "rnn"):
        print(
            f"{family} four minus one",
            interval(results["5"][family], results["history"][family]),
        )

    print(
        "RNN minus schedule",
        interval(results["5"]["rnn"], results["schedule"]["schedule"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, points in coordinates.items():
        ax = axes[0] if name in ("fixed", "adaptive") else axes[1]
        x, y = zip(*points)
        (line,) = ax.plot(
            x,
            y,
            label={"adam": "MLP", "rnn": "RNN", "lstm": "LSTM"}.get(name, name.title()),
            alpha=0.8,
        )
        for marker, xx, yy in zip(("o", "s", "^", "x"), x, y):
            ax.scatter(xx, yy, marker=marker, color=line.get_color())

    for ax in axes:
        ax.legend()
        ax.set_xlabel("Periods above five abandonments (%)")
        ax.set_ylabel("Mean daily profit (HKD)")
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Decision limit: circle = 2, square = 5, triangle = 10, cross = profit only"
    )
    fig.savefig(args.figure, dpi=180)


if __name__ == "__main__":
    main()
