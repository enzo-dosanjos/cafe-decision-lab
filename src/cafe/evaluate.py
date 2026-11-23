"""Select management rules using realized outcomes on matched validation days."""

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context

from .config import Config
from pathlib import Path

from .contracts import Array, ArtifactManifest, FEATURE_NAMES, Scenario
from .customers.policy import CustomerPolicy
from .managers.base import Manager
from .simulation.simulator import run_day


def _validation_days(scenarios: Sequence[Scenario], candidates: Mapping[str, Sequence[Manager]],
                     policy: CustomerPolicy, config: Config) -> list[dict]:
    """Run fresh complete days for every candidate on the supplied validation schedules."""
    rows = []
    fingerprint = policy.fingerprint()
    limit = config.evaluation["abandonment_limit"]

    for family, managers in candidates.items():
        for index, manager in enumerate(managers):
            for scenario in scenarios:
                periods = run_day(scenario, policy, manager, config, trajectory_id=f"{family}-{index}").periods
                decisions = [r.diagnostics["decision"] for r in periods]
                day = {"family": family, "candidate": index, "scenario_id": scenario.scenario_id,
                       "periods": len(periods), "profit": sum(r.profit for r in periods),
                       "abandonment": sum(r.abandonment for r in periods),
                       "violations": sum(r.abandonment > limit for r in periods),
                       "fallbacks": sum(d.fallback_used for d in decisions if d is not None),
                       "actions": [r.action.action_id for r in periods]}

                if decisions[0] is not None and decisions[0].predicted_feasible is not None:
                    day["predicted_feasible"] = [d.predicted_feasible for d in decisions]
                    day["fallback_used"] = [d.fallback_used for d in decisions]

                rows.append(day)

    if policy.fingerprint() != fingerprint:
        raise ValueError("customer policy changed during manager validation")

    return rows


def select_managers(candidates: Mapping[str, Sequence[Manager]], validation_scenarios: Sequence[Scenario],
                    policy: CustomerPolicy, config: Config, *, workers: int = 1) -> tuple[dict[str, Manager], list[dict]]:
    """Compare mean daily profit subject to mean period abandonment, using candidate order for ties."""
    if (not policy.frozen or not validation_scenarios or not candidates or not all(candidates.values())
            or workers < 1 or len({s.scenario_id for s in validation_scenarios}) != len(validation_scenarios)):
        raise ValueError("selection requires frozen customers, distinct days, candidates and positive workers")

    workers = min(workers, len(validation_scenarios))

    if workers == 1:
        days = _validation_days(validation_scenarios, candidates, policy, config)
    else:
        chunks = [validation_scenarios[i::workers] for i in range(workers)]

        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            batches = pool.map(_validation_days, chunks, repeat(candidates), repeat(policy), repeat(config))
            days = [row for batch in batches for row in batch]

    trials = []
    selected = {}
    limit = config.evaluation["abandonment_limit"]

    for family, managers in candidates.items():
        family_trials = []

        for index in range(len(managers)):
            rows = sorted((r for r in days if r["family"] == family and r["candidate"] == index),
                          key=lambda r: r["scenario_id"])
            periods = sum(r["periods"] for r in rows)
            abandonment = sum(r["abandonment"] for r in rows) / periods
            family_trials.append({"family": family, "candidate": index, "days": rows,
                                  "mean_daily_profit": sum(r["profit"] for r in rows) / len(rows),
                                  "mean_period_abandonment": abandonment, "feasible": abandonment <= limit,
                                  "violation_rate": sum(r["violations"] for r in rows) / periods,
                                  "fallback_rate": sum(r["fallbacks"] for r in rows) / periods})

        feasible = [r for r in family_trials if r["feasible"]]

        if feasible:
            winner = min(feasible, key=lambda r: (-r["mean_daily_profit"], r["candidate"]))
        else:
            winner = min(family_trials, key=lambda r: (r["mean_period_abandonment"], -r["mean_daily_profit"], r["candidate"]))

        selected[family] = managers[winner["candidate"]]
        trials.extend(family_trials)

    return selected, trials


def _write_csv(rows: Sequence[dict], path: 'Path') -> None:
    """Write flat result rows with a shared column order and empty missing values."""
    import csv

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(managers: Mapping[str, Manager], scenarios: Sequence[Scenario], policy: CustomerPolicy,
             config: Config, *, output_dir: 'Path', manifest: 'ArtifactManifest') -> list[dict]:
    """Save complete matched rollouts; manager labels identify every architecture and seed."""
    from dataclasses import asdict
    import json
    import numpy as np

    if (not policy.frozen or not managers or not scenarios
            or len({s.scenario_id for s in scenarios}) != len(scenarios)):
        raise ValueError("evaluation requires frozen customers, managers and distinct scenario days")

    output_dir.mkdir(parents=True, exist_ok=False)
    fingerprint = policy.fingerprint()
    period_rows, day_rows = [], []
    unit_costs = np.array([p["unit_cost"] for p in config.products])

    for label, manager in managers.items():
        for scenario in scenarios:
            result = run_day(scenario, policy, manager, config, trajectory_id=label)
            rows = []

            for record in result.periods:
                decision = record.diagnostics["decision"]
                sales = int(sum(record.sales))
                row = {"manager": label, "scenario_id": scenario.scenario_id, "period_index": record.period_index,
                       "profit": record.profit, "sales": sales, "entries": record.entries,
                       "abandonment": record.abandonment, "waste_units": int(sum(record.waste)),
                       "waste_cost": float(record.waste @ unit_costs), "preparation_cost": record.preparation_cost,
                       "staff_cost": record.staff_cost, "served_wait_total": record.mean_wait * sales,
                       "violation": int(record.abandonment > config.evaluation["abandonment_limit"]),
                       "action_id": record.action.action_id, "fallback": int(decision.fallback_used) if decision else 0,
                       "queue_A": float(record.observation.values[FEATURE_NAMES.index("current_queue_A")]),
                       "predicted_feasible": decision.predicted_feasible if decision else None}
                rows.append(row)

            period_rows.extend(rows)
            day = {"manager": label, "scenario_id": scenario.scenario_id, "periods": len(rows)}

            for field in ("profit", "sales", "entries", "abandonment", "waste_units", "waste_cost",
                          "preparation_cost", "staff_cost", "served_wait_total", "violation", "fallback"):
                day[field] = sum(r[field] for r in rows)

            day["abandonment_fraction"] = day["abandonment"] / day["entries"] if day["entries"] else None
            day["mean_wait"] = day["served_wait_total"] / day["sales"] if day["sales"] else None
            day_rows.append(day)

    if policy.fingerprint() != fingerprint:
        raise ValueError("customer policy changed during evaluation")

    _write_csv(period_rows, output_dir / "periods.csv")
    _write_csv(day_rows, output_dir / "days.csv")
    _write_csv(summarize_results(day_rows, period_rows), output_dir / "summary.csv")
    (output_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n")

    return day_rows


def summarize_results(day_rows: Sequence[dict], period_rows: Sequence[dict]) -> list[dict]:
    """Summarize each seed separately, weighting wait by purchases and retaining empty denominators."""
    import json
    from collections import Counter
    import numpy as np

    summaries = []
    adaptive = {r["scenario_id"]: r["profit"] for r in day_rows if r["manager"] == "adaptive"}

    for label in sorted({r["manager"] for r in day_rows}):
        days = [r for r in day_rows if r["manager"] == label]
        periods = [r for r in period_rows if r["manager"] == label]
        row = {"manager": label, "days": len(days)}

        for field in ("profit", "sales", "abandonment", "waste_units", "waste_cost", "preparation_cost", "staff_cost"):
            values = [r[field] for r in days]
            row[f"mean_daily_{field}"] = float(np.mean(values))
            row[f"std_daily_{field}"] = float(np.std(values, ddof=1)) if len(days) > 1 else 0.0

        valid = [r["abandonment_fraction"] for r in days if r["abandonment_fraction"] is not None]
        served = sum(r["sales"] for r in days)
        row.update(mean_daily_abandonment_fraction=float(np.mean(valid)) if valid else None,
                   abandonment_fraction_valid_days=len(valid),
                   served_mean_wait=sum(r["served_wait_total"] for r in days) / served if served else None,
                   mean_period_abandonment=float(np.mean([r["abandonment"] for r in periods])),
                   p95_period_abandonment=float(np.quantile([r["abandonment"] for r in periods], .95)),
                   violation_rate=sum(r["violation"] for r in periods) / len(periods),
                   fallback_rate=sum(r["fallback"] for r in periods) / len(periods),
                   action_frequencies=json.dumps(dict(sorted(Counter(r["action_id"] for r in periods).items()))),
                   wins_over_adaptive=sum(r["profit"] > adaptive[r["scenario_id"]] for r in days) if adaptive else None)
        summaries.append(row)

    return summaries


def paired_profit_intervals(day_rows: Sequence[dict], baseline: str, *, seed: int, resamples: int = 2000,
                            confidence: float = .95, seed_mode: str = "mean_per_day") -> list[dict]:
    """Bootstrap paired days, averaging seeds within each day or reporting them separately."""
    import numpy as np

    if seed_mode not in ("mean_per_day", "separate") or resamples < 1 or not 0 < confidence < 1:
        raise ValueError("expected valid seed_mode, positive resamples and confidence in (0,1)")

    by_manager = {}

    for row in day_rows:
        values = by_manager.setdefault(row["manager"], {})

        if row["scenario_id"] in values:
            raise ValueError("duplicate manager/scenario result")

        values[row["scenario_id"]] = row["profit"]

    if baseline not in by_manager or not by_manager[baseline]:
        raise ValueError(f"missing baseline {baseline}")

    identities = sorted(by_manager[baseline])

    if any(set(values) != set(identities) for values in by_manager.values()):
        raise ValueError("paired intervals require identical scenario days for every manager")

    groups = {}

    for label, values in by_manager.items():
        family = label.split("-seed-")[0] if seed_mode == "mean_per_day" else label
        groups.setdefault(family, []).append([values[s] for s in identities])

    rng = np.random.default_rng(seed)
    indices = rng.integers(len(identities), size=(resamples, len(identities)))
    reference = np.array([by_manager[baseline][s] for s in identities])
    rows = []

    for label, values in sorted(groups.items()):
        differences = np.mean(values, axis=0) - reference
        lower, upper = np.quantile(differences[indices].mean(axis=1), [(1 - confidence) / 2, (1 + confidence) / 2])
        rows.append({"manager": label, "baseline": baseline, "seed_mode": seed_mode, "days": len(identities),
                     "seeds": len(values), "mean_profit_difference": float(differences.mean()),
                     "ci_lower": float(lower), "ci_upper": float(upper), "confidence": confidence,
                     "wins": int((differences > 0).sum())})

    return rows


def measure_decision_latency(manager: Manager, histories: Sequence['Array'], *,
                             warmup: int = 10, repeats: int = 100) -> dict[str, float]:
    """Time full candidate selection in milliseconds, resetting each chronological pass."""
    from time import perf_counter
    import numpy as np

    if not len(histories) or warmup < 0 or repeats < 1:
        raise ValueError("latency requires histories, nonnegative warmup and positive repeats")

    predictor = getattr(manager, "predictor", None)
    device = predictor.action_mean.device if predictor is not None else None
    elapsed = []

    for repeat_index in range(warmup + repeats):
        manager.reset()

        for history in histories:
            if device is not None and device.type == "cuda":
                import torch
                torch.cuda.synchronize(device)

            start = perf_counter()
            manager.choose(history)

            if device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)

            duration = (perf_counter() - start) * 1000

            if repeat_index >= warmup:
                elapsed.append(duration)

    return {"mean_ms": float(np.mean(elapsed)), "p95_ms": float(np.quantile(elapsed, .95))}
