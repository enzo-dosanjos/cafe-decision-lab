"""Train tabular customers on independent days and select a frozen checkpoint."""

from collections import Counter
from collections.abc import Sequence
import csv
from pathlib import Path

import numpy as np

from ..config import Config, config_hash
from ..contracts import ArtifactManifest, Scenario
from ..managers.base import Manager
from ..simulation.simulator import run_day
from .policy import CustomerPolicy, load_policy, save_policy


def evaluate_customers(policy: CustomerPolicy, scenarios: Sequence[Scenario],
                       managers: Sequence[Manager], config: Config) -> list[dict]:
    """Pool completed episodes by type and overall on fixed no-update validation days."""
    if not managers or not scenarios:
        raise ValueError("evaluation requires scenarios and managers")

    totals = [Counter() for _ in config.customer_types]

    for index, scenario in enumerate(scenarios):
        result = run_day(scenario, policy, managers[index % len(managers)], config,
                         trajectory_id="customer-validation")
        diagnostics = result.diagnostics

        for type_id, returns in enumerate(diagnostics["episode_returns_by_type"]):
            totals[type_id].update(diagnostics["outcomes_by_type"][type_id])
            totals[type_id]["episodes"] += len(returns)
            totals[type_id]["return_sum"] += sum(returns)

    overall = Counter()

    for total in totals:
        overall.update(total)

    rows = []

    for label, total in zip(["all", *(c["name"] for c in config.customer_types)], [overall, *totals]):
        count = total["episodes"]
        purchased = total["purchased_A"] + total["purchased_B"]
        rows.append({
            "type": label, "episodes": count, "purchases": purchased,
            "mean_return": total["return_sum"] / count if count else None,
            "purchase_fraction": purchased / count if count else None,
            "outside_fraction": total["outside"] / count if count else None,
            "mean_served_wait": total["served_wait_total"] / purchased if purchased else None,
        })

    return rows


def check_behavior(policy: CustomerPolicy, config: Config) -> dict[str, bool]:
    """Probe learned preferences at opening and under congestion (see MODULE_API.md)."""
    tables = policy.tables
    positive = []
    overpriced = []
    wait_advantages = {}
    multipliers = config.catalogue["price_multipliers"]

    for type_id, profile in enumerate(config.customer_types):
        states = tables.get(type_id, {})
        advantages = {}

        for state, q in states.items():
            if state[0] == "arrival" and state[2:] == (0, 0, 7, 7):
                # Compare each A product with outside at an empty, stocked market.
                for product_id, product in enumerate(config.products):
                    price = round(product["base_price"] * multipliers[state[1]], 2)
                    value = profile["values"][product_id]

                    if price <= profile["budget"] and price < value:
                        positive.append(q[product_id] > q[6])
                    elif value < price <= profile["budget"]:
                        overpriced.append(q[product_id] < q[6])

            if state[0] == "waiting" and state[4] >= 2 and state[7:] == (1, 0):
                # Compare types at the same café, product, price, position and elapsed wait.
                key = state[1:6]
                advantages.setdefault(key, []).append(q[0] - q[1])

        wait_advantages[type_id] = {key: float(np.mean(values)) for key, values in advantages.items()}

    costs = [customer["waiting_cost"] for customer in config.customer_types]
    sensitive = wait_advantages[int(np.argmax(costs))]
    patient = wait_advantages[int(np.argmin(costs))]
    common = sorted(sensitive.keys() & patient.keys())

    return {
        "positive_purchase": any(positive),
        "outside_over_overpriced": any(overpriced),
        "costly_long_wait": bool(common) and sum(sensitive[k] - patient[k] for k in common) < 0,
    }


def train_customers(config: Config, train_scenarios: Sequence[Scenario],
                    validation_scenarios: Sequence[Scenario], managers: Sequence[Manager],
                    output_dir: Path, manifest: ArtifactManifest) -> CustomerPolicy:
    """Train daily, log every checkpoint and persist the earliest best eligible policy."""
    settings = config.customer_training
    scenarios = (*train_scenarios, *validation_scenarios)
    ids = [s.scenario_id for s in scenarios]

    if len(train_scenarios) != settings["days"] or len(validation_scenarios) != settings["validation_days"]:
        raise ValueError("scenario counts must match customer_training days and validation_days")

    if len(set(ids)) != len(ids) or len({s.seed for s in scenarios}) != len(ids):
        raise ValueError("training and validation require distinct scenario IDs and seeds")

    if {s.group for s in train_scenarios} & {s.group for s in validation_scenarios}:
        raise ValueError("customer training and validation groups must be disjoint")

    if manifest.config_hash != config_hash(config) or set(manifest.scenario_ids) != set(ids):
        raise ValueError("manifest must identify the resolved configuration and training/validation scenarios")

    if not managers:
        raise ValueError("training requires at least one manager")

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy.json"
    csv_path = output_dir / "validation.csv"

    if policy_path.exists() or csv_path.exists():
        raise FileExistsError(f"Use a new customer-training output directory: {output_dir}")

    policy = CustomerPolicy()
    rng = np.random.default_rng(config.seeds["customer"])
    schedule = np.linspace(settings["epsilon_start"], settings["epsilon_end"], len(train_scenarios))
    best_score = float("-inf")
    writer = None

    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        for day, (scenario, epsilon) in enumerate(zip(train_scenarios, schedule), 1):
            run_day(scenario, policy, managers[(day - 1) % len(managers)], config,
                    trajectory_id="customer-learning", learn=True, epsilon=float(epsilon),
                    exploration_seed=int(rng.integers(2**63 - 1)))

            if day % settings["validation_interval"] and day != len(train_scenarios):
                continue

            snapshot = policy.freeze()
            rows = evaluate_customers(snapshot, validation_scenarios, managers, config)
            checks = check_behavior(snapshot, config)
            score = rows[0]["mean_return"]
            improved = score is not None and all(checks.values()) and score > best_score

            for row in rows:
                entry = {"day": day, "epsilon": epsilon, **row, **checks, "new_best": improved}

                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(entry))
                    writer.writeheader()

                writer.writerow(entry)

            stream.flush()

            if improved:
                best_score = score
                temporary = policy_path.with_suffix(".tmp")
                save_policy(snapshot, temporary, manifest)
                temporary.replace(policy_path)

            print(f"Customer day {day}: validation return={score}, checks={checks}, new_best={improved}",
                  flush=True)

    if best_score == float("-inf"):
        raise ValueError(f"No eligible customer checkpoint. See {csv_path} for validation diagnostics.")

    return load_policy(policy_path, manifest)
