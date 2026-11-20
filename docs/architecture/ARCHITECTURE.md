# Café Decision Lab architecture

All four managers control café A on matched scenario days. They share a simulator,
frozen customer policies and an action catalogue so differences in profit and
service can be attributed to management decisions.

[Package diagram](diagrams/rendered/cafe_architecture.svg) · [Module APIs and detailed diagrams](MODULE_API.md)

Use Python, NumPy, PyTorch and Matplotlib. Record package versions with each experiment.

## Source tree

```text
cafe-decision-lab/
├── README.md                       # Project status and documentation index
├── docs/
│   └── architecture/
│       ├── ARCHITECTURE.md         # Structure and design decisions
│       ├── MODULE_API.md           # Function signatures, roles and types
│       └── diagrams/
│           ├── src/               # Editable PlantUML sources
│           └── rendered/          # Checked SVG previews
├── pyproject.toml                  # Package metadata and dependencies
├── requirements-lock.txt           # Exact environment used for results
├── .gitignore
├── config/
│   ├── default.json                # Full experiment defaults from specification
│   └── smoke.json                  # Explicit small-run configuration
├── src/
│   └── cafe/
│       ├── __init__.py
│       ├── contracts.py            # Shared dataclasses, field order, catalogue
│       ├── config.py               # Validated configuration and hashes
│       ├── cli.py                  # Stage dispatch and concrete manager assembly
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── scenarios.py        # Demand, arrivals, attributes, random streams
│       │   └── simulator.py        # Tick loop, queues, stock, accounting
│       ├── customers/
│       │   ├── __init__.py
│       │   ├── policy.py           # Customer states, masking, Q-learning policy
│       │   └── training.py         # Training days, behavioral checks, freezing
│       ├── managers/
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract Manager: reset / choose
│       │   ├── fixed.py            # FixedManager(Manager)
│       │   ├── adaptive.py         # AdaptiveManager(Manager)
│       │   ├── mlp.py              # MLPManager(Manager), owns MLPPredictor
│       │   ├── lstm.py             # LSTMManager(Manager), owns LSTMPredictor
│       │   └── selection.py        # Shared neural profit/constraint selection
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py             # OutcomePredictor: common I/O and decoding
│       │   ├── mlp.py              # MLPPredictor(OutcomePredictor)
│       │   ├── lstm.py             # LSTMPredictor(OutcomePredictor)
│       │   ├── training.py         # Neural loss, optimization, checkpoints
│       │   └── evaluation.py       # Forecast MAE/RMSE and predictions only
│       ├── data/
│       │   ├── __init__.py
│       │   ├── generate.py         # Simulation rollouts and exploration Manager
│       │   ├── splits.py           # Scenario-group assignments and validation
│       │   ├── windows.py          # Histories, padding, targets and masks
│       │   ├── scaling.py          # Fit/train-only and reusable transforms
│       │   └── storage.py          # CSV/NPZ records, keys and dataset manifests
│       ├── evaluate.py             # All-manager business selection/comparison
│       └── report.py               # Saved results to static figures and summaries
├── tests/
│   ├── test_simulator.py           # Accounting, capacity, boundaries, conservation
│   ├── test_customers.py           # Q transitions, masks, frozen behavior
│   ├── test_data.py                # Splits, histories, padding, train-only scalers
│   ├── test_models.py              # Shared I/O, decoding, gradients, recurrent reset
│   └── test_managers.py            # Predictor bounds, selection, fallback, resets
├── artifacts/                     # Generated, excluded from normal source commits
│   └── <experiment-id>/
│       ├── manifest.json           # Resolved config, versions, hashes, stage status
│       ├── catalogue.json          # Stable IDs and numeric action values
│       ├── scenarios/
│       │   ├── manifest.json       # Customer train/validation and management groups
│       │   └── <scenario-id>.npz   # Ordered arrivals, types, exogenous tie keys
│       ├── customers/
│       │   ├── policy.json         # Selected frozen Q-tables and metadata
│       │   └── validation.csv      # Checkpoint and per-type diagnostics
│       ├── dataset/
│       │   ├── splits.json
│       │   ├── periods.csv
│       │   ├── scalers.json
│       │   ├── train.npz
│       │   ├── validation.npz
│       │   └── test.npz
│       ├── models/
│       │   ├── mlp/seed-<n>.pt
│       │   └── lstm/seed-<n>.pt
│       └── selection.json         # Validation-selected baselines/model variants
└── results/
    └── <evaluation-id>/
        ├── manifest.json          # Source experiment, scenarios, device, seed policy
        ├── periods.csv
        ├── days.csv
        ├── forecasts.csv
        ├── summary.csv
        ├── complexity.csv
        ├── figures/
        └── interpretation.md
```

## Simulation and state ownership

`run_day` owns two café states, one FIFO queue per café, inventory, fractional
service credits, customer instances, period counters and the rolling history.
It resets the manager and all day state at opening. Café B uses fixed settings.

Each minute executes the specified order in one central loop:

1. At boundaries, finalize previous-period accounting and waste, build the
   pre-decision observation, choose A's action and replenish both cafés.
2. Process existing customers' forced and voluntary departures in FIFO order.
3. Process saved arrivals in their saved order.
4. Allocate service slots and execute FIFO purchases at current prices.
5. Charge waiting disutility only to remaining waiters and complete transitions.

At closing, process remaining departures before finalizing the final period.
Queues carry across periods, stock does not. Unused whole service slots disappear.
Preparation costs apply to all prepared units; waste has no second penalty.

Customer training and frozen evaluation use the same tick loop. Track each
customer's pending Q transition in its instance to prevent reward misassignment.
Use separate random streams for scenarios, manager exploration, customer exploration
and neural training. Customer tie-breaking is keyed by scenario/customer and decision
context so extra random calls in one rollout cannot shift another customer's randomness.

## Information and model contracts

The simulator emits the exact 30 pre-decision features from specification section
7.1, with named, ordered fields defined once in `contracts`. Both neural models
receive four observations, including the current one, and the same numeric action:

| Array | Shape | Meaning |
| --- | --- | --- |
| History | `[N, 4, 30]` | Left-padded within one trajectory/day |
| Action | `[N, 7]` | Three prices, staff count, three stock quantities |
| Target | `[N, 5]` | Three sales counts, abandonment count, served mean wait |
| Target mask | `[N, 5]` | Wait invalid when no purchases; other targets valid |

The MLP flattens history, concatenates action (127 inputs), and uses hidden layers
64 and 32. The LSTM uses hidden size 32, concatenates its final output with action,
then uses a 32-unit head. Reset LSTM hidden/cell state for every history window,
including inference; manager history is not a persistent recurrent hidden state.

Scale real training observations only, preserving validity flags and restoring
padded rows to zero. Fit action scales on training actions and target scales on
valid training targets, with effective standard deviation at least 1.0.
Decode sales using raw candidate stock times sigmoid, and abandonment/wait using
softplus. Keep raw candidate actions available separately from scaled model inputs.

Both neural managers use one selector: batch the same 27 catalogue actions,
maximize predicted profit among actions with predicted abandonment at most 5,
otherwise minimize abandonment then maximize profit. Final ties use action ID.
Predictions are expected values, not guarantees; report actual constraint violations.

## Data lineage and evaluation

- Full collection: 1,000 scenarios, two exploration trajectories per scenario,
  16 periods each, giving 32,000 records. Use uniform-action and 2–4-period
  held-action exploration. Smoke collection uses 20 scenarios and one trajectory.
- Save a 70/15/15 scenario split before windows. Both trajectories from a scenario
  stay together. Explicit identifiers distinguish customer-training, customer-validation
  and management scenarios; customer learning never sees management validation/test days.
- Each row stores its actual action and actual outcome. NPZ arrays preserve scenario,
  trajectory and period keys. No copied counterfactual labels.
- Each artifact records schema/simulator version, resolved configuration hash,
  catalogue hash, relevant parent hashes and seeds. Check compatibility before use.
  Checkpoints also save feature/target order, model configuration and scalers.
- Tune all 27 fixed actions and eight adaptive variants on validation days. Select
  neural candidates using validation business outcomes, alongside forecast diagnostics.
- Evaluate forecasts on held-out exploratory trajectories and business performance
  on fresh complete rollouts of the same 150 test scenarios for every manager.
  Use three training seeds per neural architecture in the final comparison.
- Bootstrap paired independent scenario days, keeping policy results together.
  State whether seed results are separate or averaged per day. Weight served waits
  by served customers. Report daily abandonment denominator validity explicitly.

## Implementation sequence and checks

| Milestone | First implementation | Evidence required |
| --- | --- | --- |
| M1 | Packaging/config/contracts, scenarios, simulator, fixed manager, smoke CLI | Saved deterministic day; accounting, stock, capacity, conservation and boundary tests |
| M2 | Customer states, transitions, training/checkpoint diagnostics | Behavioral checks and selected frozen tables; hashes unchanged in frozen runs |
| M3 | Exploration, splits, records, windows, scalers | Disjoint groups, correct shapes, training-only scaling, reproducible saved schedules |
| M4 | Adaptive baseline, MLP, common selector, validation rollouts | Selected baseline settings; predictor bounds, fallback/ties, masked loss and comparable runs |
| M5 | LSTM and three-seed comparison | Identical history/action/target contracts; hidden-state reset checks; paired evaluation |
| M6 | Report, runtime measurement, manifests, reproduction instructions | Summary tables, required static figures, recorded package versions and interpretation |

M1 smoke may use an explicitly labeled deterministic customer fixture to exercise
queues and accounting before Q-learning exists. It is not evidence about trained
customer behavior. Extend smoke to cover the full pipeline as stages become available.

CLI commands after package installation with `python -m pip install -e .`:

```sh
python -m cafe.cli smoke
python -m cafe.cli train-customers --config config/default.json
python -m cafe.cli generate-data --config config/default.json
python -m cafe.cli train --model mlp --seed 0
python -m cafe.cli train --model lstm --seed 0
python -m cafe.cli evaluate
python -m cafe.cli report
python -m unittest discover -s tests
```

Expose an explicit experiment/artifact-directory argument on stage commands so
multiple runs cannot silently overwrite or mix one another. Default commands target
one documented default experiment. Repeat model training for seeds 1 and 2 for final results.

## Decisions still to validate

Start with the specified semantics and numerical defaults. Adjust customer behavior
or data volume only from the required diagnostics, and record any semantic change
before regenerating affected artifacts. No decision about a web application,
distributed system or optional neural extension is needed for the core experiment.
