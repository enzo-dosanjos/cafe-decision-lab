# Café Decision Lab

A controlled two-café experiment comparing fixed, adaptive, MLP and LSTM managers
on profit, customer abandonment, waiting and waste.

**Status:** market simulation, customer training and management dataset generation run end to end.
The next milestone is the adaptive baseline and MLP outcome predictor.

From the repository root, install into your Python environment and run:

```sh
python -m pip install -e .
python -m unittest discover -s tests
python -m cafe.cli smoke --rollout
```

The smoke command writes the action catalogue, opening history and selected action to
`runs/smoke/contracts.json`. With `--rollout`, it also verifies a zero-arrival day and saves
period summaries to `runs/smoke/day.json`. With the smoke defaults, each of the
two periods costs HKD 230 in prepared stock and HKD 80 in staffing.
Use `--output-dir` to keep separate runs.

## Customer training

From the repository root, run the full experiment in a fresh output directory:

```sh
python -m cafe.cli train-customers --config config/default.json --output-dir artifacts/customer-training
```

The defaults train for 1,000 days with epsilon decreasing from 1.0 to 0.05.
Every 100 days the trainer evaluates a frozen snapshot on the same 30 independent
validation days. Among checkpoints passing all behavioral checks, the highest
mean episode return wins, with ties going to the earlier day. The trainer also
evaluates the final day when the interval does not divide the training length.

The directory contains the resolved settings, catalogue, scenario schedules and
lineage manifest. `customers/validation.csv` records outcomes by customer type and
checkpoint. `customers/policy.json` contains the best eligible frozen tables.
If every checkpoint fails, the CSV explains which checks failed and the command
exits with an error. For a smaller pilot use `config/smoke.json` and a separate
output directory.

## Management dataset

Pass the customer experiment containing the selected frozen checkpoint and its manifest.
For example, if the 4,000-day experiment is stored at `artifacts/customers-4000`, run:

```sh
python -m cafe.cli generate-data --config config/default.json --customer-dir artifacts/customers-4000 --output-dir artifacts/management-4000
```

Use a new output directory for each experiment. The full configuration generates
1,000 independent days with two trajectories per day: uniform actions each period
and actions held for 2–4 periods. Both trajectories share the same arrivals and
stay in the same split. The 32,000 period rows are split across 700 training,
150 validation and 150 test scenarios. For a 20-day integration run, use
`--config config/smoke.json` and a separate output directory.

The experiment stores replayable schedules under `scenarios/`. Its `dataset/`
directory contains factual period records, scenario splits, raw four-period windows
for each split, and scalers fitted on training periods. The manifest identifies
the frozen customer tables and checkpoint file by hash.

## Documentation

- [Architecture and proposed source tree](docs/architecture/ARCHITECTURE.md)
- [Module APIs, function roles and dependency rules](docs/architecture/MODULE_API.md)
- [Package diagram](docs/architecture/diagrams/rendered/cafe_architecture.svg)
- [Manager and model contracts](docs/architecture/diagrams/rendered/manager_models.svg)

The module API document links to six detailed diagrams covering shared types,
simulation and customers, managers and models, data, neural training, and evaluation.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── config/                    # Full defaults and smoke overrides
├── src/cafe/                  # Simulation, customer policies, managers and CLI
├── tests/                     # Contract, customer-policy and simulator checks
└── docs/
    └── architecture/
        ├── ARCHITECTURE.md
        ├── MODULE_API.md
        └── diagrams/
            ├── src/           # PlantUML sources
            └── rendered/      # SVG previews
```
