# Café Decision Lab

A controlled two-café experiment comparing fixed, adaptive, MLP and LSTM managers
on profit, customer abandonment, waiting and waste.

**Status:** the market simulator runs complete days with a fixed manager and a tabular
customer policy. Saved arrival schedules support deterministic replay. The policy
supports Q-learning updates, freezing and persistence. The next milestone is the
multi-day customer trainer and validation-based checkpoint selection.

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
