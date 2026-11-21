# Café Decision Lab

A controlled two-café experiment comparing fixed, adaptive, MLP and LSTM managers
on profit, customer abandonment, waiting and waste.

**Status:** shared configuration and record types, the fixed manager, and the contract smoke command are implemented. Full market rollouts require the simulation and customer-policy modules.

From the repository root, install into your Python environment and run:

```sh
python -m pip install -e .
python -m unittest discover -s tests
python -m cafe.cli smoke
```

The smoke command writes the action catalogue, opening history and selected action to
`runs/smoke/contracts.json`. With the simulator and customer policy installed, run
`python -m cafe.cli smoke --rollout` to verify a zero-arrival day against exact
preparation and staffing costs and save its period summaries to `runs/smoke/day.json`.
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
├── src/cafe/                  # Shared contracts, managers and CLI
├── tests/                     # Foundation checks
└── docs/
    └── architecture/
        ├── ARCHITECTURE.md
        ├── MODULE_API.md
        └── diagrams/
            ├── src/           # PlantUML sources
            └── rendered/      # SVG previews
```

The first milestone is the simulator, fixed manager, deterministic checks and an
executable smoke run. Later stages train and freeze customer policies, generate
datasets, train predictors and run paired business evaluation.
