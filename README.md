# Café Decision Lab

A controlled two-café experiment comparing fixed, adaptive, MLP and LSTM managers
on profit, customer abandonment, waiting and waste.

**Status:** architecture and API design are documented. The simulator, training
pipeline and evaluation code have not been implemented. Commands in the architecture
document are planned interfaces, not runnable instructions yet.

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
├── .gitignore
└── docs/
    └── architecture/
        ├── ARCHITECTURE.md
        ├── MODULE_API.md
        └── diagrams/
            ├── src/           # PlantUML sources
            └── rendered/      # SVG previews
```

Implementation will introduce sibling `simulation`, `customers`, `managers`,
`models` and `data` packages under `src/cafe/`. Customer training belongs beside
customer policies; neural training belongs beside predictor definitions.

The first milestone is the simulator, fixed manager, deterministic checks and an
executable smoke run. Later stages train and freeze customer policies, generate
datasets, train predictors and run paired business evaluation.
