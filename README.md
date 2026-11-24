# Café Decision Lab

A controlled two-café experiment comparing fixed, adaptive, MLP and LSTM managers
on profit, customer abandonment, waiting and waste.

[Final report](docs/FINAL_REPORT.md) · [Model comparison](docs/experiments/model-comparison.md)

## Reproduce the reported experiment

Start in a fresh checkout. The commands below regenerate the selected customer policy,
the management dataset, the three-seed baseline selection and the fresh-day comparison.
The original customer run used Python 3.12.16 and NumPy 1.26.4 on CPU. For reproducing
the original artifact versions, the dataset stage used Python 3.14.8 and NumPy 2.4.6,
and neural training used Python 3.13.18, NumPy 2.5.4 and CPU PyTorch 2.14.1. Package versions are recorded in the
artifacts and can affect hashes and numerical results.

```sh
# Use the three stage-specific interpreters to reproduce the recorded environment.
python3.12 -m venv venv/customers
python3.14 -m venv venv/data
python3.13 -m venv venv/models
venv/customers/bin/python -m pip install -e . numpy==1.26.4
venv/data/bin/python -m pip install -e . numpy==2.4.6
venv/models/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
venv/models/bin/python -m pip install -e ".[neural,report]" numpy==2.5.4
. venv/models/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
venv/customers/bin/python -m cafe.cli train-customers --config config/customers-4000.json --output-dir artifacts/customers-4000
venv/data/bin/python -m cafe.cli generate-data --config config/default.json --customer-dir artifacts/customers-4000 --output-dir artifacts/management-4000
for model in mlp lstm; do
  for seed in 0 1 2; do
    python -m cafe.cli train --model "$model" --seed "$seed" --experiment-dir artifacts/management-4000
  done
done
python -m cafe.cli validate-managers --models mlp lstm --seeds 0 1 2 --experiment-dir artifacts/management-4000 --customer-dir artifacts/customers-4000 --output-dir runs/validation-three-seed
python scripts/model_comparison.py --experiment-dir artifacts/management-4000 --customer-dir artifacts/customers-4000 --output-dir runs/model-comparison
python scripts/confirm_manager.py --experiment-dir artifacts/management-4000 --customer-dir artifacts/customers-4000 --study-dir runs/model-comparison --selection runs/validation-three-seed/selection.json
```

The reported frozen table fingerprint is `a43b7230f395762c659de2cc4a36e9e764b9fe46ad591ffa74b646f274477ace`.
Dataset generation writes elapsed wall-clock seconds to `timing.json` in the experiment
directory. Evaluation includes this shared duration in `complexity.csv`; older datasets
without a timer leave that field empty.

**Status:** fixed, adaptive, MLP and LSTM managers support validation selection and
matched held-out evaluation with three neural training seeds. Forecast diagnostics,
paired day-bootstrap intervals, seed variability and static report figures are available.
The full comparison has been exercised on 150 held-out scenario days, with recurrent
reset, matched-day and accounting checks.

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

## MLP training and manager validation

Install the neural dependencies, train the seed-0 MLP, then compare managers using
the same customer experiment used to generate the dataset:

```sh
python -m pip install -e ".[neural]"
python -m cafe.cli train --model mlp --seed 0 --experiment-dir artifacts/management-4000
python -m cafe.cli validate-managers --seed 0 --experiment-dir artifacts/management-4000 --customer-dir artifacts/customers-4000 --output-dir runs/validation-mlp-0
```

## LSTM and three-seed comparison

The LSTM uses the same four observations, numeric actions, scaling, decoded outputs,
loss and selector as the MLP. Its hidden/cell state starts at zero for every window;
only recurrent gradients are clipped, using the configured norm limit. Existing
checkpoints are never overwritten. Keep the seed-0 MLP and train the remaining seeds:

```sh
python -m cafe.cli train --model mlp --seed 1 --experiment-dir artifacts/management-4000
python -m cafe.cli train --model mlp --seed 2 --experiment-dir artifacts/management-4000
python -m cafe.cli train --model lstm --seed 0 --experiment-dir artifacts/management-4000
python -m cafe.cli train --model lstm --seed 1 --experiment-dir artifacts/management-4000
python -m cafe.cli train --model lstm --seed 2 --experiment-dir artifacts/management-4000
python -m cafe.cli validate-managers --models mlp lstm --seeds 0 1 2 --experiment-dir artifacts/management-4000 --customer-dir artifacts/customers-4000 --output-dir runs/validation-three-seed
python -m cafe.cli evaluate --experiment-dir artifacts/management-4000 --customer-dir artifacts/customers-4000 --selection runs/validation-three-seed/selection.json --output-dir results/three-seed
python -m pip install -e ".[report]"
python -m cafe.cli report --results-dir results/three-seed --output-dir results/three-seed/report
```

To reuse the original fixed/adaptive validation trials, pass
`--baseline-selection runs/manager-validation-mlp-0/selection.json` to validation.
The command checks dataset lineage, candidate settings and validation day IDs before
reuse. It preserves the source file and ranks the six neural candidates on validation
business results. No architecture or history alternatives are tried by these commands.
Evaluation verifies checkpoint hashes against validation and evaluates **all three
seeds**, including the validation-selected seed, without selecting on test outcomes.

The output contains period/day accounting, per-seed business summaries, factual
validation/test forecasts, MAE/RMSE per target, full-selection CPU latency and paired
profit comparisons against fixed and adaptive. The report adds paired LSTM-minus-MLP
forecast-error intervals, paired daily business-metric differences and seed SD/range tables. Confidence intervals resample 150
independent scenario days, retaining all seeds and trajectories within each day.
Profit intervals include both averages across seeds per day and separate-seed results;
forecast intervals average seed-specific error metrics rather than ensemble predictions.
They describe day variability conditional on the trained seeds. Wait is weighted by
purchases; undefined abandonment fractions are excluded and their valid-day count is
explicit. Generated checkpoints, tables and figures stay in ignored local directories.

Training reads the saved training and validation splits and training-only scalers.
It selects the epoch with the lowest validation loss and saves its weights and
per-epoch metrics under `models/mlp/` in the dataset experiment. Validation runs
all fixed actions, eight adaptive rules and the trained MLP on complete matched
days. `selection.json` records each candidate's settings, daily actions, realized
profit and abandonment, plus the selected candidate for each family. Set
`--workers` to parallelize independent validation days on CPU. Use a fresh
output directory for each comparison.

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

## Service limits, history length and scheduled actions

After running the published model comparison above, run each sensitivity experiment
in its own output directory. Every reported violation uses the same threshold of five
abandonments per period.

```sh
for mode in 2 5 10 profit history schedule; do
  python scripts/service_study.py --mode "$mode" --experiment-dir artifacts/management-4000 --customer-dir artifacts/customers-4000 --study-dir runs/model-comparison --selection runs/validation-three-seed/selection.json --output-dir runs/service-study
done
python scripts/summarize_service.py --service-dir runs/service-study --confirmation runs/model-comparison/confirmation.json --figure runs/service-study/profit-service.png
```
