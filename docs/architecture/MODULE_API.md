# Module API contracts

## Reading the contracts

- `Array` means a NumPy ndarray; `Tensor` means a PyTorch tensor; `Generator` means a NumPy random generator. `Path`, `Sequence` and `Mapping` are standard-library types. `self` is omitted from instance method signatures.
- `StateKey` is a tuple of phase and discrete state values. `dict` result rows have explicit scenario/trajectory/period/manager/model-seed keys where applicable; they are CSV-compatible records, not hidden global state.
- Manager history is `[H,F]`; model history is `[N,H,F]`; numeric actions are `[N,7]`; predictions/targets/masks are `[N,5]`. Defaults are `H=4`, `F=30`. Models receive raw units, apply scaling once, and return raw-unit predictions.
- `Observation.values` follows the 30 feature names in specification section 7.1. Action order is prices x3, staff, stock x3; target order is sales x3, abandonment, served mean wait. Define `FEATURE_NAMES`, `ACTION_NAMES` and `TARGET_NAMES` in `contracts.py`.
- All package `__init__.py` files are empty or expose inert metadata. They must not eagerly import training, data generation, concrete neural managers or PyTorch.
- Constructors validate their inputs. Functions reject incompatible artifacts or malformed arrays with clear exceptions. No function below reads ambient global configuration.
- All input paths identify local artifacts. A stage writes only to its explicit output directory. Loaders validate parent lineage before computation; CLI constructs the expected manifest from selected upstream artifacts.
- `CustomerView` contains what one customer can observe and its own type. It excludes future arrivals, latent demand and other customers’ types. A manager never receives this view, Scenario or CafeState.
- `DayResult.diagnostics` includes market arrivals/outside counts, competitor accounting, customer episode returns by type, total served waits, departure reasons and conservation counts. Period diagnostics include queues, served-wait totals and selected-manager DecisionInfo. Daily summaries derive from counts, not unweighted averages.
- `Config` sections contain the numerical defaults and learning settings specified in the PDF; validation checks nested keys/types, not just the outer dictionaries. This avoids inventing a large class hierarchy for configuration.
- Dataclass fields in the diagrams are constructor parameters. Private CafeState/CustomerInstance objects never leave the simulator. Customer policies own Q tables; simulation owns each customer’s pending transition, terminal state and waiting-cost bookkeeping.

## Diagram index

- Shared types and configuration: [PUML](diagrams/src/contracts_api.puml) · [SVG](diagrams/rendered/contracts_api.svg)
- Simulation and customer learning: [PUML](diagrams/src/simulation_api.puml) · [SVG](diagrams/rendered/simulation_api.svg)
- Managers and neural predictor contracts: [PUML](diagrams/src/manager_models.puml) · [SVG](diagrams/rendered/manager_models.svg)
- Dataset generation and preparation: [PUML](diagrams/src/data_api.puml) · [SVG](diagrams/rendered/data_api.svg)
- Neural training and forecast evaluation: [PUML](diagrams/src/training_api.puml) · [SVG](diagrams/rendered/training_api.svg)
- Business evaluation, reports and CLI: [PUML](diagrams/src/orchestration_api.puml) · [SVG](diagrams/rendered/orchestration_api.svg)

Dashed arrows in the detailed diagrams show direct dependencies. References to shared
types and configuration appear in signatures and in the shared types diagram.

## Modules, objects and function roles

### `contracts`

Scenario and observation values are read-only after construction. The [shared types diagram](diagrams/rendered/contracts_api.svg) lists record fields.

| Signature | Role |
| --- | --- |
| `build_catalogue(config: Config) -> tuple[Action, ...]` | Enumerate feasible price/staff/stock combinations in stable ID order. |
| `validate_action(action: Action, catalogue: Sequence[Action]) -> None` | Reject unknown IDs or numeric values inconsistent with the catalogue. |
| `make_history(observations: Sequence[Observation], H: int) -> Array` | Take the latest H observations, left-pad with zeros, and preserve validity flags. Used online and offline. |

### `config`

The `demand` section controls scenario generation. Peak centers and widths use minutes. The daily multiplier is sampled once per day, and the latent multiplier varies by period.

| Signature | Role |
| --- | --- |
| `load_config(path: Path) -> Config` | Read JSON, apply documented defaults, and validate every section. |
| `validate_config(config: Config) -> None` | Check durations, probabilities, costs, action profiles, model dimensions and seed settings; reject invalid input. |
| `config_hash(config: Config) -> str` | Hash canonical resolved configuration for artifact compatibility. |

### `simulation.scenarios`

Owns exogenous randomness; saved schedules never depend on managers.

| Signature | Role |
| --- | --- |
| `generate_scenario(config: Config, scenario_id: str, group: str, seed: int) -> Scenario` | Sample demand, ordered arrivals, customer types and reproducible tie seeds. |
| `save_scenario(scenario: Scenario, path: Path) -> None` | Write numeric NPZ arrays with IDs and metadata; no pickle. |
| `load_scenario(path: Path) -> Scenario` | Validate and restore a saved immutable schedule. |
| `customer_rng(scenario: Scenario, customer_id: int, minute: int, phase: str) -> Generator` | Derive a stable decision stream independent of policy-dependent random call counts. |

### `simulation.simulator`

Owns market state and the single event loop.

| Signature | Role |
| --- | --- |
| `run_day(scenario: Scenario, customer_policy: CustomerPolicy, manager: Manager, config: Config, *, trajectory_id: str, learn: bool = False, epsilon: float = 0.0, exploration_seed: int \| None = None) -> DayResult` | Reset day state and manager, execute all minute events, finalize closing departures, and return period/diagnostic records. Learning requires a mutable policy and an explicit exploration seed; frozen runs require epsilon=0. |
| `build_observation(period_index: int, previous: PeriodRecord \| None, queue_lengths: tuple[int, int], config: Config) -> Observation` | Construct only the 30 allowed pre-decision features; opening previous values are zero. |
| `period_profit(action: Action, sales: Array, unit_costs: Array, staff_period_cost: float) -> float` | Revenue minus costs of all prepared stock and all staffed employees; no additional waste penalty. |

### `customers.policy`

Contains customer behavior and Q updates, without importing simulation or training. StateKey is a phase-tagged tuple of discrete bins; customer action IDs are distinct from manager Action IDs.

**CustomerPolicy** — class.

```python
tables: dict
frozen: bool
```

| Signature | Role |
| --- | --- |
| `__init__(tables: dict \| None = None, *, frozen: bool = False)` | Own one arrival/wait table per customer type; frozen instances reject updates. |
| `choose(type_id: int, state: StateKey, valid_actions: Sequence[int], *, epsilon: float, rng: Generator) -> int` | Choose a masked epsilon-greedy customer action with reproducible unbiased tie resolution; unseen states read as zero without mutating frozen tables. |
| `update(pending: PendingTransition, reward: float, next_state: StateKey \| None, next_valid_actions: Sequence[int], *, terminal: bool, alpha: float, gamma: float) -> None` | Apply one Q update. Terminal transitions have no bootstrap; next-state maximum is over valid actions only. |
| `freeze() -> CustomerPolicy` | Return an independent read-only snapshot for validation or downstream use. |
| `fingerprint() -> str` | Hash canonical tables and state schema to detect accidental mutation. |

| Signature | Role |
| --- | --- |
| `encode_state(view: CustomerView) -> StateKey` | Apply the specification arrival/wait binning; no whole-market state or customer-ID key. |
| `valid_actions(view: CustomerView, config: Config) -> tuple[int, ...]` | Mask unavailable/unaffordable choices; force leave for stock, budget, patience or closing conditions. |
| `save_policy(policy: CustomerPolicy, path: Path, manifest: ArtifactManifest) -> None` | Serialize JSON Q-table entries with explicit tuple keys and lineage. |
| `load_policy(path: Path, expected: ArtifactManifest) -> CustomerPolicy` | Check lineage and schema; restore frozen tables. |

### `customers.training`

Runs customer learning before management dataset generation. Receives exploration managers from the caller so it does not import data generation.

| Signature | Role |
| --- | --- |
| `train_customers(config: Config, train_scenarios: Sequence[Scenario], validation_scenarios: Sequence[Scenario], managers: Sequence[Manager], output_dir: Path, manifest: ArtifactManifest) -> CustomerPolicy` | Validate disjoint scenario groups, schedule epsilon, train days, evaluate checkpoints, save diagnostics, and freeze the best behaviorally valid checkpoint. |
| `evaluate_customers(policy: CustomerPolicy, scenarios: Sequence[Scenario], managers: Sequence[Manager], config: Config) -> list[dict]` | Run no-update validation days and aggregate episode returns, choices, purchases and waiting by type. |
| `check_behavior(policy: CustomerPolicy, config: Config) -> dict[str, bool]` | Run controlled purchase/outside/waiting checks; fail checkpoint acceptance if core behavior is broken. |

### `managers.base`

Common parent for all managers. No neural or simulation imports.

**Manager** — abstract class.

```python
last_decision: DecisionInfo | None
```

| Signature | Role |
| --- | --- |
| `reset() -> None` | Clear day-local state and diagnostics. |
| `choose(history: Array) -> Action` | Abstract action-selection contract; raw history [H,F], one catalogue action out. |

### `managers.fixed`

Return the validation-selected constant action.

**FixedManager** — class inheriting `Manager`.

| Signature | Role |
| --- | --- |
| `__init__(action: Action)` | Store validated settings and dependencies; no file I/O. |
| `reset() -> None` | Clear day-local diagnostics; retain the constant action. |
| `choose(history: Array) -> Action` | Return the validation-selected constant action. Record last_decision. |

### `managers.adaptive`

Maintain entry-demand EMA and apply the specified staffing, stock and price rules.

**AdaptiveManager** — class inheriting `Manager`.

| Signature | Role |
| --- | --- |
| `__init__(catalogue: Sequence[Action], config: Config, *, alpha: float, buffer: float, queue_threshold: int)` | Store validated settings and dependencies; no file I/O. |
| `reset() -> None` | Reset per-day state; adaptive EMA starts at 20, and all decision diagnostics clear. |
| `choose(history: Array) -> Action` | Maintain entry-demand EMA and apply the specified staffing, stock and price rules. Record last_decision. |

### `managers.mlp`

Use the MLP predictor and the shared neural selector.

**MLPManager** — class inheriting `Manager`.

| Signature | Role |
| --- | --- |
| `__init__(predictor: MLPPredictor, catalogue: Sequence[Action], config: Config)` | Store validated settings and dependencies; no file I/O. |
| `reset() -> None` | Clear day-local diagnostics; predictor weights and fitted scalers stay fixed. |
| `choose(history: Array) -> Action` | Use the MLP predictor and the shared neural selector. Record last_decision. |

### `managers.lstm`

Use the LSTM predictor and the same selector.

**LSTMManager** — class inheriting `Manager`.

| Signature | Role |
| --- | --- |
| `__init__(predictor: LSTMPredictor, catalogue: Sequence[Action], config: Config)` | Store validated settings and dependencies; no file I/O. |
| `reset() -> None` | Clear day-local diagnostics; predictor weights and fitted scalers stay fixed. |
| `choose(history: Array) -> Action` | Use the LSTM predictor and the same selector. Record last_decision. |

### `managers.selection`

Shared neural candidate evaluation; receives costs and constraint explicitly, never a simulator.

| Signature | Role |
| --- | --- |
| `select_action(predictor: OutcomePredictor, history: Array, catalogue: Sequence[Action], unit_costs: Array, staff_period_cost: float, abandonment_limit: float) -> tuple[Action, DecisionInfo]` | Batch all candidates, compute expected profit, select feasible maximum or abandonment/profit fallback, then lowest action ID. Convert NumPy history/actions to tensors on the predictor device and return diagnostics. |

### `models.base`

PyTorch-only common neural parent. Array-to-tensor conversion occurs at callers; forward accepts raw tensors and scales them exactly once.

**OutcomePredictor** — abstract class inheriting `nn.Module`.

```python
scalers: registered tensor buffers
```

| Signature | Role |
| --- | --- |
| `__init__(scalers: ScalerStats)` | Register fitted statistics as buffers; preserve validity indicators and padded zeros. |
| `forward(raw_history: Tensor, raw_actions: Tensor) -> Tensor` | Differentiable scaling, subclass logits, bounded raw-unit decoding; output [N,5]. |
| `predict(raw_history: Tensor, raw_actions: Tensor) -> Tensor` | Inference-mode wrapper using eval mode; same shapes and units as forward. |
| `forward_logits(scaled_history: Tensor, scaled_actions: Tensor) -> Tensor` | Abstract architecture-specific computation returning five logits per candidate. |

### `models.mlp`

Flatten history, concatenate the candidate action and apply the MLP.

**MLPPredictor** — class inheriting `OutcomePredictor`.

| Signature | Role |
| --- | --- |
| `__init__(scalers: ScalerStats, *, H: int = 4, F: int = 30, hidden_sizes: tuple[int, int] = (64, 32))` | Create the specified network and initialize the common predictor parent. |
| `forward_logits(scaled_history: Tensor, scaled_actions: Tensor) -> Tensor` | Flatten history, concatenate the candidate action and apply the MLP. |

### `models.lstm`

Encode history with a fresh zero-state LSTM, concatenate action, and apply the prediction head.

**LSTMPredictor** — class inheriting `OutcomePredictor`.

| Signature | Role |
| --- | --- |
| `__init__(scalers: ScalerStats, *, F: int = 30, hidden_size: int = 32, head_size: int = 32)` | Create the specified network and initialize the common predictor parent. |
| `forward_logits(scaled_history: Tensor, scaled_actions: Tensor) -> Tensor` | Encode history with a fresh zero-state LSTM, concatenate action, and apply the prediction head. |

### `data.generate`

Produces factual period records with frozen customers. Splitting and window preparation are separate calls orchestrated by the CLI.

**ExplorationManager** — class inheriting `Manager`.

| Signature | Role |
| --- | --- |
| `__init__(catalogue: Sequence[Action], seed: int, *, mode: str)` | Choose uniform-per-period or random-hold exploration; hold lengths are 2–4 periods. |
| `reset() -> None` | Reset action-hold state and the trajectory RNG to its configured seed. |
| `choose(history: Array) -> Action` | Select a catalogue action using the exploration stream only. |

| Signature | Role |
| --- | --- |
| `build_dataset(scenarios: Sequence[Scenario], customer_policy: CustomerPolicy, config: Config, exploration_seeds: Mapping[str, Sequence[int]], output_dir: Path, manifest: ArtifactManifest) -> list[PeriodRecord]` | Create independently seeded exploration managers, run factual trajectories, verify policy hashes, save schedules and records, and return rows. Full runs use two modes per scenario; smoke uses one. |

### `data.splits`

Assign whole scenario groups before constructing windows; shared schedules cannot cross splits.

| Signature | Role |
| --- | --- |
| `make_split_manifest(scenario_ids: Sequence[str], *, seed: int, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> dict[str, str]` | Reproducibly map unique management scenario IDs to train/validation/test. |
| `validate_splits(records: Sequence[PeriodRecord], splits: Mapping[str, str], excluded_ids: Sequence[str]) -> None` | Reject missing/unknown assignments and reuse of customer train/validation IDs; all trajectories follow their scenario assignment. |

### `data.windows`

Build raw-unit arrays only. Scaling is owned by the predictor forward path, preventing double normalization.

| Signature | Role |
| --- | --- |
| `make_windows(records: Sequence[PeriodRecord], splits: Mapping[str, str], *, split: str, H: int) -> DatasetArrays` | Group by scenario and trajectory, order periods, construct causal histories with make_history, and attach actual actions, targets, masks and explicit keys. Reject missing/duplicate periods. |

### `data.scaling`

Fits plain numeric statistics without PyTorch or simulation imports. Runtime transformations are performed by the common predictor using these saved statistics.

| Signature | Role |
| --- | --- |
| `fit_scalers(records: Sequence[PeriodRecord], splits: Mapping[str, str]) -> ScalerStats` | Use unique real training observations/actions, not repeated overlapping windows. Compute valid-only target scales, floor standard deviations at 1.0, and fail when training has no valid waits. |
| `save_scalers(scalers: ScalerStats, path: Path) -> None` | Write JSON fitted values and training IDs. |
| `load_scalers(path: Path, expected_training_ids: Sequence[str]) -> ScalerStats` | Validate shape, finite values, standard-deviation floor and training membership. |

### `data.storage`

Thin format functions for dataset records and manifests; not a generic storage framework.

| Signature | Role |
| --- | --- |
| `save_records(records: Sequence[PeriodRecord], path: Path, *, splits: Mapping[str, str] \| None = None) -> None` | Write flat CSV numeric actions, keys and outcomes. Before splitting, use split=unassigned; CLI rewrites with the saved split map before window construction. |
| `load_records(path: Path) -> list[PeriodRecord]` | Validate required columns/types and reconstruct records. |
| `save_arrays(arrays: DatasetArrays, path: Path) -> None` | Write numeric NPZ arrays and explicit keys without pickle. |
| `load_arrays(path: Path, *, H: int, F: int) -> DatasetArrays` | Validate shapes, keys, masks and finite numeric values. |
| `save_manifest(manifest: ArtifactManifest, splits: Mapping[str, str], directory: Path) -> None` | Write artifact lineage and scenario split JSON sidecars. |
| `load_manifest(directory: Path, expected: ArtifactManifest) -> tuple[ArtifactManifest, dict[str, str]]` | Fail on incompatible schema, simulator, configuration, catalogue, parent hashes or field order; validate split assignments. |

### `models.training`

One supervised training loop supports both predictor subclasses; customer training is not imported.

| Signature | Role |
| --- | --- |
| `train_predictor(model_name: str, train: DatasetArrays, validation: DatasetArrays, scalers: ScalerStats, config: Config, *, seed: int, device: str, output_dir: Path, manifest: ArtifactManifest) -> OutcomePredictor` | Seed before construction, instantiate MLP/LSTM, optimize masked loss, early-stop, restore and save the best checkpoint, and log forecast metrics/runtime. |
| `outcome_loss(predictions: Tensor, targets: Tensor, target_mask: Tensor, target_std: Tensor) -> Tensor` | Compute mean normalized sales MSE plus abandonment MSE plus valid-wait MSE; zero wait contribution when a batch has no valid waits. |
| `save_checkpoint(model: OutcomePredictor, path: Path, config: Config, seed: int, manifest: ArtifactManifest) -> None` | Save weights, buffers, architecture, field order, seed and complete lineage. |
| `load_checkpoint(path: Path, expected: ArtifactManifest, *, device: str = "cpu") -> OutcomePredictor` | Validate checkpoint compatibility and instantiate the matching predictor without simulator access. |

### `models.evaluation`

Forecast accuracy on factual held-out trajectories; no manager tuning or market rollouts.

| Signature | Role |
| --- | --- |
| `evaluate_forecasts(model: OutcomePredictor, arrays: DatasetArrays, *, batch_size: int = 128) -> tuple[list[dict], dict[str, float]]` | Predict raw outcomes with keyed rows, compute per-target MAE/RMSE and valid-only wait errors; return data for plots. |

### `evaluate`

Business evaluation applies to all managers. Candidate managers and loaded predictors are passed by the CLI; this module does not construct network architectures. Forecast evaluation is dispatched separately by the CLI.

| Signature | Role |
| --- | --- |
| `select_managers(candidates: Mapping[str, Sequence[Manager]], validation_scenarios: Sequence[Scenario], policy: CustomerPolicy, config: Config) -> tuple[dict[str, Manager], list[dict]]` | Rank candidates within each family on validation mean daily profit under mean abandonment constraint; fallback to abandonment/profit and deterministic candidate order. Return selected managers and all trial metrics. |
| `evaluate(managers: Mapping[str, Manager], scenarios: Sequence[Scenario], policy: CustomerPolicy, config: Config, *, output_dir: Path, manifest: ArtifactManifest) -> list[dict]` | Run matched full days for each manager/seed, confirm frozen policy unchanged, write period/day tables and return keyed daily rows. |
| `summarize_results(day_rows: Sequence[dict], period_rows: Sequence[dict]) -> list[dict]` | Compute weighted waits, profit, sales, waste units/cost, abandonment validity/violations, action/fallback frequencies and baseline wins. |
| `paired_profit_intervals(day_rows: Sequence[dict], baseline: str, *, seed: int, resamples: int = 2000, confidence: float = 0.95, seed_mode: str = "mean_per_day") -> list[dict]` | Bootstrap independent scenario days, pairing policies and explicitly handling model seeds; never treat seeds as independent days. |
| `measure_decision_latency(manager: Manager, histories: Sequence[Array], *, warmup: int = 10, repeats: int = 100) -> dict[str, float]` | Time complete action selection after warmup, preserve chronological history order with resets between passes, synchronize accelerator work, and report mean/p95. Caller supplies a separate measurement manager. |

### `report`

Reads saved CSV/JSON only, using standard library, NumPy and Matplotlib. No training or simulation imports.

| Signature | Role |
| --- | --- |
| `build_report(results_dir: Path, output_dir: Path) -> list[Path]` | Produce required static plots, complexity table and a factual interpretation draft; return generated paths. |
| `select_representative_day(day_rows: Sequence[dict], *, reference_manager: str = "adaptive") -> str` | Choose the scenario nearest median reference-manager profit, with scenario-ID tie break, before plotting all managers on that same day. |

### `cli`

Composition root. Loads artifacts, constructs concrete objects, and dispatches stages; stage imports remain lazy.

| Signature | Role |
| --- | --- |
| `main(argv: Sequence[str] \| None = None) -> int` | Parse stage/config/experiment/model/seed/device arguments; invoke the relevant module functions and return a process exit code. |
| `smoke(config: Config, output_dir: Path) -> dict` | Run a bounded deterministic smoke pipeline through implemented stages; label fixture customers/debug seeds and report checks. No placeholder performance claims. |

## Integrity rules to verify during implementation

- No simulation import of customer training, data generation, model training, concrete managers or reporting. Customer policy never imports simulation.
- No model import of managers, customers or simulation. Model definitions depend on fitted ScalerStats, not on the code that generated the dataset.
- `run_day` resets the manager, queues, stock, service credit and pending customer episodes. It charges each waiting minute once and completes all terminal transitions, including closing departures.
- `CustomerPolicy.update` cannot mutate a frozen instance. Training exploration and keyed evaluation ties use separate streams.
- Scalers are fitted on real unique training rows, not padded or repeated window entries. Predictor scaling preserves indicators and zero padding. Output decoding uses raw action stock.
- Fixed/adaptive/MLP/LSTM choose methods return the same Action type. Neural models share the same forward/predict contract and selector. LSTM state is zero for each window.
- Match every dataset array to explicit keys and an upstream manifest. Scenario groups, feature order and hashes must be checked before training or evaluation.
- No forecast loss or test result silently selects business manager settings; validation ranking and test reporting remain separate.
