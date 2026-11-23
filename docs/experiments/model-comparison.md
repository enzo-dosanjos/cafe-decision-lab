# Model selection for café management

The original fixed, adaptive, MLP and LSTM comparison used the architecture and training defaults from the project specification. It established how those configurations behaved. This follow-up compares alternatives on the existing validation days. The original test results were already known when the study was designed.

## What the inputs and decision require

The simulator supplies four consecutive observations and one proposed catalogue action. Each row has continuous or count targets for three products' sales, abandonment and mean served wait. The predictors therefore share the original regression loss and physical output bounds. The affine control uses a single linear map before decoding. A threshold perceptron in its original binary classification form cannot output these five quantities. The convolutional models filter along time, with observation fields as channels. The recurrent models start each window with fresh state.

Each forecast model uses the same decision rule. It scores all 27 actions, maximizes expected profit subject to predicted abandonment of at most five, and falls back to minimum predicted abandonment if necessary. The limit, available actions, simulator cost model and frozen customer population are experiment design choices.

## Experiments

| Modeling decision | Alternatives tried |
|---|---|
| Perceptron and feed-forward networks | Single affine layer with output decoding, a 32-unit sigmoid hidden layer, and the original 64/32 ReLU MLP |
| Recurrent networks | Vanilla tanh RNN and LSTM, each with 32 hidden units |
| Convolutional networks | Temporal convolution with max pooling. A supplemental screen tested shallow tanh, deeper stacks, parallel filters and a residual connection motivated by the historical vision examples |
| Optimization | Full-batch GD, single-example SGD, mini-batch SGD, momentum SGD, AdaGrad, RMSprop and Adam. Two learning rates per optimizer, 0.001 and 0.01 |
| Learning-rate schedules | Constant, step decay, exponential decay, cosine annealing and cosine warm restarts |
| Regularization and norms | No additional regularizer, L1 penalty, coupled L2 weight decay, 20% dropout and batch normalization |

The pilot used 64 sampled training scenarios (2,048 periods), seed 0, eight epochs and all 150 validation scenarios for scoring. Scalers were held fixed at their original training-only values. All candidates saw the same number of example passes. Full-batch GD received eight updates while single-example SGD received 16,384. Pilot scores therefore compare behavior at that exposure budget and cannot establish optimizer superiority at convergence.

Full fits used all 700 training scenarios, seeds 0–2 and a 35-epoch cap. Each kept its lowest validation-loss epoch. The full comparison includes the six main architectures, the original Adam setting, and the pilot finalists for optimizer, regularization and schedule. Five supplementary experiments adapted convolutional blocks from LeNet-5, AlexNet, VGG, GoogLeNet and ResNet to four time steps. Their pilot losses ranged from 1.746 to 1.964, compared with 1.714 for the simple temporal CNN.

For business validation each checkpoint controlled the café for all 150 matched days. The table averages the daily results across three training seeds. Forecast loss uses the saved normalized three-group loss. The fixed comparison is the validation-selected constant action 4.

## Pilot evidence

| Optimizer | Loss at 0.001 | Loss at 0.01 |
|---|---:|---:|
| Full-batch GD | 3.631 | 3.588 |
| Single-example SGD | 1.328 | 1.430 |
| Mini-batch SGD | 3.564 | 3.313 |
| Momentum SGD | 3.330 | 1.513 |
| AdaGrad | 3.249 | 1.357 |
| RMSprop | 1.336 | 1.328 |
| Adam | 1.601 | 1.307 |

At Adam 0.001, batch normalization improved pilot loss from 1.601 to 1.430. Dropout at 20% scored 1.649, L1 at 0.0001 scored 1.602 and L2 at 0.001 scored 1.603. Keeping the learning rate constant beat all four schedules at eight epochs: exponential decay scored 1.841, warm restarts 1.939, step decay 1.962 and cosine annealing 2.115. The pilot advanced the best alternative in each category alongside the original settings. This gives the alternatives a longer run even when the control wins the pilot.

## Validation results

| Recipe | Learning rate | Forecast loss | HKD/day | Abandonments/period | Periods above five |
|---|---:|---:|---:|---:|---:|
| Vanilla RNN | 0.001 | 1.040 | 3,373.00 | 1.52 | 7.6% |
| LSTM | 0.001 | 1.046 | 3,355.18 | 1.49 | 7.5% |
| Temporal CNN | 0.001 | 1.058 | 3,339.96 | 1.38 | 6.4% |
| MLP / batch normalization | 0.001 | 1.045 | 3,309.85 | 1.34 | 6.1% |
| MLP / Adam | 0.001 | 1.043 | 3,307.12 | 1.33 | 6.0% |
| MLP / exponential decay | 0.001 | 1.054 | 3,295.07 | 1.41 | 6.8% |
| MLP / RMSprop | 0.01 | 1.085 | 3,269.91 | 1.35 | 6.5% |
| MLP / Adam | 0.01 | 1.077 | 3,213.65 | 1.31 | 6.1% |
| Shallow sigmoid | 0.001 | 1.077 | 3,197.83 | 1.12 | 4.4% |
| Affine | 0.001 | 1.277 | 2,323.29 | 0.90 | 3.4% |

The follow-up sensitivity screen varied hidden widths, activation, initialization, batch size, dropout probability and L2 strength individually. Batch size 32 scored 1.252 and widths 128/64 scored 1.385, versus 1.601 for the matched MLP control. These are single-seed short-run forecast results. Batch size 32 also receives four times as many updates per epoch as batch size 128. These supplementary variants were evaluated separately from the full-data shortlist.

## Fresh-day confirmation

The validation winner was fixed before these rollouts. A new scenario seed namespace generated 150 customer schedules. Every manager saw the same schedules. Neural results average the three training seeds within each day.

| Manager | Mean HKD/day | Periods above five abandonments |
|---|---:|---:|
| fixed | 3,284.30 | 15.54% |
| rnn-0.001 | 3,466.32 | 8.08% |
| adam-0.001 | 3,389.74 | 6.28% |

The paired gain over fixed was HKD 182.02 per day (95% paired day bootstrap interval 140.75 to 224.47).
The paired gain over adam-0.001 was HKD 76.58 per day (95% paired day bootstrap interval 60.78 to 92.42).

Intervals describe variation across days in this simulator, conditional on these trained seeds.

## Reproduction

From the repository root, substitute the directory containing the selected 4,000-day customer experiment. Use a fresh output directory.

```sh
OMP_NUM_THREADS=1 python scripts/model_comparison.py --experiment-dir artifacts/management-4000 --customer-dir artifacts/customers-4000 --output-dir runs/model-comparison-repeat
OMP_NUM_THREADS=1 python scripts/confirm_manager.py --experiment-dir artifacts/management-4000 --customer-dir artifacts/customers-4000 --study-dir runs/model-comparison-repeat
```

For exact replay of an archived confirmation, pass its scenario ID prefix with `--scenario-prefix` (the portion before the three-digit day index). Customer tie-breaking uses the scenario ID as well as the seed.
