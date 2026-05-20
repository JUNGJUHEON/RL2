# Training Diagnostics Report

Source diagnostics: `out/angry_birds/ab_agent_20260516_220759_rainbow_D_full_n3_noisy/diagnostics.jsonl`
Checkpoint: `checkpoint_00039998`
Model: `ab_agent_20260516_220759_rainbow_D_full_n3_noisy`
Stem: `PretrainedConvNeXtTiny`
Q head: `DistributionalDuelingQNetwork`
Run: D: ConvNeXtTiny full Rainbow, n_step=3, NoisyNet ON

## Plain-English Read

ConvNeXt status: **healthy_technical_signal**

ConvNeXt is producing finite, non-constant image features. That means the visual stem is technically working, but policy quality still needs evaluation.

RL status: **learning_loss_decreased**

RL loss is lower at this checkpoint than at the first logged update.

Important interpretation: finite non-collapsed ConvNeXt features mean the visual stem is technically working. They do not prove the policy is strong; use evaluation runs for that.

## Key Numbers

- Activation rows: 11394
- Episodes: 11331
- Wins: 6584
- Win rate: 0.581061
- Average return: 5.62412
- Average score: 33145.3
- Learning updates: 10015
- Loss first/last/min/max: 3.04511 / 1.57143 / 0.0868733 / 26.932
- Last epsilon: 0
- ConvNeXt image feature std min/mean: 0.23541 / 0.512276
- ConvNeXt NaN/Inf total: 0 / 0
- Q output NaN/Inf total: 0 / 0

## Files

- convnext_rl_step_csv: `convnext_rl_step_diagnostics.csv`
- episode_csv: `episode_diagnostics.csv`
- learning_csv: `learning_diagnostics.csv`
- activation_plot: `convnext_activation_health.png`
- rl_plot: `rainbow_rl_training_health.png`
- summary_json: `summary.json`
- summary_markdown: `README.md`
