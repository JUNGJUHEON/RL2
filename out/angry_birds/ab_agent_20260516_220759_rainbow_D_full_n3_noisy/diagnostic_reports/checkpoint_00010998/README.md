# Training Diagnostics Report

Source diagnostics: `out/angry_birds/ab_agent_20260516_220759_rainbow_D_full_n3_noisy/diagnostics.jsonl`
Checkpoint: `checkpoint_00010998`
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

- Activation rows: 3037
- Episodes: 3017
- Wins: 1469
- Win rate: 0.486908
- Average return: 4.60039
- Average score: 28623.2
- Learning updates: 2765
- Loss first/last/min/max: 3.04511 / 0.982923 / 0.215679 / 26.932
- Last epsilon: 0
- ConvNeXt image feature std min/mean: 0.283144 / 0.564717
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
