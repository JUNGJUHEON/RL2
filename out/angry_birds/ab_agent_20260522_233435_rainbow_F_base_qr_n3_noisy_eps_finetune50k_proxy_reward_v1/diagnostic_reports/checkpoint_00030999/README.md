# Training Diagnostics Report

Source diagnostics: `out/angry_birds/ab_agent_20260522_233435_rainbow_F_base_qr_n3_noisy_eps_finetune50k_proxy_reward_v1/diagnostics.jsonl`
Checkpoint: `checkpoint_00030999`
Model: `ab_agent_20260522_233435_rainbow_F_base_qr_n3_noisy_eps_finetune50k_proxy_reward_v1`
Stem: `PretrainedConvNeXtTiny`
Q head: `QuantileDuelingQNetwork`
Run: F_base: QR-Rainbow, n_step=3, NoisyNet ON, epsilon 0.05->0.01, ConvNeXt fine-tune at 50k

## Plain-English Read

ConvNeXt status: **healthy_technical_signal**

ConvNeXt is producing finite, non-constant image features. That means the visual stem is technically working, but policy quality still needs evaluation.

RL status: **learning_loss_decreased**

RL loss is lower at this checkpoint than at the first logged update.

Important interpretation: finite non-collapsed ConvNeXt features mean the visual stem is technically working. They do not prove the policy is strong; use evaluation runs for that.

## Key Numbers

- Activation rows: 8829
- Episodes: 8786
- Wins: 5437
- Win rate: 0.618825
- Average return: 5.99987
- Average score: 34633.3
- Train level pool / coverage: filtered / 200/200 (1)
- Learning updates: 7750
- Loss first/last/min/max: 7.44433 / 0.398971 / 0.0806571 / 7.5478
- Last epsilon: 0.0211692
- Last action/angle/tap entropy: 2.71768 / 2.17633 / 1.78201
- ConvNeXt image feature std min/mean: 0.218851 / 0.497755
- ConvNeXt fine-tune status: backbone_frozen_until_scheduled_finetune; backbone grad norm last/mean: 0 / 0
- QR status: qr_quantiles_logged; quantiles: 51; std mean: 2.51758; p90-p10 spread mean: 6.28492
- Auxiliary heads: not_auxiliary_run; aux loss last/mean: 0 / 0; weighted aux last/mean: 0 / 0
- Proxy reward status: not_proxy_reward_run; positive rate: 0; avg proxy bonus: 0
- Proxy counts tap-score/tap-win/pig/best-score: 0 / 0 / 0 / 0
- ConvNeXt NaN/Inf total: 0 / 0
- Q output NaN/Inf total: 0 / 0

## Added Components Compared With Model D

- QR-DQN head: qr_quantiles_logged
- Low epsilon plus NoisyNet exploration: last epsilon 0.0211692
- Scheduled ConvNeXt fine-tuning: backbone_frozen_until_scheduled_finetune
- Proxy reward shaping: not_proxy_reward_run
- All-map training pool: broad_all_map_coverage
- Action/tap usage: action entropy 2.71768, tap entropy 1.78201

## Files

- convnext_rl_step_csv: `convnext_rl_step_diagnostics.csv`
- episode_csv: `episode_diagnostics.csv`
- learning_csv: `learning_diagnostics.csv`
- reward_action_csv: `reward_action_diagnostics.csv`
- level_summary_csv: `level_summary_diagnostics.csv`
- activation_plot: `convnext_activation_health.png`
- rl_plot: `rainbow_rl_training_health.png`
- summary_json: `summary.json`
- summary_markdown: `README.md`
