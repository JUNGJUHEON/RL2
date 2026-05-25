# Training Diagnostics Report

Source diagnostics: `out/angry_birds/ab_agent_20260522_233435_rainbow_F_base_qr_n3_noisy_eps_finetune50k_proxy_reward_v1/diagnostics.jsonl`
Checkpoint: `checkpoint_00052999`
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

- Activation rows: 15232
- Episodes: 15158
- Wins: 9757
- Win rate: 0.643687
- Average return: 6.26112
- Average score: 35738.3
- Train level pool / coverage: filtered / 200/200 (1)
- Learning updates: 13250
- Loss first/last/min/max: 7.44433 / 1.4934 / 0.0562833 / 15.2516
- Last epsilon: 0.0115027
- Last action/angle/tap entropy: 2.82469 / 2.16604 / 1.79011
- ConvNeXt image feature std min/mean: 0.218851 / 0.497911
- ConvNeXt fine-tune status: backbone_gradients_enabled; backbone grad norm last/mean: 0.0280086 / 0.00959366
- QR status: qr_quantiles_logged; quantiles: 51; std mean: 2.64725; p90-p10 spread mean: 6.5405
- Auxiliary heads: not_auxiliary_run; aux loss last/mean: 0 / 0; weighted aux last/mean: 0 / 0
- Proxy reward status: not_proxy_reward_run; positive rate: 0; avg proxy bonus: 0
- Proxy counts tap-score/tap-win/pig/best-score: 0 / 0 / 0 / 0
- ConvNeXt NaN/Inf total: 0 / 0
- Q output NaN/Inf total: 0 / 0

## Added Components Compared With Model D

- QR-DQN head: qr_quantiles_logged
- Low epsilon plus NoisyNet exploration: last epsilon 0.0115027
- Scheduled ConvNeXt fine-tuning: backbone_gradients_enabled
- Proxy reward shaping: not_proxy_reward_run
- All-map training pool: broad_all_map_coverage
- Action/tap usage: action entropy 2.82469, tap entropy 1.79011

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
