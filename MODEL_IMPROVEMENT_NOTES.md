# Angry Birds RL Notes

This file summarizes the important decisions and observations from the RL2
training/evaluation discussion.

## Current Run

- Main run: `ab_agent_20260516_220759_rainbow_D_full_n3_noisy`
- Training preset: `compare`
- Target training steps: 50000
- Latest checkpoint in the finished run: `000049998.weights.h5`
- Exported model path: `out/angry_birds/ab_agent_20260516_220759_rainbow_D_full_n3_noisy/saved_model`
- Eval script uses the SavedModel interface:
  - input `image`: `(batch, 128, 128, 3)`
  - input `bird`: `(batch, 5)`
  - output action values: `(batch, 200)`

## Eval Result

The 2026-05-20 eval run used 20 sampled training levels with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /home/jung/anaconda3/envs/aibirds_tf215/bin/python evaluate_ab.py
```

Summary:

- Passed: 16 / 20
- Failed: 4 / 20
- Win rate: 80.0%
- Average score: 41953
- Max score: 87030
- Average shots: 3.05
- Failed levels: 67, 102, 304, 308

Saved eval outputs:

- `out/angry_birds/ab_agent_20260516_220759_rainbow_D_full_n3_noisy/evaluation_results.csv`
- `out/angry_birds/ab_agent_20260516_220759_rainbow_D_full_n3_noisy/evaluation_summary.txt`
- `out/angry_birds/ab_agent_20260516_220759_rainbow_D_full_n3_noisy/score_distribution.png`

## Resume Accounting

Resume should compute remaining steps from the original run target and the
actual checkpoint that was loaded.

For example, if the target is 50000 and the loaded checkpoint is 1998, the next
transition is 1999 and the remaining count should be:

```text
50000 - 1999 = 48001
```

The code now prefers stable run metadata for the original target because
`training_config.json` can be rewritten with a remaining-step count during
resume.

## Final Compatibility Rules

The professor guidance means final evaluation compatibility is mostly tied to
the original evaluation contract.

Safe or mostly safe:

- Tune `train_ab.py` hyperparameters.
- Change reward weights inside the allowed reward calculation section.
- Train different internal Keras models as long as the exported SavedModel
  keeps the same `image`, `bird`, and 200-action interface.
- Add diagnostics that do not change observation/action/server behavior.
- Use `evaluate_ab.py` for local analysis, while remembering the final grader
  may use the original evaluation script.

Risky for final compatibility:

- Changing observation shape, such as replacing `(128, 128, 3)` with a wide
  aspect-preserved tensor.
- Changing the action space from 200 actions to a finer grid.
- R2D2-style sequence inputs or recurrent state inputs if the final evaluator
  expects single-frame `image` and `bird`.
- Ground-truth contact rewards that require simulator protocol changes.
- Any change to simulator interaction, observation format, or action format.

## Current Bottlenecks

Likely reasons the model plateaus:

- The easy levels are mostly solved, so recent win rate is dominated by hard
  levels where small angle or tap errors fail.
- The game image is squeezed into a square, which distorts geometry and may hurt
  projectile precision.
- The action space is coarse: 20 angle bins by 10 tap bins.
- NoisyNet replaces epsilon exploration, and with epsilon at 0 the policy can
  concentrate on favorite reliable actions.
- Replay memory of 30000 may be small for 200 mixed levels and can push out
  useful older diversity.
- The reward is still win/score heavy, so it gives weak feedback about why a
  failed shot was close or bad.
- ConvNeXt is frozen, so the visual representation may not specialize to pigs,
  blocks, TNT, and trajectory-relevant geometry.

## Model Improvement Options

Best next steps that keep the final contract safer:

1. Reward tuning
   - Keep score and win as the main objective.
   - Add careful shaping only from already available signals.
   - Avoid reward terms that depend on new simulator protocol data.
   - Penalize repeated failing shots lightly.
   - Reward meaningful score delta, not just level completion.

2. More robust training
   - Continue from the 50000-step model with a larger target, such as 100000 or
     200000, but expect diminishing returns if the bottleneck is representation
     or action resolution.
   - Increase replay memory when training longer.
   - Changing replay memory size or batch size after a completed `compare` run
     is okay when resuming, but it changes the training distribution and should
     be recorded.

3. ConvNeXt fine-tuning
   - Realistic advantage: lets the visual backbone adapt to Angry Birds objects
     and geometry instead of using generic ImageNet features.
   - Safer version: unfreeze only the last ConvNeXt block/stage, use a much
     smaller learning rate for the backbone, and keep the same SavedModel
     signature.
   - Main risk: overfitting or destabilizing a model that is already competent.

4. Diagnostics
   - Keep action/tap distribution summaries from `diagnostics.jsonl`.
   - Evaluate per-level failures and repeated actions.
   - Track score delta per shot in local eval if useful.
   - Contact diagnostics are useful for reward design, but true object-contact
     labels are not available through the safe final interface unless the
     simulator or vision pipeline is instrumented separately.

5. Model architecture
   - Current model is Rainbow-style DQN with distributional/dueling/noisy
     pieces.
   - R2D2-style recurrent Rainbow could help if partial observability or shot
     history is the real bottleneck.
   - It is not a guaranteed large win here because most state is visible in the
     current screenshot and final compatibility is much harder.

## Practical Recommendation

For this project, the most logical safe path is:

1. Keep the final-compatible input/action/export contract.
2. Use the current 50k model as the baseline.
3. Run broader eval sets to identify repeat failure levels.
4. Tune reward and replay settings first.
5. Try light ConvNeXt fine-tuning only after baseline eval is stable.
6. Treat aspect-preserved inputs, finer action grids, and R2D2 as research
   branches unless the final evaluator is changed to support them.
