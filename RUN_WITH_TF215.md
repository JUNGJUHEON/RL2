# RL2 / TensorFlow 2.15 Notes

This copy is intended for the `aibirds_tf215` conda environment.

```bash
conda activate aibirds_tf215
cd ~/RL2
python scripts/smoke_tf215.py
python train_ab.py
```

The environment uses TensorFlow 2.15.1 and Keras 2.15.0 so it stays below
the TensorFlow 2.16/Keras 3 split. Training output is written under
`~/RL2/out/`, separate from `~/RL/out/`.

For evaluation, `evaluate_ab.py` defaults to the finished Rainbow DQN run. You
can override the model name, sample count, and sim speed from the terminal:

```bash
conda activate aibirds_tf215
cd ~/RL2
AB_MODEL_NAME=ab_agent_20260516_220759_rainbow_D_full_n3_noisy \
AB_NUM_EVAL_LEVELS=20 \
AB_SIM_SPEED=3 \
python evaluate_ab.py
```

This folder installs conda activation hooks for `aibirds_tf215` from
`conda_hooks/`. They clear stale `LD_LIBRARY_PATH`, `LD_PRELOAD`, and
`PYTHONPATH` values before TensorFlow imports, so the env behaves independently.

If the hooks have not been installed yet:

```bash
cp conda_hooks/activate.d/rl2_tf215_clean.sh \
  ~/anaconda3/envs/aibirds_tf215/etc/conda/activate.d/
cp conda_hooks/deactivate.d/rl2_tf215_restore.sh \
  ~/anaconda3/envs/aibirds_tf215/etc/conda/deactivate.d/
```
