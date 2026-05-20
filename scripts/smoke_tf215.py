"""Quick compatibility check for the aibirds_tf215 conda environment.

Run from the project root:

    conda activate aibirds_tf215
    python scripts/smoke_tf215.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import keras
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.model.angry_birds import PretrainedConvNeXtTiny
from src.agent.model.q_network import DistributionalDuelingQNetwork


def main():
    print("python executable:", os.sys.executable)
    print("tensorflow:", tf.__version__)
    print("keras:", keras.__version__)
    print("tf built with cuda:", tf.test.is_built_with_cuda())
    print("physical gpus:", tf.config.list_physical_devices("GPU"))
    print("logical gpus:", tf.config.list_logical_devices("GPU"))

    stem = PretrainedConvNeXtTiny(weights=None, trainable_backbone=False)
    inputs, latent = stem.get_functional_graph([(128, 128, 3), (5,)])

    q_net = DistributionalDuelingQNetwork(
        latent_v_dim=256,
        latent_a_dim=256,
        noise_std_init=0.5,
        activation="relu",
        num_atoms=51,
        v_min=-10.0,
        v_max=30.0,
    )
    q_net.set_num_actions(200)
    q_net.set_sequential(False)

    outputs = q_net(latent)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    sample = [
        tf.zeros((1, 128, 128, 3), dtype=tf.float32),
        tf.zeros((1, 5), dtype=tf.float32),
    ]
    y = model(sample, training=False)
    print("rainbow output shape:", y.shape)
    print("first action probability sum:", float(tf.reduce_sum(y[0, 0]).numpy()))

    tmp_dir = tempfile.mkdtemp(prefix="ab_tf215_smoke_")
    try:
        weights_path = os.path.join(tmp_dir, "model.weights.h5")
        saved_model_path = os.path.join(tmp_dir, "saved_model")
        model.save_weights(weights_path)
        model.load_weights(weights_path)
        tf.saved_model.save(model, saved_model_path)
        loaded = tf.saved_model.load(saved_model_path)
        print("savedmodel signatures:", sorted(loaded.signatures.keys()))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("smoke_tf215: OK")


if __name__ == "__main__":
    main()
