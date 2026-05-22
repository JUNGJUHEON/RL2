"""
학습 스크립트
=========================================
"""

import os
from datetime import datetime

import numpy as np


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int_or_none(name: str, default=None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    return float(raw)


def _env_bool(name: str, default=False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────
# 하드웨어 설정 (GPU 있으면 True, 없으면 False)
# ─────────────────────────────────────────────────────────────
USE_GPU = True
GPU_MEMORY_LIMIT_MB = None  # None = TensorFlow can use the full RTX GPU memory it needs.
# Configure CUDA visibility before importing modules that create TensorFlow objects.
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if USE_GPU else "-1"

from src.utils.utils import setup_hardware, split_params

HARDWARE_STATUS = setup_hardware(use_gpu=USE_GPU, gpu_memory_limit=GPU_MEMORY_LIMIT_MB)

from src.agent.agent import Agent, continue_practice, restore
from src.utils.params import ParamScheduler
from src.envs.angry_birds import AngryBirds
import src.agent.model as model

# =============================================================
#  실행 모드 선택
# =============================================================
MODE = _env_str("AB_MODE", "new")          # "new"      : 처음부터 새로 학습
                                           # "resume"   : 기존 체크포인트에서 이어서 학습
                                           # "evaluate" : 체크포인트로 평가만 실행 (학습 없음)

# 이어서 학습할 경우에만 사용 (MODE = "resume" 또는 "evaluate" 일 때)
# "auto"이면 out/angry_birds/ 아래에서 checkpoint step이 가장 큰 run을 자동 선택합니다.
# 특정 run을 쓰려면 폴더 이름을 직접 입력하세요.
RESUME_MODEL = _env_str("AB_RESUME_MODEL", "auto")
RESUME_CHECKPOINT_NO = _env_int_or_none("AB_RESUME_CHECKPOINT_NO")  # None = 가장 최신 체크포인트 자동 선택
RESUME_TARGET_STEPS = _env_int_or_none("AB_RESUME_TARGET_STEPS")
RESTORE_CONVNEXT_TRAINABLE_BACKBONE = os.environ.get("AB_RESTORE_CONVNEXT_TRAINABLE_BACKBONE")

# evaluate 모드 옵션: None 이면 가장 최신 체크포인트 자동 선택
# 특정 step의 체크포인트를 쓰려면 숫자로 지정 (예: 25000)
EVAL_CHECKPOINT_NO = _env_int_or_none("AB_EVAL_CHECKPOINT_NO")


def _checkpoint_step_from_name(file_name: str):
    for suffix in (".weights.h5", ".index"):
        if file_name.endswith(suffix):
            step_text = file_name[:-len(suffix)]
            break
    else:
        return None

    try:
        return int(step_text)
    except ValueError:
        return None


def find_auto_resume_model(root="out/angry_birds"):
    candidates = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No run directory found: {root}")

    for run_name in os.listdir(root):
        run_dir = os.path.join(root, run_name)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.isdir(checkpoint_dir):
            continue

        best_step = None
        best_mtime = 0.0
        for file_name in os.listdir(checkpoint_dir):
            step = _checkpoint_step_from_name(file_name)
            if step is None:
                continue
            path = os.path.join(checkpoint_dir, file_name)
            mtime = os.path.getmtime(path)
            if best_step is None or step > best_step or (step == best_step and mtime > best_mtime):
                best_step = step
                best_mtime = mtime

        if best_step is not None:
            candidates.append((best_step, best_mtime, run_name))

    if not candidates:
        raise FileNotFoundError(f"No checkpointed runs found under {root}")

    best_step, _best_mtime, run_name = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    print(f"Auto resume selected: {run_name} (checkpoint step {best_step})")
    return run_name


def resolve_resume_model(model_name: str):
    if model_name is None or str(model_name).lower() in {"auto", "latest", "best", "max"}:
        return find_auto_resume_model()
    return model_name

# =============================================================
#  모델 / 알고리즘 프리셋
# =============================================================
# 여기만 바꾸면 모델을 쉽게 전환할 수 있습니다.
MODEL_PRESET = _env_str("AB_MODEL_PRESET", "convnext_tiny_pretrained")

# ConvNeXt fine-tuning 옵션. 첫 안정화 run은 False 권장.
CONVNEXT_TRAINABLE_BACKBONE = _env_str("AB_CONVNEXT_TRAINABLE_BACKBONE", "0").lower() in {
    "1", "true", "yes", "on"
}
CONVNEXT_FINETUNE_AT_STEP = _env_int_or_none("AB_CONVNEXT_FINETUNE_AT_STEP", 50000)
CONVNEXT_FINETUNE_LR_SCALE = _env_float("AB_CONVNEXT_FINETUNE_LR_SCALE", 0.1)

# Rainbow A/B/C/D/E/F experiment knob.
# F_base is QR-Rainbow plus train-only proxy reward and ConvNeXt backbone
# gradient unfreezing at 50k. F_aux adds train-only auxiliary heads.
RAINBOW_RUN_VARIANT = _env_str("AB_RAINBOW_RUN_VARIANT", "F_base")

TRAINING_SIZE_PRESET = _env_str("AB_TRAINING_SIZE_PRESET", "f100")
TRAINING_SIZE_PRESETS = {
    "smoke": {
        "num_parallel_steps": 5000,
        "memory_size": 30000,
        "replay_batch_size": 16,
        "notes": "Crash check only.",
    },
    "diagnostic": {
        "num_parallel_steps": 10000,
        "memory_size": 30000,
        "replay_batch_size": 16,
        "notes": "Short diagnostics sanity run.",
    },
    "compare": {
        "num_parallel_steps": 50000,
        "memory_size": 30000,
        "replay_batch_size": 16,
        "notes": "A/B/C/D comparison size.",
    },
    "full": {
        "num_parallel_steps": 200000,
        "memory_size": 100000,
        "replay_batch_size": 32,
        "notes": "First serious full-performance run for RTX 4060 Ti class hardware.",
    },
    "f100": {
        "num_parallel_steps": 100000,
        "memory_size": 100000,
        "replay_batch_size": 32,
        "notes": "Model F target: 100k steps, 100k replay, batch 32.",
    },
}

MODEL_EXPERIMENT_PROFILE = _env_str("AB_MODEL_EXPERIMENT_PROFILE", "proxy_reward_v1")
MODEL_EXPERIMENT_PROFILES = {
    "baseline_compat": {
        "slug": "baseline",
        "label": "Baseline-compatible: filtered maps, square resize, legacy score-delta reward",
        "env": {
            "AB_REWARD_PROFILE": "legacy_delta",
            "AB_TRAIN_LEVEL_POOL": "filtered",
        },
    },
    "reward_fix_v1": {
        "slug": "reward_fix_v1",
        "label": "Reward fix: guard score resets and reward per-level best-score improvements",
        "env": {
            "AB_REWARD_PROFILE": "clamped_delta_best",
            "AB_TRAIN_LEVEL_POOL": "filtered",
            "AB_LEVEL_BEST_SCORE_BONUS_SCALE": "0.25",
        },
    },
    "allmap_reward_v1": {
        "slug": "allmap_reward_v1",
        "label": "All-map reward run: all 400 maps with guarded score-delta reward",
        "env": {
            "AB_REWARD_PROFILE": "clamped_delta_best",
            "AB_TRAIN_LEVEL_POOL": "all400",
            "AB_LEVEL_BEST_SCORE_BONUS_SCALE": "0.25",
        },
    },
    "proxy_reward_v1": {
        "slug": "proxy_reward_v1",
        "label": "Proxy-shaped reward: all 400 maps with tap-success and pig-like score bonuses",
        "env": {
            "AB_REWARD_PROFILE": "shaped_proxy_v1",
            "AB_TRAIN_LEVEL_POOL": "all400",
            "AB_LEVEL_BEST_SCORE_BONUS_SCALE": "0.25",
            "AB_TAP_SCORE_DELTA_BONUS": "0.03",
            "AB_TAP_WIN_BONUS": "0.10",
            "AB_PIG_PROXY_SCORE_THRESHOLD": "5000",
            "AB_PIG_PROXY_BONUS": "0.10",
            "AB_PIG_PROXY_MAX_BONUS": "0.30",
        },
    },
}

if MODEL_EXPERIMENT_PROFILE not in MODEL_EXPERIMENT_PROFILES:
    raise ValueError(
        f"MODEL_EXPERIMENT_PROFILE={MODEL_EXPERIMENT_PROFILE!r} is invalid. "
        f"Choose one of {list(MODEL_EXPERIMENT_PROFILES.keys())}."
    )

SELECTED_MODEL_EXPERIMENT = MODEL_EXPERIMENT_PROFILES[MODEL_EXPERIMENT_PROFILE]
for key, value in SELECTED_MODEL_EXPERIMENT["env"].items():
    os.environ.setdefault(key, str(value))

print(
    f"Model experiment profile: {MODEL_EXPERIMENT_PROFILE} "
    f"({SELECTED_MODEL_EXPERIMENT['label']})"
)
print(
    "Effective AB env toggles: "
    f"reward={os.environ.get('AB_REWARD_PROFILE')}, "
    f"image=square_resize, "
    f"train_pool={os.environ.get('AB_TRAIN_LEVEL_POOL')}, "
    f"best_bonus_scale={os.environ.get('AB_LEVEL_BEST_SCORE_BONUS_SCALE', 'default')}, "
    f"tap_score_bonus={os.environ.get('AB_TAP_SCORE_DELTA_BONUS', 'default')}, "
    f"tap_win_bonus={os.environ.get('AB_TAP_WIN_BONUS', 'default')}, "
    f"pig_proxy_bonus={os.environ.get('AB_PIG_PROXY_BONUS', 'default')}"
)

if TRAINING_SIZE_PRESET not in TRAINING_SIZE_PRESETS:
    raise ValueError(
        f"TRAINING_SIZE_PRESET={TRAINING_SIZE_PRESET!r} is invalid. "
        f"Choose one of {list(TRAINING_SIZE_PRESETS.keys())}."
    )

SELECTED_TRAINING_SIZE = TRAINING_SIZE_PRESETS[TRAINING_SIZE_PRESET]
print(
    f"Training size preset: {TRAINING_SIZE_PRESET} "
    f"({SELECTED_TRAINING_SIZE['num_parallel_steps']} steps, "
    f"memory={SELECTED_TRAINING_SIZE['memory_size']}, "
    f"batch={SELECTED_TRAINING_SIZE['replay_batch_size']})"
)

LEARNING_RATE_INIT = _env_float("AB_LEARNING_RATE_INIT", 0.0003)
LEARNING_RATE_HALF_LIFE = _env_int_or_none("AB_LEARNING_RATE_HALF_LIFE", 100000)
LEARNING_RATE_MIN = _env_float("AB_LEARNING_RATE_MIN", 0.0)
RESUME_LEARNING_RATE_INIT = os.environ.get("AB_RESUME_LEARNING_RATE_INIT")
RESUME_LEARNING_RATE_HALF_LIFE = _env_int_or_none("AB_RESUME_LEARNING_RATE_HALF_LIFE", LEARNING_RATE_HALF_LIFE)
RESUME_LEARNING_RATE_MIN = _env_float("AB_RESUME_LEARNING_RATE_MIN", LEARNING_RATE_MIN)

RAINBOW_RUN_VARIANTS = {
    "A": {
        "slug": "rainbow_A_c51_n1_no_noisy",
        "label": "A: ConvNeXtTiny + Double + Dueling + PER + C51, n_step=1, NoisyNet OFF",
        "q_head_preset": "c51_distributional",
        "n_step": 1,
        "noise_std_init": 0.0,
        "epsilon_init": 1.0,
        "epsilon_min": 0.10,
        "gamma": 0.99,
    },
    "B": {
        "slug": "rainbow_B_c51_n3_no_noisy",
        "label": "B: ConvNeXtTiny + Double + Dueling + PER + C51, n_step=3, NoisyNet OFF",
        "q_head_preset": "c51_distributional",
        "n_step": 3,
        "noise_std_init": 0.0,
        "epsilon_init": 1.0,
        "epsilon_min": 0.10,
        "gamma": 0.99,
    },
    "C": {
        "slug": "rainbow_C_c51_n1_noisy",
        "label": "C: ConvNeXtTiny + Double + Dueling + PER + C51, n_step=1, NoisyNet ON",
        "q_head_preset": "full_rainbow",
        "n_step": 1,
        "noise_std_init": 0.5,
        "epsilon_init": 0.0,
        "epsilon_min": 0.0,
        "gamma": 0.99,
    },
    "D": {
        "slug": "rainbow_D_full_n3_noisy",
        "label": "D: ConvNeXtTiny full Rainbow, n_step=3, NoisyNet ON",
        "q_head_preset": "full_rainbow",
        "n_step": 3,
        "noise_std_init": 0.5,
        "epsilon_init": 0.0,
        "epsilon_min": 0.0,
        "gamma": 0.99,
    },
    "E": {
        "slug": "rainbow_E_final_safe_proxy",
        "label": "E: final-safe full Rainbow, n_step=3, NoisyNet ON, all400 + proxy reward",
        "q_head_preset": "full_rainbow",
        "n_step": 3,
        "noise_std_init": 0.5,
        "epsilon_init": 0.0,
        "epsilon_min": 0.0,
        "gamma": 0.99,
    },
    "F": {
        "slug": "rainbow_F_base_qr_n3_noisy_eps_finetune50k",
        "label": "F_base alias: QR-Rainbow, n_step=3, NoisyNet ON, epsilon 0.05->0.01, ConvNeXt fine-tune at 50k",
        "q_head_preset": "qr_rainbow",
        "n_step": 3,
        "noise_std_init": 0.5,
        "epsilon_init": 0.05,
        "epsilon_min": 0.01,
        "gamma": 0.99,
        "convnext_finetune": True,
        "auxiliary_heads": False,
    },
    "F_base": {
        "slug": "rainbow_F_base_qr_n3_noisy_eps_finetune50k",
        "label": "F_base: QR-Rainbow, n_step=3, NoisyNet ON, epsilon 0.05->0.01, ConvNeXt fine-tune at 50k",
        "q_head_preset": "qr_rainbow",
        "n_step": 3,
        "noise_std_init": 0.5,
        "epsilon_init": 0.05,
        "epsilon_min": 0.01,
        "gamma": 0.99,
        "convnext_finetune": True,
        "auxiliary_heads": False,
    },
    "F_aux": {
        "slug": "rainbow_F_aux_qr_n3_noisy_eps_finetune50k",
        "label": "F_aux: F_base plus train-only auxiliary outcome heads",
        "q_head_preset": "qr_rainbow",
        "n_step": 3,
        "noise_std_init": 0.5,
        "epsilon_init": 0.05,
        "epsilon_min": 0.01,
        "gamma": 0.99,
        "convnext_finetune": True,
        "auxiliary_heads": True,
    },
}

if RAINBOW_RUN_VARIANT not in RAINBOW_RUN_VARIANTS:
    raise ValueError(
        f"RAINBOW_RUN_VARIANT={RAINBOW_RUN_VARIANT!r} is invalid. "
        f"Choose one of {list(RAINBOW_RUN_VARIANTS.keys())}."
    )

SELECTED_RAINBOW_RUN = RAINBOW_RUN_VARIANTS[RAINBOW_RUN_VARIANT]
Q_HEAD_PRESET = SELECTED_RAINBOW_RUN["q_head_preset"]

# Full Rainbow knobs. These are derived from RAINBOW_RUN_VARIANT above.
FULL_RAINBOW_N_STEP = SELECTED_RAINBOW_RUN["n_step"]
FULL_RAINBOW_NOISE_STD_INIT = SELECTED_RAINBOW_RUN["noise_std_init"]
FULL_RAINBOW_NUM_ATOMS = 51
FULL_RAINBOW_NUM_QUANTILES = int(_env_str("AB_QR_NUM_QUANTILES", "51"))
FULL_RAINBOW_V_MIN = -10.0
FULL_RAINBOW_V_MAX = 30.0
USE_DISTRIBUTIONAL_RAINBOW = Q_HEAD_PRESET in ("c51_distributional", "full_rainbow", "qr_rainbow")
USE_NOISYNET = FULL_RAINBOW_NOISE_STD_INIT > 0
USE_FULL_RAINBOW = USE_DISTRIBUTIONAL_RAINBOW and USE_NOISYNET and FULL_RAINBOW_N_STEP >= 3
USE_QR_RAINBOW = Q_HEAD_PRESET == "qr_rainbow"
EFFECTIVE_CONVNEXT_TRAINABLE_BACKBONE = (
    CONVNEXT_TRAINABLE_BACKBONE or
    (SELECTED_RAINBOW_RUN.get("convnext_finetune", False) and CONVNEXT_FINETUNE_AT_STEP is not None)
)
AUXILIARY_HEADS_CONFIG = None
if SELECTED_RAINBOW_RUN.get("auxiliary_heads", False):
    AUXILIARY_HEADS_CONFIG = {
        "enabled": True,
        "loss_weight": _env_float("AB_AUXILIARY_LOSS_WEIGHT", 0.05),
        "hidden_dim": _env_int_or_none("AB_AUXILIARY_HIDDEN_DIM", 128),
        "heads": [
            {"name": "reward_norm", "type": "regression", "weight": 0.5},
            {"name": "score_delta_norm", "type": "regression", "weight": 1.0},
            {"name": "terminal", "type": "binary", "weight": 0.5},
            {"name": "win", "type": "binary", "weight": 1.0},
            {"name": "positive_score_delta", "type": "binary", "weight": 0.5},
        ],
    }

MODEL_PRESET_INFO = {
    "classic_conv": {
        "status": "implemented",
        "type": "custom CNN stem",
        "notes": "Original project stem. Input convention is 0..1 after env.preprocess().",
    },
    "convnext_tiny_pretrained": {
        "status": "implemented",
        "type": "Keras/TensorFlow ConvNeXtTiny ImageNet stem",
        "notes": "Chosen current run. Keeps output shape compatible with DQN Q-head.",
    },
    "convnext_tiny_scratch": {
        "status": "implemented",
        "type": "Keras/TensorFlow ConvNeXtTiny stem without ImageNet weights",
        "notes": "Same architecture but random visual features; usually worse for limited RL data.",
    },
    "convnext_tiny_pretrained_finetune": {
        "status": "implemented",
        "type": "Pretrained ConvNeXtTiny with trainable backbone",
        "notes": "Higher capacity, slower, more overfitting/instability risk.",
    },
    "keras_convnextv2_style_scratch": {
        "status": "not_implemented",
        "type": "custom Keras ConvNeXtV2-style encoder",
        "notes": "Discussed, but not added because pretrained ConvNeXtTiny is safer.",
    },
    "pytorch_timm_convnextv2_pretrained": {
        "status": "not_implemented",
        "type": "PyTorch/timm pretrained ConvNeXtV2",
        "notes": "Dummy-tested only. Not safe for TensorFlow SavedModel evaluation path.",
    },
}

Q_HEAD_PRESET_INFO = {
    "dueling_dqn": {
        "status": "implemented",
        "type": "DoubleQNetwork dueling-style Q head",
        "notes": "Current default; outputs scalar Q-values with shape (batch, 200).",
    },
    "vanilla_dqn": {
        "status": "implemented",
        "type": "simple dense Q head",
        "notes": "Smaller baseline; outputs scalar Q-values with shape (batch, 200).",
    },
    "noisy_dueling_dqn": {
        "status": "implemented_experimental",
        "type": "dueling Q head using NoisyDense",
        "notes": "Scalar-Q NoisyNet ablation. full_rainbow is the complete Rainbow preset.",
    },
    "c51_distributional": {
        "status": "implemented_experimental",
        "type": "C51 distributional Q head",
        "notes": "A/B ablation head. Dueling C51 without NoisyNet; Agent still uses true Double DQN target.",
    },
    "full_rainbow": {
        "status": "implemented_experimental",
        "type": "Rainbow DQN head: dueling C51 with NoisyDense",
        "notes": "C/D head. Dueling C51 with NoisyNet; Agent uses true Double DQN target and PER.",
    },
    "qr_rainbow": {
        "status": "implemented_experimental",
        "type": "QR-DQN Rainbow head: dueling quantile regression with NoisyDense",
        "notes": "Model F head. Outputs quantile values; SavedModel exports expected Q-values for evaluator compatibility.",
    },
    "iqn": {
        "status": "not_implemented",
        "type": "implicit quantile network distributional RL head",
        "notes": "More advanced than QR-DQN; still not wired for this compatibility path.",
    },
    "fqf": {
        "status": "not_implemented",
        "type": "learned quantile fraction distributional RL head",
        "notes": "More advanced than IQN; not suitable for this quick project path.",
    },
}

RL_COMPONENT_STATUS = {
    "DQN": {
        "status": "implemented",
        "notes": "Scalar-Q path remains available through dueling_dqn / vanilla_dqn presets.",
    },
    "target_network": {
        "status": "implemented",
        "notes": "target_learner exists and syncs every target_sync_period.",
    },
    "Double_DQN": {
        "status": "implemented_for_distributional_rainbow",
        "notes": "A/B/C/D select next action with online network and evaluate it with target network.",
    },
    "Dueling_DQN": {
        "status": "implemented",
        "notes": "DoubleQNetwork splits V(state) and A(state, action).",
    },
    "Prioritized_Replay": {
        "status": "implemented",
        "notes": "ReplayMemory samples by priority and updates priorities from TD error.",
    },
    "n_step_return": {
        "status": "implemented",
        "notes": f"Replay supports n_step; selected run uses n_step={FULL_RAINBOW_N_STEP}.",
    },
    "Rainbow_DQN": {
        "status": "implemented_experimental_with_ablation_knob",
        "notes": "A/B/C/D run matrix toggles n_step and NoisyNet; D is the closest full Rainbow run.",
    },
    "C51": {
        "status": "implemented_experimental",
        "notes": f"Uses {FULL_RAINBOW_NUM_ATOMS} atoms on [{FULL_RAINBOW_V_MIN}, {FULL_RAINBOW_V_MAX}].",
    },
    "QR_DQN": {
        "status": "implemented_experimental" if USE_QR_RAINBOW else "available_not_selected",
        "notes": f"Uses {FULL_RAINBOW_NUM_QUANTILES} quantiles and exports mean Q-values for the existing evaluator.",
    },
    "NoisyNet": {
        "status": "on" if USE_NOISYNET else "off_for_selected_ablation",
        "notes": f"Selected run uses sigma0={FULL_RAINBOW_NOISE_STD_INIT}; paper-style NoisyNet uses sigma0=0.5 and epsilon=0.",
    },
    "R2D2": {"status": "not_implemented", "notes": "Sequence/LSTM code has TODOs."},
    "NGU": {"status": "not_implemented", "notes": "No intrinsic novelty reward yet."},
    "Agent57": {"status": "not_implemented", "notes": "Too large for this project path."},
    "PPO": {"status": "not_implemented", "notes": "Different actor-critic pipeline."},
    "SAC_TD3": {"status": "not_implemented", "notes": "Continuous-control methods; not natural for 200 discrete actions."},
    "DreamerV3": {"status": "not_implemented", "notes": "Would require a world-model rewrite."},
    "MuZero": {"status": "not_implemented", "notes": "Would require latent dynamics and planning/MCTS."},
}


def build_stem_network(preset: str):
    """Builds the selected visual/state stem. Not-implemented presets fail loudly."""
    if preset == "classic_conv":
        return model.angry_birds.ClassicConv(latent_dim=256)
    if preset == "convnext_tiny_pretrained":
        return model.angry_birds.PretrainedConvNeXtTiny(
            image_feature_dim=512,
            bird_embed_dim=32,
            weights="imagenet",
            trainable_backbone=EFFECTIVE_CONVNEXT_TRAINABLE_BACKBONE,
            dropout_rate=0.0,
        )
    if preset == "convnext_tiny_scratch":
        return model.angry_birds.PretrainedConvNeXtTiny(
            image_feature_dim=512,
            bird_embed_dim=32,
            weights=None,
            trainable_backbone=True,
            dropout_rate=0.0,
        )
    if preset == "convnext_tiny_pretrained_finetune":
        return model.angry_birds.PretrainedConvNeXtTiny(
            image_feature_dim=512,
            bird_embed_dim=32,
            weights="imagenet",
            trainable_backbone=True,
            dropout_rate=0.0,
        )
    raise ValueError(
        f"MODEL_PRESET={preset!r} is not implemented. "
        f"Available implemented presets: "
        f"{[k for k, v in MODEL_PRESET_INFO.items() if v['status'].startswith('implemented')]}"
    )


def build_q_network(preset: str):
    """Builds the selected Q / distributional head."""
    if preset == "dueling_dqn":
        return model.q_network.DoubleQNetwork()
    if preset == "vanilla_dqn":
        return model.q_network.VanillaQNetwork()
    if preset == "noisy_dueling_dqn":
        return model.q_network.DoubleQNetwork(
            latent_v_dim=256,
            latent_a_dim=256,
            noise_std_init=0.5,
            activation="relu",
        )
    if preset == "c51_distributional":
        return model.q_network.DistributionalDuelingQNetwork(
            latent_v_dim=256,
            latent_a_dim=256,
            noise_std_init=0.0,
            activation="relu",
            num_atoms=FULL_RAINBOW_NUM_ATOMS,
            v_min=FULL_RAINBOW_V_MIN,
            v_max=FULL_RAINBOW_V_MAX,
        )
    if preset == "full_rainbow":
        return model.q_network.DistributionalDuelingQNetwork(
            latent_v_dim=256,
            latent_a_dim=256,
            noise_std_init=FULL_RAINBOW_NOISE_STD_INIT,
            activation="relu",
            num_atoms=FULL_RAINBOW_NUM_ATOMS,
            v_min=FULL_RAINBOW_V_MIN,
            v_max=FULL_RAINBOW_V_MAX,
        )
    if preset == "qr_rainbow":
        return model.q_network.QuantileDuelingQNetwork(
            latent_v_dim=256,
            latent_a_dim=256,
            noise_std_init=FULL_RAINBOW_NOISE_STD_INIT,
            activation="relu",
            num_quantiles=FULL_RAINBOW_NUM_QUANTILES,
        )
    raise ValueError(
        f"Q_HEAD_PRESET={preset!r} is not implemented. "
        f"Available implemented presets: "
        f"{[k for k, v in Q_HEAD_PRESET_INFO.items() if v['status'].startswith('implemented')]}"
    )
# =============================================================

if MODE == "new":
    # ─────────────────────────────────────────────────────────
    # 처음부터 새로 학습
    # ─────────────────────────────────────────────────────────
    seed = np.random.randint(int(1e9))
    np.random.seed(seed)

    env = AngryBirds(num_par_inst=1)

    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")  # 예: 20240414_153022
    RUN_NAME = (
        "ab_agent_"
        + RUN_TIMESTAMP
        + "_"
        + SELECTED_RAINBOW_RUN["slug"]
        + "_"
        + SELECTED_MODEL_EXPERIMENT["slug"]
    )

    hyperparams = {
        # ── 기본 설정 ──
        # 모델 이름에 타임스탬프 자동 포함 → out/angry_birds/ab_agent_20240414_153022/
        "name": RUN_NAME,
        "env": env,
        "seed": seed,

        # ── 학습 스텝 수 ──
        "num_parallel_steps": SELECTED_TRAINING_SIZE["num_parallel_steps"],

        # ── 신경망 모델 ──
        # ORIGINAL:
        # "stem_network": model.angry_birds.ClassicConv(latent_dim=256),
        #
        # UPDATED: MODEL_PRESET / Q_HEAD_PRESET로 쉽게 전환합니다.
        "stem_network": build_stem_network(MODEL_PRESET),
        "q_network": build_q_network(Q_HEAD_PRESET),
        "auxiliary_heads_config": AUXILIARY_HEADS_CONFIG,

        # ── 학습률 (Learning Rate) ──
        "learning_rate": ParamScheduler(
            init_value=LEARNING_RATE_INIT,
            decay_mode="exp",
            half_life_period=LEARNING_RATE_HALF_LIFE,
            minimum=LEARNING_RATE_MIN,
        ),

        # ── 탐험 정책 (Epsilon-Greedy) ──
        "epsilon": ParamScheduler(
            init_value=SELECTED_RAINBOW_RUN["epsilon_init"],
            decay_mode="exp",
            half_life_period=25000,
            minimum=SELECTED_RAINBOW_RUN["epsilon_min"],
        ),

        # ── 리플레이 메모리 ──
        # 권장 비율: memory_size ≈ 총 스텝의 60%
        # 너무 크게 설정하면 메모리 문제 발생 가능
        "memory_size": SELECTED_TRAINING_SIZE["memory_size"],
        # Replay batch is selected by TRAINING_SIZE_PRESET.
        # RTX 4060 Ti class GPUs should usually handle 32 for this 128x128 ConvNeXtTiny path.
        "replay_batch_size": SELECTED_TRAINING_SIZE["replay_batch_size"],
        "replay_period": 4,
        "replay_size_multiplier": 4,

        # ── 보상 할인율 ──
        "gamma": SELECTED_RAINBOW_RUN["gamma"],
        "alpha": 0.7,
        "n_step": FULL_RAINBOW_N_STEP,

        # ── 타겟 네트워크 동기화 ──
        # 설정한 스텝마다 동기화해 Q값 oscillation 방지.
        "target_sync_period": 500,

        # ── 진단 로그 ──
        # out/angry_birds/<run>/diagnostics.jsonl 에 JSON Lines로 저장됩니다.
        "diagnostics_period": 100,
        "diagnostics_full_period": 500,
        "diagnostics_profile": "full",
        # Human/GPT-readable CSV + PNG reports under:
        # out/angry_birds/<run>/diagnostic_reports/checkpoint_<step>/
        "checkpoint_diagnostics": True,
        "convnext_finetune_at_step": (
            CONVNEXT_FINETUNE_AT_STEP if SELECTED_RAINBOW_RUN.get("convnext_finetune", False) else None
        ),
        "convnext_finetune_lr_scale": CONVNEXT_FINETUNE_LR_SCALE,
        "use_tqdm": True,

        # ── Codex/GPT 친화 run metadata ──
        "run_metadata": {
            "hardware": {
                "use_gpu": USE_GPU,
                "gpu_memory_limit_mb": GPU_MEMORY_LIMIT_MB,
                "runtime_status": HARDWARE_STATUS,
                "expected_machine": "Ubuntu + NVIDIA RTX 4060 Ti + Intel Core Ultra 9 285K",
            },
            "training_size_preset": TRAINING_SIZE_PRESET,
            "training_size_preset_info": SELECTED_TRAINING_SIZE,
            "training_size_presets": TRAINING_SIZE_PRESETS,
            "rainbow_run_variant": RAINBOW_RUN_VARIANT,
            "rainbow_run_variant_info": SELECTED_RAINBOW_RUN,
            "rainbow_run_variants": RAINBOW_RUN_VARIANTS,
            "model_experiment_profile": MODEL_EXPERIMENT_PROFILE,
            "model_experiment_profile_info": SELECTED_MODEL_EXPERIMENT,
            "model_experiment_profiles": MODEL_EXPERIMENT_PROFILES,
            "effective_ab_env": {
                "reward_profile": os.environ.get("AB_REWARD_PROFILE"),
                "image_preprocess": "square_resize",
                "train_level_pool": os.environ.get("AB_TRAIN_LEVEL_POOL"),
                "level_best_score_bonus_scale": os.environ.get("AB_LEVEL_BEST_SCORE_BONUS_SCALE"),
                "tap_score_delta_bonus": os.environ.get("AB_TAP_SCORE_DELTA_BONUS"),
                "tap_win_bonus": os.environ.get("AB_TAP_WIN_BONUS"),
                "pig_proxy_score_threshold": os.environ.get("AB_PIG_PROXY_SCORE_THRESHOLD"),
                "pig_proxy_bonus": os.environ.get("AB_PIG_PROXY_BONUS"),
                "pig_proxy_max_bonus": os.environ.get("AB_PIG_PROXY_MAX_BONUS"),
            },
            "model_preset": MODEL_PRESET,
            "convnext_trainable_backbone_requested": CONVNEXT_TRAINABLE_BACKBONE,
            "convnext_trainable_backbone_effective": EFFECTIVE_CONVNEXT_TRAINABLE_BACKBONE,
            "convnext_finetune_at_step": (
                CONVNEXT_FINETUNE_AT_STEP if SELECTED_RAINBOW_RUN.get("convnext_finetune", False) else None
            ),
            "convnext_finetune_lr_scale": CONVNEXT_FINETUNE_LR_SCALE,
            "auxiliary_heads_enabled": AUXILIARY_HEADS_CONFIG is not None,
            "auxiliary_heads_config": AUXILIARY_HEADS_CONFIG,
            "learning_rate": {
                "init": LEARNING_RATE_INIT,
                "half_life_period": LEARNING_RATE_HALF_LIFE,
                "minimum": LEARNING_RATE_MIN,
            },
            "q_head_preset": Q_HEAD_PRESET,
            "model_preset_info": MODEL_PRESET_INFO,
            "q_head_preset_info": Q_HEAD_PRESET_INFO,
            "rl_component_status": RL_COMPONENT_STATUS,
            "distributional_rainbow_enabled": USE_DISTRIBUTIONAL_RAINBOW,
            "noisynet_enabled": USE_NOISYNET,
            "full_rainbow_enabled": USE_FULL_RAINBOW,
            "full_rainbow_config": {
                "n_step": FULL_RAINBOW_N_STEP,
                "noise_std_init": FULL_RAINBOW_NOISE_STD_INIT,
                "num_atoms": FULL_RAINBOW_NUM_ATOMS,
                "num_quantiles": FULL_RAINBOW_NUM_QUANTILES,
                "v_min": FULL_RAINBOW_V_MIN,
                "v_max": FULL_RAINBOW_V_MAX,
                "double_dqn": True,
                "dueling": True,
                "prioritized_replay": True,
                "c51": Q_HEAD_PRESET in ("c51_distributional", "full_rainbow"),
                "qr_dqn": USE_QR_RAINBOW,
                "noisynet": USE_NOISYNET,
                "epsilon_greedy_init": SELECTED_RAINBOW_RUN["epsilon_init"],
                "epsilon_greedy_min": SELECTED_RAINBOW_RUN["epsilon_min"],
            },
            "current_honest_name": SELECTED_RAINBOW_RUN["label"],
            "target_direction": "Model F: QR-Rainbow with proxy reward and scheduled ConvNeXt fine-tuning",
            "not_implemented_warning": (
                "IQN/FQF, R2D2, NGU, Agent57, PPO, SAC/TD3, Dreamer, and MuZero are listed "
                "for transparency but are not active in this run."
            ),
        },
    }

    hparams_agent, hparams_practice = split_params(hyperparams, Agent.__init__)
    agent = Agent(**hparams_agent)
    agent.practice(**hparams_practice)

elif MODE == "resume":
    # ─────────────────────────────────────────────────────────
    # 체크포인트에서 이어서 학습
    # 환경은 continue_practice 내부에서 생성하므로 여기선 만들지 않음
    # ─────────────────────────────────────────────────────────
    resume_model = resolve_resume_model(RESUME_MODEL)
    resume_overrides = {}
    stem_config_override = None
    if RESUME_TARGET_STEPS is not None:
        resume_overrides["num_parallel_steps"] = int(RESUME_TARGET_STEPS)
    if RESUME_LEARNING_RATE_INIT is not None and RESUME_LEARNING_RATE_INIT != "":
        resume_overrides["learning_rate"] = ParamScheduler(
            init_value=float(RESUME_LEARNING_RATE_INIT),
            decay_mode="exp",
            half_life_period=RESUME_LEARNING_RATE_HALF_LIFE,
            minimum=RESUME_LEARNING_RATE_MIN,
        )
    if RESTORE_CONVNEXT_TRAINABLE_BACKBONE is not None and RESTORE_CONVNEXT_TRAINABLE_BACKBONE != "":
        stem_config_override = {
            "trainable_backbone": _env_bool("AB_RESTORE_CONVNEXT_TRAINABLE_BACKBONE", False)
        }
    continue_practice(
        resume_model,
        AngryBirds,
        checkpoint_no=RESUME_CHECKPOINT_NO,
        stem_config_override=stem_config_override,
        **resume_overrides,
    )

elif MODE == "evaluate":
    # ─────────────────────────────────────────────────────────
    # 체크포인트에서 모델을 복원해 평가 (학습 없음)
    # saved_model 없이 checkpoints/ 파일만 있어도 동작합니다.
    # ─────────────────────────────────────────────────────────
    resume_model = resolve_resume_model(RESUME_MODEL)
    stem_config_override = None
    if RESTORE_CONVNEXT_TRAINABLE_BACKBONE is not None and RESTORE_CONVNEXT_TRAINABLE_BACKBONE != "":
        stem_config_override = {
            "trainable_backbone": _env_bool("AB_RESTORE_CONVNEXT_TRAINABLE_BACKBONE", False)
        }
    agent = restore(
        resume_model,
        AngryBirds,
        checkpoint_no=EVAL_CHECKPOINT_NO,
        stem_config_override=stem_config_override,
    )
    agent.just_play(verbose=True)

else:
    raise ValueError("MODE는 'new', 'resume', 'evaluate' 중 하나여야 합니다. 현재 값: %r" % MODE)
