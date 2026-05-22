import time
import os
import json
import numpy as np
import signal
from typing import Type

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # let TF only print errors
import tensorflow as tf

from src.mem.mem import ReplayMemory
from src.envs.env import ParallelEnvironment
from src.agent.model import QNetwork, StemNetwork, DoubleQNetwork, get_class_from_name
from src.utils.params import ParamScheduler
from src.utils.stack import StateStacker
from src.utils.stats import Statistics, remove_folder
from src.utils.text_sty import print_warning, green, red, print_info, print_success
from src.utils.timer import Timer
from src.utils.utils import copy_object_with_config, plot_highscores, config2text, \
    ask_to_override_model, config2json, json2config, log_model_graph, random_choice_along_last_axis, \
    get_function_signature_list, serialize_function_parameters
from src.utils.ram import print_total_ram_usage, print_pympler_ram_usage, print_leaking_objects_count, \
    print_memory_block_summary, print_heap

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# Miscellaneous
PLOT_SAVE_STATS_PERIOD = 1000  # number of transitions between each learning statistics plot
PRINT_STATS_PERIOD = 100
CHECKPOINT_SAVE_PERIOD = 1000   # AngryBirds는 학습이 느려 자주 저장
EARLY_CHECKPOINT_SAVE_PERIOD = None
EARLY_CHECKPOINT_SAVE_UNTIL = 0
MAX_CHECKPOINTS_TO_KEEP = 3
TEST_PERIOD = 10000

KERAS_WEIGHTS_SUFFIX = ".weights.h5"
FINAL_WEIGHTS_FILENAME = "trained_model" + KERAS_WEIGHTS_SUFFIX

GRADIENT_DIAGNOSTIC_NAMES = (
    "total",
    "convnext_backbone",
    "classic_conv",
    "image_projection",
    "bird_embedding",
    "q_network",
    "distributional_q_network",
    "other",
)


def _checkpoint_filename(checkpoint_no: int) -> str:
    return f"{int(checkpoint_no):09d}{KERAS_WEIGHTS_SUFFIX}"


def _checkpoint_step_from_file_name(file_name: str):
    if file_name.endswith(KERAS_WEIGHTS_SUFFIX):
        step_text = file_name[:-len(KERAS_WEIGHTS_SUFFIX)]
    elif file_name.endswith(".index"):
        step_text = file_name[:-len(".index")]
    else:
        return None

    try:
        return int(step_text)
    except ValueError:
        return None


def _checkpoint_candidates(checkpoint_dir: str, checkpoint_no: int):
    step = int(checkpoint_no)
    return [
        os.path.join(checkpoint_dir, _checkpoint_filename(step)),
        os.path.join(checkpoint_dir, f"{step}{KERAS_WEIGHTS_SUFFIX}"),
        os.path.join(checkpoint_dir, str(step)),
    ]


def _checkpoint_temp_path(model_path: str) -> str:
    if model_path.endswith(KERAS_WEIGHTS_SUFFIX):
        return model_path[:-len(KERAS_WEIGHTS_SUFFIX)] + ".tmp" + KERAS_WEIGHTS_SUFFIX
    return model_path + ".tmp" + KERAS_WEIGHTS_SUFFIX


def _iter_checkpoint_files(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return []

    checkpoints = []
    for file_name in os.listdir(checkpoint_dir):
        step = _checkpoint_step_from_file_name(file_name)
        if step is None:
            continue
        if file_name.endswith(KERAS_WEIGHTS_SUFFIX):
            path = os.path.join(checkpoint_dir, file_name)
        else:
            path = os.path.join(checkpoint_dir, file_name[:-len(".index")])
        checkpoints.append((step, path))
    return sorted(checkpoints, reverse=True)


def _prune_old_checkpoints(checkpoint_dir: str, keep_count=MAX_CHECKPOINTS_TO_KEEP):
    checkpoints = [
        (step, path)
        for step, path in _iter_checkpoint_files(checkpoint_dir)
        if path.endswith(KERAS_WEIGHTS_SUFFIX)
    ]
    for _step, path in checkpoints[keep_count:]:
        try:
            os.remove(path)
            print_info(f"Removed old checkpoint to save disk space: {path}")
        except OSError as exc:
            print_warning(f"Warning: unable to remove old checkpoint {path}: {exc}")

    for file_name in os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []:
        if ".tmp" in file_name and file_name.endswith(KERAS_WEIGHTS_SUFFIX):
            path = os.path.join(checkpoint_dir, file_name)
            try:
                os.remove(path)
            except OSError:
                pass


def _resolve_checkpoint_paths(model_dir: str, checkpoint_no: int = None):
    checkpoint_dir = os.path.join(model_dir, "checkpoints")

    if checkpoint_no is not None:
        paths = []
        for path in _checkpoint_candidates(checkpoint_dir, checkpoint_no):
            if os.path.exists(path):
                paths.append(path)
            if os.path.exists(path + ".index"):
                paths.append(path)
        if paths:
            return paths
        raise FileNotFoundError(
            f"\n\n[오류] checkpoint {checkpoint_no} 파일을 찾을 수 없습니다.\n"
            f"경로: {checkpoint_dir}\n\n"
        )

    paths = []
    final_weights_path = os.path.join(model_dir, FINAL_WEIGHTS_FILENAME)
    if os.path.exists(final_weights_path):
        paths.append(final_weights_path)

    legacy_final_prefix = os.path.join(model_dir, "trained_model")
    if os.path.exists(legacy_final_prefix + ".index"):
        paths.append(legacy_final_prefix)

    checkpoint_paths = _iter_checkpoint_files(checkpoint_dir)
    if checkpoint_paths:
        latest_step, latest_path = checkpoint_paths[0]
        print(f"  (체크포인트 직접 탐색: step {latest_step}, {latest_path})")
        paths.extend(path for _step, path in checkpoint_paths)

    if paths:
        return paths

    raise FileNotFoundError(
        f"\n\n[오류] 저장된 체크포인트가 없습니다.\n"
        f"경로: {checkpoint_dir}\n\n"
        f"이 세션은 첫 체크포인트 저장 전에 종료된 것 같습니다.\n"
        f"train_ab.py 에서 MODE = \"new\" 로 변경하여 처음부터 시작하세요.\n"
    )


def _resolve_checkpoint_path(model_dir: str, checkpoint_no: int = None):
    return _resolve_checkpoint_paths(model_dir, checkpoint_no)[0]


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "get_config"):
        return obj.get_config()
    return str(obj)


class Agent:
    """Deep Q-Network (DQN) agent for playing Tetris, Snake or other games"""

    def __init__(self,
                 name: str,
                 env: ParallelEnvironment,
                 stem_network: StemNetwork,
                 q_network: QNetwork = DoubleQNetwork(),
                 replay_batch_size=512,
                 stack_size=1,
                 optimizer=tf.keras.optimizers.Adam(),
                 use_pretrained=False,
                 seed=None,
                 auxiliary_heads_config=None,
                 **kwargs):  # keep kwargs for backwards compatibility
        """Constructor
        :param env: The (instantiated) environment in which the agent acts
        :param stem_network: The main stem model
        :param q_network: The Q-network (coming after the stem model)
        :param name: A string identifying the agent in file names
        :param use_double: deprecated, but left for backwards compatibility
        """

        print("Initializing DQN agent...")

        # General
        self.name = name
        self.seed = seed
        self.policy = None

        self.model_dir = get_model_dir(env, name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        # Environment
        self.env = env
        self.num_par_envs = self.env.num_par_inst
        self.state_shapes = self.env.get_state_shapes()
        if hasattr(self.env, 'get_state_dtypes'):
            self.state_dtypes = self.env.get_state_dtypes()
        else:
            self.state_dtypes = [state_comp.dtype for state_comp in self.env.get_states()]
        self.num_actions = self.env.get_number_of_actions()

        # Frame stacking
        self.stack_size = stack_size
        self.stacker = StateStacker(self.state_shapes, self.state_dtypes, self.stack_size, self.num_par_envs)
        self.stack_shapes = self.stacker.get_stack_shapes()

        # For training
        self.replay_batch_size = replay_batch_size
        self.learning_rate = None
        self.epsilon = None
        self.delta = None
        self.use_mc_return = None  # Monte Carlo
        self.action_masking = None

        # Model architecture
        self.stem_network = stem_network
        self.sequential = stem_network.sequential
        self.sequence_len = stem_network.sequence_len
        self.q_network = q_network
        self.optimizer = optimizer
        self.auxiliary_heads_config = self._normalize_auxiliary_heads_config(auxiliary_heads_config)
        self.auxiliary_label_names = [head["name"] for head in self.auxiliary_heads_config["heads"]]
        self.auxiliary_label_types = [head["type"] for head in self.auxiliary_heads_config["heads"]]
        self.auxiliary_label_weights = np.asarray(
            [head["weight"] for head in self.auxiliary_heads_config["heads"]],
            dtype="float32",
        )
        self.auxiliary_output_dim = len(self.auxiliary_label_names)
        self.auxiliary_loss_weight = float(self.auxiliary_heads_config["loss_weight"])
        self.auxiliary_regression_mask = tf.constant(
            [label_type == "regression" for label_type in self.auxiliary_label_types],
            dtype=tf.bool,
        )
        self.auxiliary_label_weight_tensor = tf.constant(self.auxiliary_label_weights, dtype=tf.float32)
        self.online_learner = self.init_online_learner()
        self.q_net_layer_id = np.where([isinstance(layer, QNetwork) for layer in self.online_learner.layers])[0][0]
        # log_model_graph(self.online_learner, self.stack_shapes)

        if use_pretrained:
            self.load_pretrained_model()

        # Double Q-Learning and Distributed RL: by default identical to online_learner
        self.target_learner = self.online_learner
        self.actor = self.online_learner

        # Training loss
        self.training_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.training_loss_metric = tf.keras.metrics.MeanSquaredError()

        # Memory and statistics
        self.memory = None
        self.stats = Statistics(env=self.env, log_path=self.model_dir)
        self.diagnostics_path = os.path.join(self.model_dir, "diagnostics.jsonl")
        self._diagnostics_warning_printed = False
        self.last_plan_diagnostics = {}
        self.last_learning_diagnostics = {}
        self.last_gradient_diagnostics = {}
        self.activation_probe_model = None
        self.activation_probe_names = []
        self.action_counts = np.zeros(self.num_actions, dtype="int64")
        self.recent_rewards = []
        self.convnext_update_enabled = tf.Variable(True, trainable=False, dtype=tf.bool)
        self.convnext_gradient_scale = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self._last_convnext_update_enabled = None

        self.save_config()

        self.stop = False

        print('DQN agent initialized.')

    def init_online_learner(self) -> tf.keras.Model:
        model = self.init_model(self.stem_network, self.q_network, self.replay_batch_size)
        model.summary()

        # the following needs separate GraphViz installation from https://graphviz.gitlab.io/download/
        # this helped for GraphViz bugfix: https://datascience.stackexchange.com/questions/74500
        # tf.keras.utils.plot_model(model, to_file=self.out_path + 'model_plot.png', show_shapes=True,
        #                           show_layer_names=True, expand_nested=True, dpi=400)

        return model

    def init_model(self, stem_net: StemNetwork, q_net: QNetwork, batch_size):
        q_net.set_num_actions(self.num_actions)
        q_net.set_sequential(self.sequential)
        inputs, latent = stem_net.get_functional_graph(self.stack_shapes, batch_size)
        q_outputs = q_net(latent)
        if self.uses_auxiliary_heads():
            aux_latent = tf.keras.layers.Dense(
                int(self.auxiliary_heads_config.get("hidden_dim", 128)),
                activation="relu",
                name="auxiliary_latent",
            )(latent)
            aux_outputs = tf.keras.layers.Dense(
                self.auxiliary_output_dim,
                name="auxiliary_predictions",
            )(aux_latent)
            outputs = [q_outputs, aux_outputs]
        else:
            outputs = q_outputs
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='huber_loss', optimizer=self.optimizer)  # Huber loss equiv. to gradient clipping
        return model

    def copy_online_learner(self):
        stem_net = copy_object_with_config(self.stem_network)
        q_net = copy_object_with_config(self.q_network)
        model = self.init_model(stem_net, q_net, self.num_par_envs)
        model.set_weights(self.online_learner.get_weights())
        return model

    def load_pretrained_model(self):
        pretrained_path = "out/" + self.env.NAME + "/pretrained"
        if not os.path.exists(pretrained_path):
            Exception("You specified to load a pretrained model. However, there is no pretrained model "
                      "at '%s'." % pretrained_path)

        self.online_learner.load_weights(pretrained_path + "/pretrained", by_name=True)

    def _normalize_auxiliary_heads_config(self, config):
        if not config:
            return {"enabled": False, "loss_weight": 0.0, "hidden_dim": 128, "heads": []}

        heads = []
        for head in config.get("heads", []):
            label_type = head.get("type", "regression")
            if label_type not in {"regression", "binary"}:
                raise ValueError(f"Invalid auxiliary head type: {label_type!r}")
            heads.append({
                "name": str(head["name"]),
                "type": label_type,
                "weight": float(head.get("weight", 1.0)),
            })

        return {
            "enabled": bool(config.get("enabled", bool(heads))),
            "loss_weight": float(config.get("loss_weight", 0.05)),
            "hidden_dim": int(config.get("hidden_dim", 128)),
            "heads": heads,
        }

    def uses_auxiliary_heads(self):
        return bool(self.auxiliary_heads_config.get("enabled")) and self.auxiliary_output_dim > 0

    def _extract_q_output(self, model_output):
        if isinstance(model_output, (list, tuple)):
            return model_output[0]
        if isinstance(model_output, dict):
            return model_output.get("q", next(iter(model_output.values())))
        return model_output

    def _predict_q(self, keras_model, states, batch_size=None, verbose=0):
        model_output = keras_model.predict(states, batch_size=batch_size, verbose=verbose)
        return self._extract_q_output(model_output)

    def is_c51_distributional(self):
        return hasattr(self.q_network, "num_atoms") and hasattr(self.q_network, "get_support_np")

    def is_quantile_distributional(self):
        return hasattr(self.q_network, "num_quantiles")

    def is_distributional(self):
        return self.is_c51_distributional() or self.is_quantile_distributional()

    def uses_noisynet(self):
        return getattr(self.q_network, "noise_std_init", 0) > 0

    def get_distributional_support_np(self):
        if not self.is_c51_distributional():
            return None
        return self.q_network.get_support_np()

    def distribution_to_q_values(self, distributions):
        output = np.asarray(distributions)
        if self.is_c51_distributional():
            support = self.get_distributional_support_np()
            return np.sum(output * support, axis=-1)
        if self.is_quantile_distributional():
            return np.mean(output, axis=-1)
        return output

    def _diagnose_distributional_output(self, distributions):
        dist_np = np.asarray(distributions)
        if dist_np.size == 0:
            return {}
        clipped = np.clip(dist_np.astype("float32", copy=False), 1e-8, 1.0)
        entropy = -np.sum(clipped * np.log(clipped), axis=-1)
        return {
            "distribution": self._diagnose_array(dist_np),
            "atom_entropy": self._diagnose_array(entropy),
            "atom_probability_sums": self._diagnose_array(np.sum(dist_np, axis=-1)),
            "support": self._diagnose_array(self.get_distributional_support_np()),
        }

    def _diagnose_quantile_output(self, quantiles):
        quantile_np = np.asarray(quantiles)
        if quantile_np.size == 0:
            return {}
        q_values = np.mean(quantile_np, axis=-1)
        return {
            "quantiles": self._diagnose_array(quantile_np),
            "expected_q_values": self._diagnose_array(q_values),
            "quantile_std": self._diagnose_array(np.std(quantile_np, axis=-1)),
            "quantile_p10": self._diagnose_array(np.percentile(quantile_np, 10, axis=-1)),
            "quantile_p90": self._diagnose_array(np.percentile(quantile_np, 90, axis=-1)),
            "num_quantiles": int(quantile_np.shape[-1]),
        }

    def _build_auxiliary_labels(self, scores, rewards, terminals, wins, env_step_diag=None):
        if not self.uses_auxiliary_heads():
            return None

        score_delta = np.zeros(self.num_par_envs, dtype="float32")
        if isinstance(env_step_diag, dict) and "score_delta" in env_step_diag and self.num_par_envs == 1:
            score_delta[0] = float(env_step_diag.get("score_delta", 0.0))

        label_values = {
            "reward_norm": np.asarray(rewards, dtype="float32").reshape(self.num_par_envs) / 10.0,
            "score_delta_norm": score_delta / 10000.0,
            "terminal": np.asarray(terminals, dtype="float32").reshape(self.num_par_envs),
            "win": np.asarray(wins, dtype="float32").reshape(self.num_par_envs),
            "positive_score_delta": (score_delta > 0).astype("float32"),
        }

        return {
            name: label_values[name]
            for name in self.auxiliary_label_names
            if name in label_values
        }

    def _get_auxiliary_targets(self, trans_ids):
        if not self.uses_auxiliary_heads():
            return None
        labels = self.memory.get_aux_labels(trans_ids, self.auxiliary_label_names)
        missing = [name for name in self.auxiliary_label_names if name not in labels]
        if missing:
            raise KeyError(f"Missing auxiliary replay labels: {missing}")
        return np.stack(
            [np.asarray(labels[name], dtype="float32").reshape(len(trans_ids)) for name in self.auxiliary_label_names],
            axis=-1,
        ).astype("float32")

    def _compute_auxiliary_loss(self, aux_targets, aux_predictions):
        regression_errors = tf.square(aux_targets - aux_predictions)
        binary_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=aux_targets,
            logits=aux_predictions,
        )
        per_label_losses = tf.where(self.auxiliary_regression_mask, regression_errors, binary_losses)
        weighted_losses = per_label_losses * self.auxiliary_label_weight_tensor
        normalizer = tf.maximum(tf.reduce_sum(self.auxiliary_label_weight_tensor), 1.0)
        return tf.reduce_mean(tf.reduce_sum(weighted_losses, axis=-1) / normalizer)

    def _diagnose_array(self, arr, include_values=False):
        arr_np = np.asarray(arr)
        diag = {
            "shape": list(arr_np.shape),
            "dtype": str(arr_np.dtype),
            "nan_count": int(np.isnan(arr_np).sum()) if np.issubdtype(arr_np.dtype, np.floating) else 0,
            "inf_count": int(np.isinf(arr_np).sum()) if np.issubdtype(arr_np.dtype, np.floating) else 0,
        }
        if arr_np.size > 0:
            arr_float = arr_np.astype("float32", copy=False)
            diag.update({
                "min": float(np.min(arr_float)),
                "max": float(np.max(arr_float)),
                "mean": float(np.mean(arr_float)),
                "std": float(np.std(arr_float)),
            })
        if include_values:
            diag["values"] = arr_np.tolist()
            diag["sum"] = float(np.sum(arr_np.astype("float32", copy=False)))
        return diag

    def _diagnose_states(self, states):
        state_diag = []
        for idx, state_comp in enumerate(states):
            include_values = np.asarray(state_comp).size <= 32
            state_diag.append(self._diagnose_array(state_comp, include_values=include_values))
        return state_diag

    def _write_diagnostics(self, event, **payload):
        record = {
            "event": event,
            "wall_time": time.time(),
            "model": self.name,
            **payload,
        }
        try:
            with open(self.diagnostics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
        except Exception as exc:
            if not self._diagnostics_warning_printed:
                print_warning(f"Warning: unable to write diagnostics to {self.diagnostics_path}: {exc}")
                self._diagnostics_warning_printed = True

    def _write_json_artifact(self, filename, payload):
        out_path = os.path.join(self.model_dir, filename)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=_json_default, indent=2, ensure_ascii=False)
        except Exception as exc:
            print_warning(f"Warning: unable to write {out_path}: {exc}")

    def _export_checkpoint_diagnostics_report(self, checkpoint_label):
        report_dir = os.path.join(self.model_dir, "diagnostic_reports", str(checkpoint_label))
        try:
            from src.utils.diagnostic_report import export_training_diagnostics_report

            summary = export_training_diagnostics_report(
                diagnostics_path=self.diagnostics_path,
                output_dir=report_dir,
                checkpoint_label=str(checkpoint_label),
            )
            print_info(
                f"Saved diagnostic report to {report_dir} "
                f"({summary.get('convnext_status')}, {summary.get('rl_status')})."
            )
            return summary
        except Exception as exc:
            print_warning(f"Warning: unable to export checkpoint diagnostics report: {exc}")
            return None

    def _latest_loss(self):
        losses = self.stats.get_losses()
        if len(losses) == 0:
            return np.nan
        return float(losses[-1])

    def _diagnose_action_distribution(self):
        total = int(np.sum(self.action_counts))
        if total == 0:
            return {"total_actions": 0}
        probs = self.action_counts.astype("float64") / total
        nonzero = probs[probs > 0]
        entropy = -np.sum(nonzero * np.log(nonzero))
        top_ids = np.argsort(self.action_counts)[-10:][::-1]
        return {
            "total_actions": total,
            "unique_actions": int(np.count_nonzero(self.action_counts)),
            "entropy": float(entropy),
            "top10_actions": top_ids.astype("int").tolist(),
            "top10_counts": self.action_counts[top_ids].astype("int").tolist(),
        }

    def _diagnose_memory(self):
        if self.memory is None:
            return {}
        diag = {
            "size": int(self.memory.get_size()),
            "num_transitions": int(self.memory.get_num_transitions()),
            "num_learnable_transitions": int(self.memory.get_num_learnable_transitions()),
            "n_step": int(self.memory.n_step),
            "stack_size": int(self.memory.stack_size),
            "sequential": bool(self.memory.sequential),
        }
        try:
            priorities = self.memory.get_priorities().flatten()
            rewards = self.memory.get_rewards().flatten()
            actions = self.memory.get_actions().flatten()
            diag["priorities"] = self._diagnose_array(priorities)
            diag["rewards"] = self._diagnose_array(rewards)
            if actions.size:
                action_counts = np.bincount(actions.astype("int"), minlength=self.num_actions)
                top_ids = np.argsort(action_counts)[-10:][::-1]
                diag["sampled_action_space"] = {
                    "unique_actions": int(np.count_nonzero(action_counts)),
                    "top10_actions": top_ids.astype("int").tolist(),
                    "top10_counts": action_counts[top_ids].astype("int").tolist(),
                }
        except Exception as exc:
            diag["error"] = str(exc)
        return diag

    def _build_activation_probe(self):
        probe_layer_names = [
            "convnext_pixel_input",
            "convnext_image_feature",
            "bird_embedding",
            "image_latent",
            "latent",
            "double_Q_network",
            "default_Q_network",
            "distributional_dueling_Q_network",
            "quantile_dueling_Q_network",
        ]
        outputs = []
        names = []
        for layer_name in probe_layer_names:
            try:
                layer = self.online_learner.get_layer(layer_name)
            except ValueError:
                continue
            outputs.append(layer.output)
            names.append(layer_name)
        if not outputs:
            self.activation_probe_model = None
            self.activation_probe_names = []
            return
        self.activation_probe_model = tf.keras.Model(
            inputs=self.online_learner.inputs,
            outputs=outputs,
            name="diagnostic_activation_probe",
        )
        self.activation_probe_names = names

    def _diagnose_model_activations(self, states_preprocessed):
        if self.activation_probe_model is None and not self.activation_probe_names:
            self._build_activation_probe()
        if self.activation_probe_model is None:
            return {}
        try:
            outputs = self.activation_probe_model.predict(
                states_preprocessed,
                batch_size=max(1, len(states_preprocessed[0])),
                verbose=0,
            )
            if len(self.activation_probe_names) == 1:
                outputs = [outputs]
            return {
                name: self._diagnose_array(out)
                for name, out in zip(self.activation_probe_names, outputs)
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _gradient_group_norm(self, grads, weights, keywords, exclude_keywords=()):
        selected = []
        for grad, weight in zip(grads, weights):
            if grad is None:
                continue
            name = weight.name.lower()
            if any(keyword in name for keyword in keywords) and \
                    not any(keyword in name for keyword in exclude_keywords):
                selected.append(grad)
        if selected:
            return tf.linalg.global_norm(selected)
        return tf.constant(0.0, dtype=tf.float32)

    def _summarize_gradients(self, grads, weights):
        valid_grads = [grad for grad in grads if grad is not None]
        if valid_grads:
            total_norm = tf.linalg.global_norm(valid_grads)
        else:
            total_norm = tf.constant(0.0, dtype=tf.float32)

        convnext = self._gradient_group_norm(grads, weights, ["convnext"], ["convnext_image_feature"])
        classic = self._gradient_group_norm(grads, weights, ["conv_", "image_latent"])
        image_proj = self._gradient_group_norm(grads, weights, ["convnext_image_feature"])
        bird = self._gradient_group_norm(grads, weights, ["bird_embedding"])
        q_network = self._gradient_group_norm(grads, weights, ["double_q_network", "default_q_network"])
        distributional_q_network = self._gradient_group_norm(
            grads,
            weights,
            ["distributional_dueling_q_network", "quantile_dueling_q_network"],
        )

        selected_keywords = [
            "convnext",
            "conv_",
            "image_latent",
            "convnext_image_feature",
            "bird_embedding",
            "double_q_network",
            "default_q_network",
            "distributional_dueling_q_network",
            "quantile_dueling_q_network",
        ]
        other_grads = []
        for grad, weight in zip(grads, weights):
            if grad is None:
                continue
            name = weight.name.lower()
            if not any(keyword in name for keyword in selected_keywords):
                other_grads.append(grad)
        other = tf.linalg.global_norm(other_grads) if other_grads else tf.constant(0.0, dtype=tf.float32)

        return tf.stack([total_norm, convnext, classic, image_proj, bird, q_network, distributional_q_network, other])

    def reinit_env(self, num_par_envs):
        self.num_par_envs = num_par_envs
        new_env = self.env.copy(num_par_envs)
        del self.env
        self.env = new_env

    def get_hidden_states_of(self, model):
        if self.sequential:
            return None  # TODO
        else:
            return None

    def reset_cell_states_for(self, model, ids):
        pass  # TODO

    def practice(self, num_parallel_steps,
                 replay_period,
                 gamma,
                 learning_rate: ParamScheduler = ParamScheduler(0.0001),
                 target_sync_period=None,
                 actor_sync_period=None,
                 replay_size_multiplier=4,
                 replay_epochs=1,
                 epsilon: ParamScheduler = ParamScheduler(0),
                 use_mc_return=False,
                 action_masking=False,
                 alpha=0.7,
                 max_replay_size=None,
                 policy="greedy",
                 min_hist_len=0,
                 memory_size=1000000,
                 n_step=1,
                 sequence_shift=None,
                 eta=0.9,
                 diagnostics_period=100,
                 diagnostics_full_period=500,
                 diagnostics_profile="full",
                 checkpoint_diagnostics=False,
                 convnext_finetune_at_step=None,
                 convnext_finetune_lr_scale=0.1,
                 run_metadata=None,
                 use_tqdm=True,
                 verbose=False):
        """The agent's main training routine.

        :param num_parallel_steps: Number of (parallel) transitions to play
        :param replay_period: Number of parallel steps between each training of the online network
        :param replay_size_multiplier: Factor determining the number of transitions to be learned from each
                   hyperparams cycle (the replay size). Each time, the replay size is determined as
                   follows: replay_size = replay_size_multiplier * new_transitions
        :param replay_epochs: Number of epochs per replay
        :param learning_rate: (dynamic) learning rate used for training
        :param target_sync_period: The number of levels between each synchronization of online and target network. The
                   higher the number, the stronger Double Q-Learning and the less overestimation.
                   sync_period == 1 means "Double Q-Learning off"
        :param actor_sync_period: The number of levels between each synchronization of learner and actor.
        :param gamma: Discount factor
        :param epsilon: Epsilon class, probability for random shot (epsilon greedy policy)
        :param use_mc_return: If True, uses Monte Carlo return targets instead of n-step TD targets
        :param action_masking:
        :param alpha: For Prioritized Experience Replay: the larger alpha the stronger prioritization
        :param max_replay_size:
        :param policy:
        :param min_hist_len:
        :param memory_size:
        :param n_step:
        :param sequence_shift:
        :param eta:
        :param checkpoint_diagnostics: If True, export CSV/PNG/Markdown diagnostics at every checkpoint.
        :param verbose:
        """

        # TODO: handle existing agent

        training_parameters_dict = serialize_function_parameters(fn=self.practice, local_scope=locals())
        self.save_training_parameters(training_parameters_dict)

        # Setup SIGINT handler
        signal.signal(signal.SIGINT, self.end_practice)

        self.check_replay_size(replay_size_multiplier, replay_period, max_replay_size)

        self.set_policy(policy)

        self.policy = policy
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.use_mc_return = use_mc_return
        self.action_masking = action_masking
        self.convnext_gradient_scale.assign(float(convnext_finetune_lr_scale))
        self._update_convnext_finetune_state(
            current_transition=int(self.stats.init_trans_no),
            convnext_finetune_at_step=convnext_finetune_at_step,
        )

        # Setup target learner (target learner == online learner if no sync period provided)
        if target_sync_period is not None:
            self.target_learner = self.copy_online_learner()

        # Setup actor (actor == online learner if no sync period provided)
        if actor_sync_period is not None:
            self.actor = self.copy_online_learner()

        # Init replay memory
        self.memory = ReplayMemory(size=memory_size,
                                   state_shapes=self.state_shapes,
                                   state_dtypes=self.state_dtypes,
                                   n_step=n_step,
                                   stack_size=self.stack_size,
                                   num_par_envs=self.env.num_par_inst,
                                   hidden_state_shapes=self.stem_network.get_hidden_state_shape(),
                                   sequence_len=self.sequence_len,
                                   sequence_shift=sequence_shift,
                                   eta=eta)

        # Save all hyperparameters
        self.write_hyperparams_file(num_parallel_steps, replay_period, replay_size_multiplier,
                                    target_sync_period, actor_sync_period, alpha, gamma, min_hist_len)
        if run_metadata is not None:
            self._write_json_artifact("run_metadata.json", run_metadata)

        new_transitions = 0
        returns = np.zeros(self.num_par_envs)
        self.action_counts[:] = 0
        self.recent_rewards = []

        # Reset all environments
        self.env.reset()

        print("DQN agent starts practicing...")

        self.stats.start_timer()

        if diagnostics_period is not None and diagnostics_period > 0:
            env_diag = self.env.get_diagnostics_config() if hasattr(self.env, "get_diagnostics_config") else {}
            self._write_diagnostics(
                "training_start",
                num_parallel_steps=num_parallel_steps,
                num_parallel_envs=self.num_par_envs,
                state_shapes=[list(shape) for shape in self.state_shapes],
                stack_shapes=[list(shape) for shape in self.stack_shapes],
                num_actions=self.num_actions,
                stem_model_class=self.stem_network.__class__.__name__,
                stem_model_config=self.stem_network.get_config(),
                q_network_class=self.q_network.__class__.__name__,
                q_network_config=self.q_network.get_config(),
                replay_batch_size=self.replay_batch_size,
                replay_period=replay_period,
                replay_size_multiplier=replay_size_multiplier,
                memory_size=memory_size,
                n_step=n_step,
                gamma=gamma,
                alpha=alpha,
                target_sync_period=target_sync_period,
                diagnostics_period=diagnostics_period,
                diagnostics_full_period=diagnostics_full_period,
                diagnostics_profile=diagnostics_profile,
                convnext_finetune_at_step=convnext_finetune_at_step,
                convnext_finetune_lr_scale=convnext_finetune_lr_scale,
                convnext_update_enabled=bool(self.convnext_update_enabled.numpy()),
                auxiliary_heads_config=self.auxiliary_heads_config,
                run_metadata=run_metadata,
                env=env_diag,
            )

        progress_bar = None
        step_iter = range(1, num_parallel_steps + 1)
        if use_tqdm and tqdm is not None:
            progress_bar = tqdm(step_iter, total=num_parallel_steps,
                                desc=self.name, unit="step")
            step_iter = progress_bar

        for i in step_iter:
            if self.stop:
                break

            # Compute one parallel step
            # Computation time (percentage of total) behind each step component
            done_transitions = self.stats.init_trans_no + (i - 1) * self.num_par_envs

            # State preparation (2 %)
            states = self.env.get_states()
            self.stacker.add_states(states)
            hidden_states = self.get_hidden_states_of(self.actor)

            # Predict the next action to take (30 %)
            current_epsilon = self.epsilon.get_value(done_transitions)
            if self.uses_noisynet():
                self.reset_noise_for(self.actor)
            state_stacks = self.stacker.get_stacks()
            actions, _ = self.plan_epsilon_greedy(state_stacks,
                                                  current_epsilon)

            # Perform actions, observe new environment state, level score and application state (13 %)
            rewards, scores, terminals, times, wins, game_overs = self.env.step(actions)
            returns += rewards
            np.add.at(self.action_counts, np.asarray(actions).astype("int"), 1)
            self.recent_rewards.extend(np.asarray(rewards).reshape(-1).astype("float32").tolist())
            if len(self.recent_rewards) > 1000:
                self.recent_rewards = self.recent_rewards[-1000:]
            env_step_diag = self.env.get_last_step_diagnostics() \
                if hasattr(self.env, "get_last_step_diagnostics") else {}

            if diagnostics_period is not None and diagnostics_period > 0:
                if i == 1 or i % diagnostics_period == 0 or np.any(game_overs):
                    do_full_diagnostics = (
                        diagnostics_profile == "full" and
                        diagnostics_full_period is not None and
                        diagnostics_full_period > 0 and
                        (i == 1 or i % diagnostics_full_period == 0 or np.any(game_overs))
                    )
                    model_activations = {}
                    if do_full_diagnostics:
                        model_activations = self._diagnose_model_activations(
                            self.env.preprocess(state_stacks)
                        )
                    action0 = int(np.asarray(actions).reshape(-1)[0])
                    self._write_diagnostics(
                        "training_step",
                        loop_step=i,
                        transition=int(done_transitions),
                        epsilon=float(current_epsilon),
                        learning_rate=float(self.learning_rate.get_value(done_transitions)),
                        memory_transitions=int(self.memory.get_num_transitions()) if self.memory is not None else 0,
                        action=action0,
                        action_text=self.env.actions[action0] if action0 < len(self.env.actions) else "",
                        reward=float(np.asarray(rewards).reshape(-1)[0]),
                        score=int(np.asarray(scores).reshape(-1)[0]),
                        terminal=bool(np.asarray(terminals).reshape(-1)[0]),
                        win=bool(np.asarray(wins).reshape(-1)[0]),
                        game_over=bool(np.asarray(game_overs).reshape(-1)[0]),
                        env_time=int(np.asarray(times).reshape(-1)[0]),
                        episode_return=float(np.asarray(returns).reshape(-1)[0]),
                        recent_rewards=self._diagnose_array(np.asarray(self.recent_rewards, dtype="float32")),
                        states=self._diagnose_states(states),
                        q=self.last_plan_diagnostics,
                        env_step=env_step_diag,
                        action_distribution=self._diagnose_action_distribution(),
                        memory=self._diagnose_memory() if do_full_diagnostics else {},
                        model_activations=model_activations,
                        full_diagnostics=bool(do_full_diagnostics),
                    )

            # Save observations (1 %)
            auxiliary_labels = self._build_auxiliary_labels(scores, rewards, terminals, wins, env_step_diag)
            new_transitions += self.memory.memorize_observations(states, hidden_states, actions,
                                                                 scores, rewards, terminals, gamma,
                                                                 aux_labels=auxiliary_labels)

            # Reset state stacker for envs with true terminals (0 %)
            terminated_env_ids = np.where(terminals)[0]
            self.stacker.reset_stacks(terminated_env_ids)

            # Handle finished envs (0 %)
            if np.any(game_overs):
                fin_env_ids = np.where(game_overs)[0]

                # Save stats
                for idx in fin_env_ids:
                    self.stats.denote_episode_stats(returns[idx], scores[idx], times[idx], wins[idx],
                                                    idx, self.memory)
                    if diagnostics_period is not None and diagnostics_period > 0:
                        current_level = getattr(self.env, "current_level", None)
                        self._write_diagnostics(
                            "episode_end",
                            loop_step=i,
                            transition=int(done_transitions),
                            env_id=int(idx),
                            level=None if current_level is None else int(current_level),
                            episode_return=float(returns[idx]),
                            score=int(scores[idx]),
                            shots=int(times[idx]),
                            win=bool(wins[idx]),
                            memory_transitions=int(self.memory.get_num_transitions()),
                            action_distribution=self._diagnose_action_distribution(),
                        )

                # Reset all finished envs and update their corresponding current return
                self.env.reset_finished()
                returns[fin_env_ids] = 0

                # Reset actor's LSTM states (if any) to zero
                self.reset_cell_states_for(self.actor, fin_env_ids)

            # Every X episodes, plot informative graphs (23 %)
            if i % PLOT_SAVE_STATS_PERIOD == 0:
                self.stats.plot_stats(self.memory, self.model_dir + "plots/")
                self.stats.save(self.model_dir)

            # If environment has test levels, test on it (0 %)
            if i % TEST_PERIOD == 0 and self.env.has_test_levels():
                self.test_on_levels()

            # Training / replay (29 %)
            if i % replay_period == 0:
                self.reset_noise()  # for Noisy Nets (if activated)
                self._update_convnext_finetune_state(
                    current_transition=done_transitions,
                    convnext_finetune_at_step=convnext_finetune_at_step,
                )
                replay_size = replay_size_multiplier * new_transitions
                if self.memory.get_num_transitions() >= min_hist_len and replay_size > 0:
                    if max_replay_size is not None and replay_size > max_replay_size:
                        replay_size = max_replay_size
                    self.update_lr(current_transition=done_transitions)
                    learned_trans = self.learn(replay_size, gamma, epochs=replay_epochs, alpha=alpha, verbose=verbose)
                    if diagnostics_period is not None and diagnostics_period > 0:
                        self._write_diagnostics(
                            "learning_update",
                            loop_step=i,
                            transition=int(done_transitions),
                            replay_size=int(replay_size),
                            learned_transitions=int(learned_trans),
                            memory_transitions=int(self.memory.get_num_transitions()),
                            loss=self._latest_loss(),
                            learning_rate=float(self.learning_rate.get_value(done_transitions)),
                            learning=self.last_learning_diagnostics,
                            gradients=self.last_gradient_diagnostics,
                            memory=self._diagnose_memory(),
                        )
                    new_transitions = max(0, new_transitions - learned_trans)

            # Save model checkpoint (0 %)
            early_checkpoint = (
                EARLY_CHECKPOINT_SAVE_PERIOD is not None
                and i <= EARLY_CHECKPOINT_SAVE_UNTIL
                and i % EARLY_CHECKPOINT_SAVE_PERIOD == 0
            )
            regular_checkpoint = i % CHECKPOINT_SAVE_PERIOD == 0
            if early_checkpoint or regular_checkpoint:
                self.save_weights(overwrite=True, checkpoint_no=done_transitions)
                self.stats.save(self.model_dir)
                if diagnostics_period is not None and diagnostics_period > 0:
                    self._write_diagnostics(
                        "checkpoint_saved",
                        loop_step=i,
                        transition=int(done_transitions),
                        checkpoint_no=int(done_transitions),
                    )
                if checkpoint_diagnostics and diagnostics_period is not None and diagnostics_period > 0:
                    self._export_checkpoint_diagnostics_report(
                        f"checkpoint_{int(done_transitions):08d}"
                    )

            # Cut off old experience to reduce buffer load (0 %)
            if self.memory.get_num_transitions() > 0.95 * self.memory.get_size():
                self.memory.delete_first(n=int(0.2 * self.memory.get_size()))

            # Synchronize target and online network every sync_period levels (Double Q-Learning) (0 %)
            if target_sync_period is not None and i % target_sync_period == 0:
                self.target_learner.set_weights(self.online_learner.get_weights())

            # Synchronize learner and actor (Distributed RL) (0 %)
            if actor_sync_period is not None and i % actor_sync_period == 0:
                self.actor.set_weights(self.online_learner.get_weights())

            # Print a summary of current learning statistics (0 %)
            if i % PRINT_STATS_PERIOD == 0:
                self.stats.print_stats(i, num_parallel_steps, PRINT_STATS_PERIOD, self.num_par_envs,
                                       self.epsilon.get_value(done_transitions), self.num_par_envs)
                if progress_bar is not None:
                    progress_bar.set_postfix({
                        "eps": f"{float(current_epsilon):.3f}",
                        "loss": f"{self._latest_loss():.4g}",
                        "mem": int(self.memory.get_num_transitions()),
                    })

            # if i % 10000 == 0:
            #     print_total_ram_usage()
            #     # print_pympler_ram_usage()
            #     # print_leaking_objects_count()
            #     print_memory_block_summary()
            #     # print_heap()

        if progress_bar is not None:
            progress_bar.close()

        if diagnostics_period is not None and diagnostics_period > 0:
            self._write_diagnostics(
                "training_stop",
                stopped_by_signal=bool(self.stop),
                memory_transitions=int(self.memory.get_num_transitions()) if self.memory is not None else 0,
                latest_loss=self._latest_loss(),
                action_distribution=self._diagnose_action_distribution(),
                memory=self._diagnose_memory(),
            )
            if checkpoint_diagnostics:
                self._export_checkpoint_diagnostics_report("final")

        print("Concluding practice...")
        self.save_weights(overwrite=True)
        self.stats.save(self.model_dir)
        self.stats.logger.close()
        self.export_saved_model()
        print_success("Practicing finished successfully!")

    def end_practice(self, sig, frame):
        print_info("Stopping agent practice...")
        self.stop = True

    def learn(self, replay_size, gamma, epochs=1, alpha=0.7, verbose=False):
        """Updates the online network's weights. This is the actual learning procedure of the agent.

        :param replay_size: Number of transitions to be learned from
        :param gamma: Discount factor
        :param epochs:
        :param alpha: For Prioritized Experience Replay: the larger alpha the stronger prioritization
        :param verbose:
        :return: number of transitions learned
        """

        if replay_size == 0:
            return 0

        if not self.sequential:
            return self.learn_instances(replay_size, gamma=gamma, alpha=alpha,
                                        epochs=epochs, verbose=verbose)
        else:
            num_sequences = replay_size // self.sequence_len
            if num_sequences >= self.replay_batch_size:
                return self.learn_sequences(num_sequences, gamma=gamma, alpha=alpha,
                                            epochs=epochs, verbose=verbose)
            else:
                return 0

    def learn_instances(self, num_instances, gamma, epochs=1, alpha=0.7, verbose=False):
        """Uses batches of single instances to learn on."""

        if self.is_quantile_distributional():
            return self.learn_quantile_instances(num_instances, gamma=gamma, epochs=epochs,
                                                 alpha=alpha, verbose=verbose)
        if self.is_c51_distributional():
            return self.learn_distributional_instances(num_instances, gamma=gamma, epochs=epochs,
                                                       alpha=alpha, verbose=verbose)

        # Obtain a list of useful transitions to learn on
        trans_ids, probabilities = self.memory.recall(num_instances, alpha)
        if len(trans_ids) == 0:
            self.last_learning_diagnostics = {
                "mode": "instances",
                "sample_count": 0,
                "reason": "replay recall returned no transitions",
            }
            return 0

        # Obtain total number of experienced transitions
        exp_len = self.memory.get_num_transitions()

        # Get transitions data and preprocess
        states, _, actions, n_step_rewards, step_mask, next_states, _ = \
            self.memory.get_transitions(trans_ids)
        states_prep = self.env.preprocess(states)
        next_states_prep = self.env.preprocess(next_states)

        # Predict returns (i.e. values V(s)) for all states s
        q_vals = self._predict_q(self.online_learner, states_prep, batch_size=self.replay_batch_size, verbose=verbose)
        pred_returns = np.max(q_vals, axis=1)

        # Predict next returns
        next_q_vals = self._predict_q(self.target_learner, next_states_prep,
                                      batch_size=self.replay_batch_size, verbose=verbose)
        pred_next_returns = np.max(next_q_vals, axis=1)

        # Compute (n-step or MC) return targets and temporal-difference (TD) errors (the "surprise" of the agent)
        if self.use_mc_return:
            # Obtain Monte Carlo return for each transition
            target_returns = self.memory.get_mc_returns(trans_ids)
        else:
            target_returns = get_n_step_return(pred_next_returns, n_step_rewards, step_mask, gamma)
        td_errs = target_returns - pred_returns

        # Update transition priorities according to TD errors
        self.memory.set_priorities(trans_ids, np.abs(td_errs))

        # Prepare inputs and targets for fitting
        inputs = states_prep
        targets = self._predict_q(self.target_learner, states_prep, batch_size=self.replay_batch_size,
                                  verbose=verbose)
        targets[range(len(trans_ids)), actions] = target_returns

        # Prepare action mask for masking away unconsidered action
        if self.action_masking:
            action_mask = np.zeros(targets.shape)
            action_mask[range(len(trans_ids)), actions] = 1
        else:
            action_mask = None

        # Compute sample weights
        sample_weights = compute_sample_weights(td_errs, probabilities[trans_ids], exp_len)
        aux_targets = self._get_auxiliary_targets(trans_ids)
        self.last_learning_diagnostics = {
            "mode": "instances",
            "requested_instances": int(num_instances),
            "sample_count": int(len(trans_ids)),
            "transition_id_sample": np.asarray(trans_ids[:10]).astype("int").tolist(),
            "actions_sample": np.asarray(actions[:10]).astype("int").tolist(),
            "q_values": self._diagnose_array(q_vals),
            "next_q_values": self._diagnose_array(next_q_vals),
            "pred_returns": self._diagnose_array(pred_returns),
            "pred_next_returns": self._diagnose_array(pred_next_returns),
            "target_returns": self._diagnose_array(target_returns),
            "td_errors": self._diagnose_array(td_errs),
            "abs_td_errors": self._diagnose_array(np.abs(td_errs)),
            "sample_weights": self._diagnose_array(sample_weights),
            "sample_probabilities": self._diagnose_array(probabilities[trans_ids]),
            "n_step_rewards": self._diagnose_array(n_step_rewards),
            "step_mask_true_fraction": float(np.mean(step_mask.astype("float32"))),
            "auxiliary_heads": self.auxiliary_heads_config if self.uses_auxiliary_heads() else None,
            "auxiliary_targets": self._diagnose_array(aux_targets) if aux_targets is not None else {},
        }

        # Update the online network's weights
        loss, individual_losses, predictions = self.fit(inputs, targets, epochs=epochs, verbose=verbose,
                                                        batch_size=self.replay_batch_size,
                                                        sample_weights=sample_weights,
                                                        action_mask=action_mask,
                                                        aux_targets=aux_targets)
        self.last_learning_diagnostics.update({
            "loss": float(loss),
            "individual_losses": self._diagnose_array(individual_losses),
            "predictions_after_fit": self._diagnose_array(predictions),
            "targets": self._diagnose_array(targets),
        })

        self.stats.denote_learning_stats(loss, self.optimizer.learning_rate.numpy())
        self.stats.log_extreme_losses(individual_losses, trans_ids, states, predictions, targets, n_step_rewards,
                                      step_mask, self.env, self.memory, self.model_dir)

        return len(trans_ids)

    def project_distributional_targets(self, n_step_rewards, step_mask, next_action_distributions, gamma):
        """Projects n-step Bellman targets onto the fixed C51 support."""
        support = self.get_distributional_support_np()
        num_atoms = len(support)
        v_min = float(support[0])
        v_max = float(support[-1])
        delta_z = (v_max - v_min) / float(num_atoms - 1)

        rewards = np.asarray(n_step_rewards, dtype="float32")
        mask = np.asarray(step_mask, dtype="bool")
        next_dist = np.asarray(next_action_distributions, dtype="float32")
        batch_size = rewards.shape[0]
        n = rewards.shape[-1]

        discounts = (gamma ** np.arange(n)).astype("float32")
        reward_mask = mask[:, :n].astype("float32")
        discounted_rewards = rewards * discounts[None, :] * reward_mask
        reward_returns = np.sum(discounted_rewards, axis=-1)
        bootstrap_mask = mask[:, n].astype("float32")

        projected_atoms = reward_returns[:, None] + bootstrap_mask[:, None] * (gamma ** n) * support[None, :]
        projected_atoms = np.clip(projected_atoms, v_min, v_max)
        b = (projected_atoms - v_min) / delta_z
        lower = np.floor(b).astype("int32")
        upper = np.ceil(b).astype("int32")

        projected_dist = np.zeros((batch_size, num_atoms), dtype="float32")
        for batch_id in range(batch_size):
            for atom_id in range(num_atoms):
                prob = next_dist[batch_id, atom_id]
                lo = lower[batch_id, atom_id]
                hi = upper[batch_id, atom_id]
                if lo == hi:
                    projected_dist[batch_id, lo] += prob
                else:
                    projected_dist[batch_id, lo] += prob * (hi - b[batch_id, atom_id])
                    projected_dist[batch_id, hi] += prob * (b[batch_id, atom_id] - lo)

        projected_dist_sum = np.sum(projected_dist, axis=-1, keepdims=True)
        projected_dist_sum = np.maximum(projected_dist_sum, 1e-8)
        return projected_dist / projected_dist_sum

    def project_quantile_targets(self, n_step_rewards, step_mask, next_action_quantiles, gamma):
        """Builds n-step QR-DQN targets without projecting onto a fixed support."""
        rewards = np.asarray(n_step_rewards, dtype="float32")
        mask = np.asarray(step_mask, dtype="bool")
        next_quantiles = np.asarray(next_action_quantiles, dtype="float32")
        n = rewards.shape[-1]

        discounts = (gamma ** np.arange(n)).astype("float32")
        reward_mask = mask[:, :n].astype("float32")
        discounted_rewards = rewards * discounts[None, :] * reward_mask
        reward_returns = np.sum(discounted_rewards, axis=-1)
        bootstrap_mask = mask[:, n].astype("float32")

        return reward_returns[:, None] + bootstrap_mask[:, None] * (gamma ** n) * next_quantiles

    def learn_quantile_instances(self, num_instances, gamma, epochs=1, alpha=0.7, verbose=False):
        """QR-DQN + true Double DQN update for the Model F Rainbow preset."""

        if self.sequential:
            raise NotImplementedError("QR-Rainbow is wired for non-sequential Angry Birds states only.")

        trans_ids, probabilities = self.memory.recall(num_instances, alpha)
        if len(trans_ids) == 0:
            self.last_learning_diagnostics = {
                "mode": "quantile_instances",
                "sample_count": 0,
                "reason": "replay recall returned no transitions",
            }
            return 0

        exp_len = self.memory.get_num_transitions()
        states, _, actions, n_step_rewards, step_mask, next_states, _ = \
            self.memory.get_transitions(trans_ids)
        states_prep = self.env.preprocess(states)
        next_states_prep = self.env.preprocess(next_states)

        current_quantiles = self._predict_q(self.online_learner, states_prep,
                                            batch_size=self.replay_batch_size, verbose=verbose)
        current_q_vals = self.distribution_to_q_values(current_quantiles)
        batch_ids = np.arange(len(trans_ids))
        pred_returns = current_q_vals[batch_ids, actions]

        next_online_quantiles = self._predict_q(self.online_learner, next_states_prep,
                                                batch_size=self.replay_batch_size, verbose=verbose)
        next_online_q_vals = self.distribution_to_q_values(next_online_quantiles)
        best_next_actions = np.argmax(next_online_q_vals, axis=1)

        next_target_quantiles = self._predict_q(self.target_learner, next_states_prep,
                                                batch_size=self.replay_batch_size, verbose=verbose)
        next_selected_quantiles = next_target_quantiles[batch_ids, best_next_actions]
        target_action_quantiles = self.project_quantile_targets(
            n_step_rewards=n_step_rewards,
            step_mask=step_mask,
            next_action_quantiles=next_selected_quantiles,
            gamma=gamma,
        )

        target_returns = np.mean(target_action_quantiles, axis=1)
        td_errs = target_returns - pred_returns

        priorities = np.abs(td_errs) + 1e-6
        self.memory.set_priorities(trans_ids, priorities)

        inputs = states_prep
        targets = np.asarray(current_quantiles, dtype="float32").copy()
        targets[batch_ids, actions] = target_action_quantiles

        action_mask = np.zeros(targets.shape[:2], dtype="float32")
        action_mask[batch_ids, actions] = 1.0

        sample_weights = compute_sample_weights(priorities, probabilities[trans_ids], exp_len)
        aux_targets = self._get_auxiliary_targets(trans_ids)
        self.last_learning_diagnostics = {
            "mode": "quantile_instances",
            "requested_instances": int(num_instances),
            "sample_count": int(len(trans_ids)),
            "transition_id_sample": np.asarray(trans_ids[:10]).astype("int").tolist(),
            "actions_sample": np.asarray(actions[:10]).astype("int").tolist(),
            "double_dqn_selected_next_actions": np.asarray(best_next_actions[:10]).astype("int").tolist(),
            "current_quantiles": self._diagnose_quantile_output(current_quantiles),
            "next_online_quantiles": self._diagnose_quantile_output(next_online_quantiles),
            "next_target_quantiles": self._diagnose_quantile_output(next_target_quantiles),
            "target_action_quantiles": self._diagnose_quantile_output(target_action_quantiles),
            "q_values": self._diagnose_array(current_q_vals),
            "next_online_q_values": self._diagnose_array(next_online_q_vals),
            "pred_returns": self._diagnose_array(pred_returns),
            "target_returns": self._diagnose_array(target_returns),
            "td_errors": self._diagnose_array(td_errs),
            "abs_td_errors": self._diagnose_array(np.abs(td_errs)),
            "sample_weights": self._diagnose_array(sample_weights),
            "sample_probabilities": self._diagnose_array(probabilities[trans_ids]),
            "n_step_rewards": self._diagnose_array(n_step_rewards),
            "step_mask_true_fraction": float(np.mean(step_mask.astype("float32"))),
            "n_step": int(n_step_rewards.shape[-1]),
            "num_quantiles": int(current_quantiles.shape[-1]),
            "quantile_loss": "selected_action_pairwise_huber",
            "auxiliary_heads": self.auxiliary_heads_config if self.uses_auxiliary_heads() else None,
            "auxiliary_targets": self._diagnose_array(aux_targets) if aux_targets is not None else {},
        }

        loss, individual_losses, predictions = self.fit(inputs, targets, epochs=epochs, verbose=verbose,
                                                        batch_size=self.replay_batch_size,
                                                        sample_weights=sample_weights,
                                                        action_mask=action_mask,
                                                        aux_targets=aux_targets)
        self.last_learning_diagnostics.update({
            "loss": float(loss),
            "individual_losses": self._diagnose_array(individual_losses),
            "selected_action_losses": self._diagnose_array(individual_losses[batch_ids, actions]),
            "predictions_after_fit": self._diagnose_quantile_output(predictions),
            "targets": self._diagnose_quantile_output(targets),
        })

        self.stats.denote_learning_stats(loss, self.optimizer.learning_rate.numpy())
        selected_action_losses = individual_losses[batch_ids, actions]
        self.stats.log_extreme_losses(selected_action_losses, trans_ids, states,
                                      self.distribution_to_q_values(predictions),
                                      self.distribution_to_q_values(targets),
                                      n_step_rewards, step_mask, self.env, self.memory, self.model_dir)

        return len(trans_ids)

    def learn_distributional_instances(self, num_instances, gamma, epochs=1, alpha=0.7, verbose=False):
        """C51 + true Double DQN update for the full Rainbow preset."""

        if self.sequential:
            raise NotImplementedError("Distributional Rainbow is wired for non-sequential Angry Birds states only.")

        trans_ids, probabilities = self.memory.recall(num_instances, alpha)
        if len(trans_ids) == 0:
            self.last_learning_diagnostics = {
                "mode": "distributional_instances",
                "sample_count": 0,
                "reason": "replay recall returned no transitions",
            }
            return 0

        exp_len = self.memory.get_num_transitions()
        states, _, actions, n_step_rewards, step_mask, next_states, _ = \
            self.memory.get_transitions(trans_ids)
        states_prep = self.env.preprocess(states)
        next_states_prep = self.env.preprocess(next_states)

        current_dists = self._predict_q(self.online_learner, states_prep,
                                        batch_size=self.replay_batch_size, verbose=verbose)
        current_q_vals = self.distribution_to_q_values(current_dists)
        batch_ids = np.arange(len(trans_ids))
        pred_returns = current_q_vals[batch_ids, actions]

        next_online_dists = self._predict_q(self.online_learner, next_states_prep,
                                            batch_size=self.replay_batch_size, verbose=verbose)
        next_online_q_vals = self.distribution_to_q_values(next_online_dists)
        best_next_actions = np.argmax(next_online_q_vals, axis=1)

        next_target_dists = self._predict_q(self.target_learner, next_states_prep,
                                            batch_size=self.replay_batch_size, verbose=verbose)
        next_selected_dists = next_target_dists[batch_ids, best_next_actions]
        target_action_dists = self.project_distributional_targets(
            n_step_rewards=n_step_rewards,
            step_mask=step_mask,
            next_action_distributions=next_selected_dists,
            gamma=gamma,
        )

        support = self.get_distributional_support_np()
        target_returns = np.sum(target_action_dists * support[None, :], axis=1)
        td_errs = target_returns - pred_returns

        self.memory.set_priorities(trans_ids, np.abs(td_errs) + 1e-6)

        inputs = states_prep
        targets = np.asarray(current_dists, dtype="float32").copy()
        targets[batch_ids, actions] = target_action_dists

        action_mask = np.zeros(targets.shape[:2], dtype="float32")
        action_mask[batch_ids, actions] = 1.0

        sample_weights = compute_sample_weights(np.abs(td_errs) + 1e-6, probabilities[trans_ids], exp_len)
        aux_targets = self._get_auxiliary_targets(trans_ids)
        self.last_learning_diagnostics = {
            "mode": "distributional_instances",
            "requested_instances": int(num_instances),
            "sample_count": int(len(trans_ids)),
            "transition_id_sample": np.asarray(trans_ids[:10]).astype("int").tolist(),
            "actions_sample": np.asarray(actions[:10]).astype("int").tolist(),
            "double_dqn_selected_next_actions": np.asarray(best_next_actions[:10]).astype("int").tolist(),
            "support": self._diagnose_array(support),
            "current_distributions": self._diagnose_distributional_output(current_dists),
            "next_online_distributions": self._diagnose_distributional_output(next_online_dists),
            "next_target_distributions": self._diagnose_distributional_output(next_target_dists),
            "projected_target_distributions": self._diagnose_distributional_output(target_action_dists),
            "q_values": self._diagnose_array(current_q_vals),
            "next_online_q_values": self._diagnose_array(next_online_q_vals),
            "pred_returns": self._diagnose_array(pred_returns),
            "target_returns": self._diagnose_array(target_returns),
            "td_errors": self._diagnose_array(td_errs),
            "abs_td_errors": self._diagnose_array(np.abs(td_errs)),
            "sample_weights": self._diagnose_array(sample_weights),
            "sample_probabilities": self._diagnose_array(probabilities[trans_ids]),
            "n_step_rewards": self._diagnose_array(n_step_rewards),
            "step_mask_true_fraction": float(np.mean(step_mask.astype("float32"))),
            "n_step": int(n_step_rewards.shape[-1]),
            "c51_loss": "cross_entropy_on_selected_action_distribution",
            "auxiliary_heads": self.auxiliary_heads_config if self.uses_auxiliary_heads() else None,
            "auxiliary_targets": self._diagnose_array(aux_targets) if aux_targets is not None else {},
        }

        loss, individual_losses, predictions = self.fit(inputs, targets, epochs=epochs, verbose=verbose,
                                                        batch_size=self.replay_batch_size,
                                                        sample_weights=sample_weights,
                                                        action_mask=action_mask,
                                                        aux_targets=aux_targets)
        self.last_learning_diagnostics.update({
            "loss": float(loss),
            "individual_losses": self._diagnose_array(individual_losses),
            "selected_action_losses": self._diagnose_array(individual_losses[batch_ids, actions]),
            "predictions_after_fit": self._diagnose_distributional_output(predictions),
            "targets": self._diagnose_distributional_output(targets),
        })

        self.stats.denote_learning_stats(loss, self.optimizer.learning_rate.numpy())
        selected_action_losses = individual_losses[batch_ids, actions]
        self.stats.log_extreme_losses(selected_action_losses, trans_ids, states,
                                      self.distribution_to_q_values(predictions),
                                      self.distribution_to_q_values(targets),
                                      n_step_rewards, step_mask, self.env, self.memory, self.model_dir)

        return len(trans_ids)

    def learn_sequences(self, num_sequences, gamma, epochs=1, alpha=0.7, verbose=False):
        """Uses batches of sequences to learn on."""

        # Obtain a list of useful sequences to learn on
        seq_ids, probabilities = self.memory.recall_sequences(num_sequences, alpha, batch_size=self.replay_batch_size)

        if len(seq_ids) == 0:
            self.last_learning_diagnostics = {
                "mode": "sequences",
                "sample_count": 0,
                "reason": "replay recall returned no sequences",
            }
            return 0

        seq_num = self.memory.get_num_sequences()

        # Get sequences of transitions
        trans_ids, (states, first_hidden_states, actions, rewards, next_states, last_hidden_states, terminals), mask \
            = self.memory.get_sequences(seq_ids)

        # Predict returns (i.e. values V(s)) for all states s
        q_vals = self.online_learner.set_hidden_and_predict(first_hidden_states, states)
        pred_returns = np.max(q_vals, axis=2)

        # Predict next returns
        next_states_2d, next_states_1d = next_states
        last_states = [next_states_2d[:, np.newaxis, -1], next_states_1d[:, np.newaxis, -1]]
        next_q_vals = self.target_learner.set_hidden_and_predict(last_hidden_states, last_states)
        pred_last_returns = np.max(next_q_vals, axis=2).squeeze(axis=1)

        # Backward target return construction
        target_returns = np.zeros(shape=(len(seq_ids), self.sequence_len))
        target_returns[:, -1] = rewards[:, -1] + gamma * pred_last_returns
        target_returns[:, -1][terminals[:, -1]] = rewards[:, -1][terminals[:, -1]]
        target_returns[:, -1][~ mask[:, -1]] = 0
        for time_step in reversed(range(self.sequence_len - 1)):
            target_returns[:, time_step] = rewards[:, time_step] + gamma * target_returns[:, time_step + 1]
            target_returns[:, time_step][terminals[:, time_step]] = rewards[:, time_step][terminals[:, time_step]]
            target_returns[:, time_step][~ mask[:, time_step]] = 0

        # Temporal difference (TD) error
        td_errs = target_returns - pred_returns
        assert not np.any(np.isnan(td_errs))

        # Update transition priorities according to TD errors
        self.memory.set_priorities(trans_ids, np.abs(td_errs))
        seq_prios = self.memory.update_seq_priorities(seq_ids=seq_ids, trans_ids=trans_ids, mask=mask)

        # Prepare inputs and targets for fitting
        inputs = states
        targets = self.target_learner.set_hidden_and_predict(first_hidden_states, states)
        ids_i, ids_s = np.mgrid[0:len(seq_ids), 0:self.sequence_len]
        targets[ids_i, ids_s, actions] = target_returns

        # Compute sample weights
        sample_weights = compute_sample_weights(seq_prios, probabilities[seq_ids], seq_num)
        self.last_learning_diagnostics = {
            "mode": "sequences",
            "requested_sequences": int(num_sequences),
            "sample_count": int(len(seq_ids)),
            "sequence_id_sample": np.asarray(seq_ids[:10]).astype("int").tolist(),
            "q_values": self._diagnose_array(q_vals),
            "pred_returns": self._diagnose_array(pred_returns),
            "target_returns": self._diagnose_array(target_returns),
            "td_errors": self._diagnose_array(td_errs),
            "sample_weights": self._diagnose_array(sample_weights),
            "sequence_priorities": self._diagnose_array(seq_prios),
        }

        # Update the online network's weights
        loss, individual_losses, predictions = self.fit(inputs, targets, epochs=epochs, verbose=verbose,
                                                        batch_size=self.replay_batch_size,
                                                        sample_weights=sample_weights,
                                                        hidden_states=first_hidden_states,
                                                        seq_mask=mask)
        self.last_learning_diagnostics.update({
            "loss": float(loss),
            "individual_losses": self._diagnose_array(individual_losses),
            "predictions_after_fit": self._diagnose_array(predictions),
            "targets": self._diagnose_array(targets),
        })

        self.stats.denote_learning_stats(loss, self.optimizer.learning_rate.numpy())

        return np.prod(trans_ids.shape)

    def fit(self, x, y, epochs, batch_size, sample_weights, action_mask=None, hidden_states=None,
            seq_mask=None, aux_targets=None, verbose=False):
        assert not self.sequential or (hidden_states is not None and seq_mask is not None)
        start = time.time()

        # Prepare the training dataset
        # None을 더미 텐서로 대체 — @tf.function에 Python 객체가 전달되면
        # 매 호출마다 retrace가 발생하므로 shape이 맞는 numpy 배열을 사용한다.
        if action_mask is None:
            action_mask = np.zeros_like(y, dtype="float32")  # [N, num_actions]
        if aux_targets is None:
            aux_targets = np.zeros((len(y), max(1, self.auxiliary_output_dim)), dtype="float32")
        if not self.sequential:
            hidden_states = np.zeros((len(y), 1), dtype="float32")  # [N, 1]
            seq_mask = np.ones(len(y), dtype="float32")              # [N]
        train_dataset = (*x, y, aux_targets, action_mask, sample_weights, hidden_states, seq_mask)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).batch(batch_size)
        # train_dataset = train_dataset.shuffle(buffer_size=1024, seed=self.seed)
        predictions = np.zeros(y.shape)
        individual_losses = np.zeros(y.shape[:-1])
        train_loss = 0
        gradient_stats_sum = np.zeros(len(GRADIENT_DIAGNOSTIC_NAMES), dtype="float64")
        gradient_stats_count = 0

        for epoch in range(epochs):
            if verbose:
                print("\rEpoch %d/%d - Batch 0/%d - Loss: --" %
                      ((epoch + 1), epochs, len(train_dataset)), flush=True, end="")

            # Iterate over the batches of the dataset.
            for step, (*x_b, y_b, aux_targets_b, act_mask_b, sample_weights_b, hidden_b, seq_mask_b) in enumerate(train_dataset):
                # Train on this batch
                if self.sequential:
                    self.online_learner.reset_states()
                    self.online_learner.stem_model.set_cell_states(hidden_b)

                batch_individual_losses, batch_out, batch_gradient_stats, batch_total_loss = \
                    self.train_step(x_b, y_b, aux_targets_b, act_mask_b, sample_weights_b, seq_mask_b)
                gradient_stats_sum += batch_gradient_stats.numpy()
                gradient_stats_count += 1

                if epoch == epochs - 1:
                    at_instance = step * batch_size
                    predictions[at_instance:at_instance + len(y_b)] = batch_out
                    individual_losses[at_instance:at_instance + len(y_b)] = batch_individual_losses

                train_loss = float(batch_total_loss.numpy())

                if verbose:
                    print("\rEpoch %d/%d - Batch %d/%d - Total loss: %.4f" %
                          ((epoch + 1), epochs, (step + 1), len(train_dataset), float(train_loss)),
                          flush=True, end="")

            if hasattr(self.training_loss_metric, "reset_state"):
                self.training_loss_metric.reset_state()
            else:
                self.training_loss_metric.reset_states()
            if verbose:
                print("")

        if gradient_stats_count > 0:
            gradient_stats_avg = gradient_stats_sum / gradient_stats_count
            self.last_gradient_diagnostics = {
                name: float(value)
                for name, value in zip(GRADIENT_DIAGNOSTIC_NAMES, gradient_stats_avg)
            }
        else:
            self.last_gradient_diagnostics = {}

        if verbose:
            print("Fitting took %.2f s." % (time.time() - start))

        return train_loss, individual_losses, predictions

    @tf.function(reduce_retracing=True)
    def train_step(self, x, y, aux_targets, act_mask, sample_weight, seq_mask):
        with tf.GradientTape() as tape:
            model_out = self.online_learner(x, training=True)
            aux_out = None
            if isinstance(model_out, (list, tuple)):
                out = model_out[0]
                aux_out = model_out[1] if len(model_out) > 1 else None
            else:
                out = model_out
            if y.shape.rank == 3:
                if self.is_quantile_distributional():
                    # QR-DQN branch: quantile Huber loss on the selected action.
                    pred_quantiles = tf.expand_dims(out, axis=-1)
                    target_quantiles = tf.expand_dims(y, axis=-2)
                    td_errors = target_quantiles - pred_quantiles
                    abs_td_errors = tf.abs(td_errors)
                    huber_delta = tf.constant(1.0, dtype=tf.float32)
                    huber_losses = tf.where(
                        abs_td_errors <= huber_delta,
                        0.5 * tf.square(td_errors),
                        huber_delta * (abs_td_errors - 0.5 * huber_delta),
                    )
                    num_quantiles = tf.shape(out)[-1]
                    taus = (
                        tf.cast(tf.range(num_quantiles), tf.float32) + 0.5
                    ) / tf.cast(num_quantiles, tf.float32)
                    taus = tf.reshape(taus, [1, 1, -1, 1])
                    quantile_weights = tf.abs(
                        taus - tf.cast(td_errors < 0.0, tf.float32)
                    )
                    per_action_losses = tf.reduce_mean(
                        quantile_weights * huber_losses,
                        axis=[-2, -1],
                    )
                else:
                    # Distributional C51 branch: cross entropy on the selected action's atom distribution.
                    out_clipped = tf.clip_by_value(out, 1e-6, 1.0)
                    per_action_losses = -tf.reduce_sum(y * tf.math.log(out_clipped), axis=-1)
                use_mask = tf.reduce_any(tf.not_equal(act_mask, 0))
                act_mask_f = tf.cast(act_mask, tf.float32)
                if act_mask.shape.rank == 3:
                    act_mask_f = tf.reduce_max(act_mask_f, axis=-1)
                masked_losses = tf.multiply(per_action_losses, act_mask_f)
                masked_denominator = tf.maximum(tf.reduce_sum(act_mask_f, axis=-1), 1.0)
                selected_losses = tf.reduce_sum(masked_losses, axis=-1) / masked_denominator
                unmasked_losses = tf.reduce_mean(per_action_losses, axis=-1)
                weighted_losses = tf.where(use_mask, selected_losses, unmasked_losses)
                weighted_losses = tf.multiply(weighted_losses, sample_weight)
                q_loss = tf.reduce_mean(weighted_losses)
            else:
                # act_mask: action_masking=True이면 실제 마스크, False이면 모두 0인 더미 텐서
                # tf.reduce_any로 마스크 유무를 그래프 안에서 판단 (Python 분기 제거)
                use_mask = tf.reduce_any(tf.not_equal(act_mask, 0))
                act_mask_f = tf.cast(act_mask, tf.float32)
                y_masked   = tf.where(use_mask, tf.multiply(y,   act_mask_f), y)
                out_masked = tf.where(use_mask, tf.multiply(out, act_mask_f), out)
                weighted_losses = self.training_loss_fn(y_masked, out_masked, sample_weight=sample_weight)
                if self.sequential:
                    seq_mask = tf.cast(seq_mask, tf.float32)
                    weighted_losses = tf.multiply(weighted_losses, seq_mask)
                q_loss = tf.reduce_mean(weighted_losses)

            if self.uses_auxiliary_heads() and aux_out is not None:
                aux_loss = self._compute_auxiliary_loss(aux_targets, aux_out)
                cumulated_loss = q_loss + self.auxiliary_loss_weight * aux_loss
            else:
                cumulated_loss = q_loss

        trainable_weights = self.online_learner.trainable_weights
        grads = tape.gradient(cumulated_loss, trainable_weights)
        grads = self._gate_convnext_gradients(grads, trainable_weights)
        gradient_stats = self._summarize_gradients(grads, trainable_weights)

        # Compute unweighted losses
        if y.shape.rank == 3:
            losses = per_action_losses
        else:
            losses = self.training_loss_fn(y_masked, out_masked)
            if self.sequential:
                losses = tf.multiply(losses, seq_mask)

        # Run one step of gradient descent by the optimizer
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        if y.shape.rank == 3:
            self.training_loss_metric.update_state(tf.zeros_like(weighted_losses), weighted_losses)
        else:
            self.training_loss_metric.update_state(y, out)

        return losses, out, gradient_stats, cumulated_loss

    def plan_epsilon_greedy(self, states, epsilon, compute_return=False):
        """Epsilon greedy policy. With a probability of epsilon, a random action is returned. Else,
        the agent predicts the best shot for the given state.

        :param states: List of state matrices
        :param epsilon: If given, epsilon greedy policy will be applied, otherwise the agent plans optimally
        :param compute_return: Computes the return even if the action is a random one.
        :return: action: An index number, corresponding to an action
        """

        if np.random.random(1) < epsilon:
            # Random action
            actions = np.random.randint(self.num_actions, size=self.num_par_envs)
            if self.sequential or compute_return:  # Update hidden state
                _, pred_rets = self.plan(states)
                self.last_plan_diagnostics["policy_decision"] = "random_with_q_diagnostics"
            else:
                pred_rets = [np.nan]
                self.last_plan_diagnostics = {
                    "policy_decision": "random",
                    "epsilon": float(epsilon),
                }
            return actions, pred_rets
        else:
            # Optimal action
            return self.plan(states)

    def plan(self, states):
        # t = time.time()

        if self.sequential:
            states = [np.expand_dims(state_comp, axis=1) for state_comp in states]

        batch_size = self.num_par_envs if self.sequential else self.replay_batch_size

        # timer.add_time("Planning preparation", time.time() - t)
        # t = time.time()

        states_preprocessed = self.env.preprocess(states)

        # timer.add_time("State preprocessing", time.time() - t)
        # t = time.time()

        raw_model_output = self._predict_q(self.actor,
                                           states_preprocessed,
                                           batch_size=batch_size,
                                           verbose=0)
        if self.is_distributional():
            q_vals = self.distribution_to_q_values(raw_model_output)
        else:
            q_vals = raw_model_output
        if self.sequential:
            q_vals = np.squeeze(q_vals, axis=1)

        q_first = q_vals[0]
        top_ids = np.argsort(q_first)[-5:][::-1]
        self.last_plan_diagnostics = {
            "policy_decision": "model",
            "shape": list(q_vals.shape),
            "raw_model_output_shape": list(np.asarray(raw_model_output).shape),
            "distributional": bool(self.is_distributional()),
            "min": float(np.min(q_vals)),
            "max": float(np.max(q_vals)),
            "mean": float(np.mean(q_vals)),
            "std": float(np.std(q_vals)),
            "best_actions": q_vals.argmax(axis=1).astype("int").tolist(),
            "top5_actions_first_env": top_ids.astype("int").tolist(),
            "top5_values_first_env": q_first[top_ids].astype("float32").tolist(),
            "nan_count": int(np.isnan(q_vals).sum()),
            "inf_count": int(np.isinf(q_vals).sum()),
        }
        if self.is_c51_distributional():
            self.last_plan_diagnostics["c51"] = self._diagnose_distributional_output(raw_model_output)
        elif self.is_quantile_distributional():
            self.last_plan_diagnostics["qr_dqn"] = self._diagnose_quantile_output(raw_model_output)

        # timer.add_time("Actor prediction", time.time() - t)
        # t = time.time()

        # Pick action according to policy
        if self.policy == "greedy":
            actions = q_vals.argmax(axis=1)
        else:  # softmax
            probs = np.exp(q_vals) / np.sum(np.exp(q_vals), axis=1, keepdims=True)
            actions = random_choice_along_last_axis(probs)

        # timer.add_time("Action pick", time.time() - t)
        # t = time.time()

        predicted_returns = np.max(q_vals, axis=1)

        # timer.add_time("Return computation", time.time() - t)

        return actions, predicted_returns

    def update_lr(self, current_transition):
        new_lr = self.learning_rate.get_value(current_transition)
        self.optimizer.learning_rate.assign(new_lr)

    def _update_convnext_finetune_state(self, current_transition, convnext_finetune_at_step=None):
        if convnext_finetune_at_step is None:
            enabled = True
        else:
            enabled = int(current_transition) >= int(convnext_finetune_at_step)

        previous = self._last_convnext_update_enabled
        self.convnext_update_enabled.assign(bool(enabled))
        if previous is None or bool(previous) != bool(enabled):
            state = "enabled" if enabled else "frozen"
            if convnext_finetune_at_step is None:
                print_info(f"ConvNeXt backbone gradients: {state}.")
            else:
                print_info(
                    f"ConvNeXt backbone gradients: {state} "
                    f"(transition={int(current_transition)}, finetune_at={int(convnext_finetune_at_step)}, "
                    f"lr_scale={float(self.convnext_gradient_scale.numpy()):.4g})."
                )
            self._last_convnext_update_enabled = bool(enabled)

    def _gate_convnext_gradients(self, grads, weights):
        scale = tf.where(
            self.convnext_update_enabled,
            self.convnext_gradient_scale,
            tf.constant(0.0, dtype=tf.float32),
        )
        gated_grads = []
        for grad, weight in zip(grads, weights):
            if grad is None:
                gated_grads.append(None)
                continue
            name = weight.name.lower()
            if "convnext" in name and "convnext_image_feature" not in name:
                gated_grads.append(tf.multiply(grad, scale))
            else:
                gated_grads.append(grad)
        return gated_grads

    def reset_noise(self):
        models = {self.online_learner, self.target_learner, self.actor}
        for model in list(models):
            self.reset_noise_for(model)

    def reset_noise_for(self, model):
        q_net = model.layers[self.q_net_layer_id]
        q_net.reset_noise()

    def set_noisy(self, model, active: bool):
        q_net = model.layers[self.q_net_layer_id]
        q_net.set_noisy(active)

    def set_policy(self, policy):
        assert policy in ["greedy", "softmax"]
        self.policy = policy

    def learn_entire_experience(self, batch_size, epochs, gamma, alpha):
        experience_length = self.memory.get_num_transitions()
        self.learn_instances(experience_length, gamma, batch_size, epochs, alpha)

    def just_play(self, num_par_envs=1, policy="greedy", **kwargs):
        print("Just playing around...")
        self.set_policy(policy)

        if self.num_par_envs != num_par_envs:
            self.reinit_env(num_par_envs)
            self.num_par_envs = num_par_envs
            self.actor = self.online_learner
            self.stacker = StateStacker(self.state_shapes, self.state_dtypes, self.stack_size, self.num_par_envs)

        self.play_parallel(**kwargs)

    def play_parallel(self, epsilon=0, render_environment=True, verbose=False):
        """Plays one or more envs in parallel."""
        if verbose and self.num_par_envs == 1:
            print(" Pred. return |       Action |       Reward ")
            print("--------------------------------------------")

        self.env.reset()
        ret, score, env_time, step = 0, 0, 0, 0

        while True:
            states = self.env.get_states()
            self.stacker.add_states(states)

            # Plan action
            actions, pred_rets = self.plan_epsilon_greedy(self.stacker.get_stacks(), epsilon, compute_return=True)

            # Env step
            reward, score, terminals, env_time, _, game_overs = self.env.step(actions)
            if render_environment:
                self.env.render()

            if verbose:
                if self.num_par_envs == 1:
                    ret += reward[0]
                    rew_text = "{:>12.2f} ".format(reward[0])
                    if reward[0] > 0:
                        rew_text = green(rew_text)
                    elif reward[0] < 0:
                        rew_text = red(rew_text)
                    print("{:>13.2f}".format(pred_rets[0]) + " | " +
                          "{:>12s}".format(self.env.actions[actions[0]]) + " | " + rew_text)

                    if game_overs[0]:
                        print("--------------------------------------------")
                        print("Level finished with return %.2f, score %d, and time %d.\n" % (ret, score, env_time))
                        ret = 0

                else:
                    if step % 10 == 0:
                        print(f"Parallel step {step}.")

            # Handle finished envs
            self.env.reset_finished()
            self.stacker.reset_stacks(np.where(terminals)[0])

            step += 1

    def test_on_levels(self, render=False):
        test_env = self.env.copy(1)
        test_env.set_mode(test_env.TEST_MODE)
        num_levels = len(test_env.levels_list)
        test_scores = np.zeros(num_levels)

        for level in range(num_levels):
            # Play a whole level
            test_env.reset(lvl_no=level)
            if render:
                test_env.render()
                time.sleep(0.35)

            score = 0
            game_over = False
            while not game_over:
                state = test_env.get_states()

                # Predict the next action to take (move, rotate or do nothing)
                action, _ = self.plan_epsilon_greedy(state, 0)

                # Perform action, observe new environment state, level score and application state
                _, score, game_over, _, _ = test_env.step(action)

                if render:
                    test_env.render()
                    # time.sleep(0.35)

            test_scores[level] = score

        _, highscores_human = test_env.get_highscores()
        plot_highscores(test_scores, highscores_human, self.model_dir)

    def get_config(self):
        """Keep compatible with load_model()!"""
        stem_config = self.stem_network.get_config()
        q_config = self.q_network.get_config()
        agent_config = {"replay_batch_size": self.replay_batch_size,
                        "stack_size": self.stack_size,
                        "seed": self.seed,
                        "auxiliary_heads_config": self.auxiliary_heads_config}

        config = {"stem_model_class": self.stem_network.__class__.__name__,
                  "stem_model_config": stem_config,
                  "q_network_class": self.q_network.__class__.__name__,
                  "q_network_config": q_config,
                  "env_config": self.env.get_config(),
                  "agent_config": agent_config}

        return config

    def save_config(self):
        config = self.get_config()
        config2json(config, out_path=self.model_dir + "config.json")

    def save_training_parameters(self, parameters_dict: dict):
        out_path = self.get_training_config_path()
        config2json(parameters_dict, out_path=out_path)

    def get_training_config_path(self):
        return self.model_dir + "training_config.json"

    def save_weights(self, out_path=None, overwrite=False, checkpoint_no=None):
        """Saves the current model weights and statistics to a specified export path."""
        checkpoint_dir = None
        if checkpoint_no is not None:
            checkpoint_dir = out_path if out_path is not None else os.path.join(self.model_dir, "checkpoints")
            model_path = os.path.join(checkpoint_dir, _checkpoint_filename(checkpoint_no))
        elif out_path is None:
            model_path = os.path.join(self.model_dir, FINAL_WEIGHTS_FILENAME)
        elif out_path.endswith(KERAS_WEIGHTS_SUFFIX):
            model_path = out_path
        else:
            model_path = out_path + KERAS_WEIGHTS_SUFFIX

        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            _prune_old_checkpoints(checkpoint_dir, keep_count=max(0, MAX_CHECKPOINTS_TO_KEEP - 1))

        temp_path = _checkpoint_temp_path(model_path)
        try:
            self.online_learner.save_weights(temp_path, overwrite=True)
            os.replace(temp_path, model_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

        if checkpoint_dir is not None:
            _prune_old_checkpoints(checkpoint_dir, keep_count=MAX_CHECKPOINTS_TO_KEEP)

        print(f"Saved model weights: {model_path}")

    def export_saved_model(self):
        """학습된 모델을 TF SavedModel 형식으로 내보냅니다.
        소스 코드 없이 로드 가능하며, 경쟁 평가 시 이 파일이 사용됩니다.

        입력 시그니처:
          image : (batch, 128, 128, 3)  float32
          bird  : (batch, 5)            float32
        출력:
          Q-values / logits : (batch, 200)  float32
        """
        out_path = self.model_dir + "saved_model"
        model = self.online_learner  # Keras Model
        q_net = model.layers[self.q_net_layer_id]
        had_noisy_training = getattr(q_net, "noise_std_init", 0) > 0
        q_net.set_noisy(False)

        @tf.function(input_signature=[
            tf.TensorSpec([None, 128, 128, 3], tf.float32, name="image"),
            tf.TensorSpec([None, 5],           tf.float32, name="bird"),
        ])
        def serving_fn(image, bird):
            output = self._extract_q_output(model([image, bird], training=False))
            if self.is_c51_distributional():
                support = tf.constant(self.get_distributional_support_np(), dtype=tf.float32)
                return tf.reduce_sum(output * support, axis=-1, name="expected_q_values")
            if self.is_quantile_distributional():
                return tf.reduce_mean(output, axis=-1, name="expected_q_values")
            return output

        try:
            tf.saved_model.save(model, out_path,
                                signatures={"serving_default": serving_fn})
        finally:
            if had_noisy_training:
                q_net.set_noisy(True)
        print(f"  SavedModel 내보내기 완료: {out_path}")

    def save_experience(self, experience_path=None, overwrite=False, compress=False):
        pass

    def restore_experience(self, experience_path=None, gamma=None):
        pass

    def forget(self):
        self.memory = ReplayMemory(**self.memory.get_config())

    def write_hyperparams_file(self, num_parallel_steps, replay_period, replay_size_multiplier, target_sync_period,
                               actor_sync_period, alpha, gamma, min_hist_len):
        hyperparams_file = open(self.model_dir + "hyperparams.txt", "w+")
        text = "num_parallel_steps: %d" % num_parallel_steps + \
               "\nnum_parallel_envs: %d" % self.num_par_envs + \
               "\npolicy: %s" % self.policy + \
               "\noptimizer: %s" % self.optimizer.get_config()['name'] + \
               "\nreplay_period: %d" % replay_period + \
               "\nreplay_size_multiplier: %d" % replay_size_multiplier + \
               "\nmin_hist_len: %d" % min_hist_len + \
               "\ntarget_sync_period: " + str(target_sync_period) + \
               "\nactor_sync_period: " + str(actor_sync_period) + \
               "\n\nalpha: %f" % alpha + \
               "\ngamma: %f" % gamma + \
               "\nuse_mc_return: " + str(self.use_mc_return) + \
               "\nstack_size: %d" % self.stack_size + \
               "\nseed: " + str(self.seed)

        text += "\n\nSTEM MODEL PARAMETERS:"
        text += config2text(self.stem_network.get_config())

        text += "\n\nQ-NETWORK PARAMETERS:"
        text += config2text(self.q_network.get_config())

        text += "\n\nLEARNING RATE:"
        text += config2text(self.learning_rate.get_config())

        text += "\n\nEPSILON:"
        text += config2text(self.epsilon.get_config())

        text += "\n\nMEMORY:"
        text += config2text(self.memory.get_config())

        hyperparams_file.write(text + "\n\n")
        self.online_learner.summary(print_fn=lambda x: hyperparams_file.write(x + '\n'))
        hyperparams_file.close()

    def check_replay_size(self, replay_size_multiplier, replay_period, max_replay_size):
        exp_replay_size = replay_size_multiplier * replay_period * self.num_par_envs
        if max_replay_size is not None and exp_replay_size > max_replay_size:
            print_warning("Warning: Expected replay size %d is larger than given max replay size %d." %
                          (exp_replay_size, max_replay_size))


def continue_practice(model_name: str,
                      env_type: type(ParallelEnvironment),
                      num_par_envs: int = None,
                      checkpoint_no: int = None,
                      stem_config_override: dict = None,
                      **training_parameters_override):
    # Restore agent and training parameters
    agent = restore(model_name=model_name,
                    env_type=env_type,
                    num_par_envs=num_par_envs,
                    checkpoint_no=checkpoint_no,
                    stem_config_override=stem_config_override)
    training_parameters = restore_training_parameters(model_name=model_name, env_type=env_type)

    target_parallel_steps_override = training_parameters_override.pop("num_parallel_steps", None)
    if target_parallel_steps_override is not None:
        training_parameters["num_parallel_steps"] = int(target_parallel_steps_override)

    # Override training parameters.  training_config.json is rewritten at the
    # start of every practice() call, so on resume it can contain a remaining
    # step count rather than the original run target.  Prefer stable metadata
    # when available and account from the checkpoint that was actually loaded.
    target_parallel_steps = _resolve_resume_target_parallel_steps(
        model_name=model_name,
        env_type=env_type,
        training_parameters=training_parameters,
    )
    restored_transition = int(agent.stats.get_current_transition())
    restored_checkpoint = getattr(agent, "restored_checkpoint_no", None)
    if restored_checkpoint is not None:
        restored_checkpoint = int(restored_checkpoint)
        if restored_transition > restored_checkpoint:
            print_warning(
                f"Stats transition ({restored_transition}) is ahead of loaded checkpoint "
                f"({restored_checkpoint}); trimming stats accounting to the checkpoint."
            )
            _trim_stats_to_transition(agent.stats, restored_checkpoint)

        restored_transition = restored_checkpoint + agent.num_par_envs
        agent.stats.init_trans_no = restored_transition

    completed_parallel_steps = int(restored_transition // agent.num_par_envs)
    remaining_parallel_steps = max(0, int(target_parallel_steps) - completed_parallel_steps)
    print(
        "Resume accounting: "
        f"target={int(target_parallel_steps)} parallel steps, "
        f"checkpoint={restored_checkpoint}, "
        f"next_transition={restored_transition}, "
        f"remaining={remaining_parallel_steps} parallel steps"
    )
    training_parameters["num_parallel_steps"] = remaining_parallel_steps
    for parameter in training_parameters_override.keys():
        training_parameters[parameter] = training_parameters_override[parameter]

    # Continue practicing
    agent.practice(**training_parameters)


def _resolve_resume_target_parallel_steps(model_name: str,
                                          env_type: type(ParallelEnvironment),
                                          training_parameters: dict) -> int:
    targets = []
    config_steps = training_parameters.get("num_parallel_steps")
    if config_steps is not None:
        targets.append(int(config_steps))

    run_metadata = training_parameters.get("run_metadata")
    if isinstance(run_metadata, dict):
        preset_info = run_metadata.get("training_size_preset_info", {})
        metadata_steps = preset_info.get("num_parallel_steps")
        if metadata_steps is not None:
            targets.append(int(metadata_steps))

    metadata_path = os.path.join(get_model_dir(env_type, model_name), "run_metadata.json")
    if os.path.exists(metadata_path):
        try:
            metadata = json2config(metadata_path)
            preset_info = metadata.get("training_size_preset_info", {})
            metadata_steps = preset_info.get("num_parallel_steps")
            if metadata_steps is not None:
                targets.append(int(metadata_steps))
        except Exception as exc:
            print_warning(f"Warning: unable to read resume metadata from {metadata_path}: {exc}")

    if not targets:
        raise ValueError("Unable to determine num_parallel_steps for resume.")
    return max(targets)


def _trim_stats_to_transition(stats: Statistics, max_transition: int):
    try:
        episode_transitions = stats.episode_stats[:stats.episode_ptr, stats.TRANSITION]
        cycle_transitions = stats.cycle_stats[:stats.cycle_ptr, stats.TRANSITION]
        stats.episode_ptr = int(np.count_nonzero(episode_transitions <= max_transition))
        stats.cycle_ptr = int(np.count_nonzero(cycle_transitions <= max_transition))
        stats.init_cyc_no = max(0, stats.cycle_ptr - 1)
        if stats.episode_ptr > 0:
            stats.total_timer = stats.episode_stats[stats.episode_ptr - 1, stats.SECONDS]
        else:
            stats.total_timer = 0
    except Exception as exc:
        print_warning(f"Warning: unable to trim stats to checkpoint {max_transition}: {exc}")


def restore(model_name: str,
            env_type: type(ParallelEnvironment),
            num_par_envs: int = None,
            checkpoint_no: int = None,
            stem_config_override: dict = None):
    """Loads the most recently saved checkpoint of the specified model."""
    model_dir = get_model_dir(env_type, model_name)

    if not os.path.exists(model_dir):
        raise ValueError(f"There is no {env_type.__name__} model named '{model_name}'.")

    config_path = model_dir + "config.json"
    weights_paths = _resolve_checkpoint_paths(model_dir, checkpoint_no)

    print(f"Restoring model and statistics from '{model_dir}'.")

    config = json2config(config_path)

    stem_class = get_class_from_name(config["stem_model_class"])
    stem_config = config["stem_model_config"]
    if stem_config_override:
        stem_config.update(stem_config_override)
    q_class = get_class_from_name(config["q_network_class"])
    q_config = config["q_network_config"]
    env_config = config["env_config"]
    agent_config = config["agent_config"]

    if env_config is None:
        env_config = dict()

    if num_par_envs is not None:
        env_config.update({"num_par_inst": num_par_envs})

    stem_net = stem_class(**stem_config)
    q_net = q_class(**q_config)
    env = env_type(**env_config)

    # Load the agent, model weights and statistics
    agent = Agent(name=model_name, env=env, stem_network=stem_net,
                  q_network=q_net, override=True, **agent_config)
    loaded_weights_path = None
    last_load_error = None
    for weights_path in weights_paths:
        try:
            agent.online_learner.load_weights(weights_path)
            loaded_weights_path = weights_path
            break
        except Exception as exc:
            last_load_error = exc
            print_warning(f"Warning: unable to load checkpoint {weights_path}: {exc}")
    if loaded_weights_path is None:
        raise RuntimeError("Unable to load any checkpoint for restore.") from last_load_error

    restored_checkpoint_no = _checkpoint_step_from_file_name(os.path.basename(loaded_weights_path))
    agent.restored_checkpoint_no = restored_checkpoint_no
    if restored_checkpoint_no is not None:
        agent.stats.init_trans_no = max(agent.stats.init_trans_no, int(restored_checkpoint_no))

    agent.stats.load(model_dir)
    if restored_checkpoint_no is not None:
        agent.stats.init_trans_no = max(agent.stats.init_trans_no, int(restored_checkpoint_no))

    print(f"Successfully restored from {loaded_weights_path}.")
    return agent


def restore_training_parameters(model_name: str, env_type: type(ParallelEnvironment)) -> dict:
    model_path = get_model_dir(env_type, model_name)
    config = json2config(model_path + "training_config.json")
    config["learning_rate"] = ParamScheduler(**config["learning_rate"])
    config["epsilon"] = ParamScheduler(**config["epsilon"])
    return config


def compute_sample_weights(sample_priorities, sample_probabilities, total_size, beta=0.5):
    """Computes sample weights for training. Part of Prioritized Experience Replay."""
    weights = (total_size * sample_probabilities) ** (- beta)  # importance-sampling weights
    weights /= np.max(weights)  # normalization
    return np.abs(np.multiply(weights, sample_priorities))


def get_n_step_return(pred_next_returns, n_step_rewards, step_mask, gamma):
    # Prepare
    n = n_step_rewards.shape[-1]
    step_axis = n_step_rewards.ndim - 1

    # Compute n-step return
    discounts = gamma ** np.arange(n + 1)
    rewards_returns = np.append(n_step_rewards, np.expand_dims(pred_next_returns, axis=1), axis=step_axis)
    rewards_returns_discounted = rewards_returns * discounts
    rewards_returns_discounted[~ step_mask] = 0
    n_step_returns = np.sum(rewards_returns_discounted, axis=step_axis)

    return n_step_returns


def play(model_name: str,
         env_type: type(ParallelEnvironment),
         num_par_envs: int = None,
         checkpoint_no: int = None,
         mode=None,
         **kwargs):
    # Restore agent and training parameters
    agent = restore(model_name=model_name,
                    env_type=env_type,
                    num_par_envs=num_par_envs,
                    checkpoint_no=checkpoint_no)

    if mode is not None:
        agent.env.set_mode(mode)

    agent.just_play(num_par_envs, **kwargs)


def load_and_test(model_name, env_type, checkpoint_no=None, mode=None, render=False):
    agent = load_model(model_name, env_type, checkpoint_no)
    if mode is not None:
        agent.env.set_mode(mode)
    agent.test_on_levels(render)


def benchmark(env_type: Type[ParallelEnvironment], stem_network: StemNetwork, q_network: QNetwork,
              num_par_envs_list, replay_period, replay_batch_size, **env_kwargs):
    """Assuming
    replay_size_multiplier == 1
    n_step == 1
    stack_size == 1"""

    num_rounds = len(num_par_envs_list)
    steps_per_round = replay_period * 4

    for benchmark_round in range(num_rounds):
        num_par_envs = num_par_envs_list[benchmark_round]
        env = env_type(num_par_inst=num_par_envs, **env_kwargs)
        agent = Agent("tmp", env, stem_network, q_network, replay_batch_size)
        agent.practice(steps_per_round, replay_period, 1)
        # TODO


def get_model_dir(env: ParallelEnvironment, model_name: str):
    return f"out/{env.NAME}/{model_name}/"
