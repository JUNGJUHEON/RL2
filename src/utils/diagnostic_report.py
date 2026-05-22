"""Human/GPT-readable reports from training diagnostics.jsonl files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ACTIVATION_LAYERS = (
    "convnext_pixel_input",
    "convnext_image_feature",
    "bird_embedding",
    "latent",
    "double_Q_network",
    "default_Q_network",
    "distributional_dueling_Q_network",
    "quantile_dueling_Q_network",
)

STAT_FIELDS = ("shape", "dtype", "nan_count", "inf_count", "min", "max", "mean", "std", "sum")

LEARNING_ARRAYS = (
    "q_values",
    "next_q_values",
    "next_online_q_values",
    "pred_returns",
    "pred_next_returns",
    "target_returns",
    "td_errors",
    "abs_td_errors",
    "sample_weights",
    "sample_probabilities",
    "n_step_rewards",
    "current_distributions",
    "next_online_distributions",
    "next_target_distributions",
    "projected_target_distributions",
    "predictions_after_fit",
    "targets",
)

DIST_ARRAYS = ("distribution", "atom_entropy", "atom_probability_sums", "support")
QUANTILE_GROUPS = (
    "current_quantiles",
    "next_online_quantiles",
    "next_target_quantiles",
    "target_action_quantiles",
    "predictions_after_fit",
    "targets",
)
QUANTILE_ARRAYS = ("quantiles", "q_values", "quantile_std", "quantile_p10", "quantile_p90")


def _load_plotting():
    mpl_config = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    return plt, pd


def _shape_to_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "x".join(str(part) for part in value)
    return str(value)


def _flatten_stat(row: dict[str, Any], prefix: str, stats: Any) -> None:
    if not isinstance(stats, dict):
        return
    for field in STAT_FIELDS:
        if field not in stats:
            continue
        value = stats[field]
        if field == "shape":
            value = _shape_to_text(value)
        row[f"{prefix}_{field}"] = value


def _flatten_distribution_group(row: dict[str, Any], prefix: str, group: Any) -> None:
    if not isinstance(group, dict):
        return
    for name in DIST_ARRAYS:
        _flatten_stat(row, f"{prefix}_{name}", group.get(name))


def _flatten_quantile_group(row: dict[str, Any], prefix: str, group: Any) -> None:
    if not isinstance(group, dict):
        return
    for name in QUANTILE_ARRAYS:
        _flatten_stat(row, f"{prefix}_{name}", group.get(name))
    if "num_quantiles" in group:
        row[f"{prefix}_num_quantiles"] = group.get("num_quantiles")


def _safe_column_name(value: Any) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _flatten_auxiliary_group(row: dict[str, Any], prefix: str, group: Any) -> None:
    if not isinstance(group, dict):
        return
    if any(field in group for field in STAT_FIELDS):
        _flatten_stat(row, prefix, group)
    else:
        _flatten_stat(row, prefix, group.get("all"))
    for name, stats in group.items():
        if name == "all":
            continue
        safe_name = _safe_column_name(name)
        if safe_name:
            _flatten_stat(row, f"{prefix}_{safe_name}", stats)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append({
                    "event": "diagnostic_parse_error",
                    "line_no": line_no,
                    "error": str(exc),
                })
    return records


def _activation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        activations = record.get("model_activations")
        if record.get("event") != "training_step" or not isinstance(activations, dict):
            continue
        if not activations:
            continue

        row = {
            "event": record.get("event"),
            "model": record.get("model"),
            "wall_time": record.get("wall_time"),
            "loop_step": record.get("loop_step"),
            "transition": record.get("transition"),
            "epsilon": record.get("epsilon"),
            "learning_rate": record.get("learning_rate"),
            "memory_transitions": record.get("memory_transitions"),
            "action": record.get("action"),
            "action_text": record.get("action_text"),
            "reward": record.get("reward"),
            "score": record.get("score"),
            "episode_return": record.get("episode_return"),
            "terminal": record.get("terminal"),
            "win": record.get("win"),
            "game_over": record.get("game_over"),
            "env_time": record.get("env_time"),
            "full_diagnostics": record.get("full_diagnostics"),
        }

        q_diag = record.get("q", {})
        if isinstance(q_diag, dict):
            row["policy_decision"] = q_diag.get("policy_decision")
            row["policy_epsilon"] = q_diag.get("epsilon")
            row["q_distributional"] = q_diag.get("distributional")
            row["q_min"] = q_diag.get("min")
            row["q_max"] = q_diag.get("max")
            row["q_mean"] = q_diag.get("mean")
            row["q_std"] = q_diag.get("std")
            row["q_best_actions"] = json.dumps(q_diag.get("best_actions", []))
            row["q_top5_actions_first_env"] = json.dumps(q_diag.get("top5_actions_first_env", []))
            if isinstance(q_diag.get("qr_dqn"), dict):
                _flatten_quantile_group(row, "policy_qr_dqn", q_diag.get("qr_dqn"))
            if isinstance(q_diag.get("c51"), dict):
                _flatten_distribution_group(row, "policy_c51", q_diag.get("c51"))

        action_diag = record.get("action_distribution", {})
        if isinstance(action_diag, dict):
            row["action_total"] = action_diag.get("total_actions")
            row["action_unique"] = action_diag.get("unique_actions")
            row["action_entropy"] = action_diag.get("entropy")
            row["angle_entropy"] = action_diag.get("angle_entropy")
            row["tap_entropy"] = action_diag.get("tap_entropy")
            row["top_actions"] = json.dumps(action_diag.get("top10_actions", []))
            row["top_action_counts"] = json.dumps(action_diag.get("top10_counts", []))
            row["top_angle_bins"] = json.dumps(action_diag.get("top5_angle_bins", []))
            row["top_angle_counts"] = json.dumps(action_diag.get("top5_angle_counts", []))
            row["top_tap_bins"] = json.dumps(action_diag.get("top5_tap_bins", []))
            row["top_tap_counts"] = json.dumps(action_diag.get("top5_tap_counts", []))

        env_step = record.get("env_step", {})
        if isinstance(env_step, dict):
            for key in (
                "reward_profile",
                "level",
                "shot_idx",
                "action",
                "angle",
                "tap_ms",
                "score_before",
                "score_after",
                "score_after_raw",
                "score_after",
                "score_delta",
                "score_delta_raw",
                "score_regression_guarded",
                "score_reward",
                "win_bonus",
                "loss_penalty",
                "shot_penalty",
                "has_previous_best_score",
                "best_score_before",
                "best_score_credit_before",
                "best_score_after",
                "best_score_improvement",
                "best_score_bonus",
                "current_attempt_best_credit",
                "best_score_updated_on_game_over",
                "tap_score_bonus",
                "tap_win_bonus",
                "pig_proxy_units",
                "pig_proxy_bonus",
                "proxy_bonus",
                "final_reward",
                "won",
                "lost",
                "game_over",
                "app_state",
            ):
                row[f"env_{key}"] = env_step.get(key)

        convnext = record.get("convnext", {})
        if isinstance(convnext, dict):
            for key, value in convnext.items():
                row[f"convnext_{key}"] = value

        _flatten_stat(row, "recent_rewards", record.get("recent_rewards"))
        for layer in ACTIVATION_LAYERS:
            _flatten_stat(row, layer, activations.get(layer))
        rows.append(row)
    return rows


def _episode_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("event") != "episode_end":
            continue
        row = {
            "model": record.get("model"),
            "wall_time": record.get("wall_time"),
            "loop_step": record.get("loop_step"),
            "transition": record.get("transition"),
            "env_id": record.get("env_id"),
            "level": record.get("level"),
            "episode_return": record.get("episode_return"),
            "score": record.get("score"),
            "shots": record.get("shots"),
            "win": record.get("win"),
            "memory_transitions": record.get("memory_transitions"),
        }
        action_diag = record.get("action_distribution", {})
        if isinstance(action_diag, dict):
            row["action_unique"] = action_diag.get("unique_actions")
            row["action_entropy"] = action_diag.get("entropy")
            row["angle_entropy"] = action_diag.get("angle_entropy")
            row["tap_entropy"] = action_diag.get("tap_entropy")
            row["top_actions"] = json.dumps(action_diag.get("top10_actions", []))
            row["top_tap_bins"] = json.dumps(action_diag.get("top5_tap_bins", []))
        rows.append(row)
    return rows


def _learning_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("event") != "learning_update":
            continue
        learning = record.get("learning", {})
        gradients = record.get("gradients", {})
        row = {
            "model": record.get("model"),
            "wall_time": record.get("wall_time"),
            "loop_step": record.get("loop_step"),
            "transition": record.get("transition"),
            "replay_size": record.get("replay_size"),
            "learned_transitions": record.get("learned_transitions"),
            "memory_transitions": record.get("memory_transitions"),
            "loss": record.get("loss"),
            "learning_rate": record.get("learning_rate"),
        }
        if isinstance(learning, dict):
            row["learning_mode"] = learning.get("mode")
            row["requested_instances"] = learning.get("requested_instances")
            row["sample_count"] = learning.get("sample_count")
            row["n_step"] = learning.get("n_step")
            row["c51_loss"] = learning.get("c51_loss")
            row["quantile_loss"] = learning.get("quantile_loss")
            row["num_quantiles"] = learning.get("num_quantiles")
            row["step_mask_true_fraction"] = learning.get("step_mask_true_fraction")
            for name in LEARNING_ARRAYS:
                value = learning.get(name)
                if name in QUANTILE_GROUPS:
                    if isinstance(value, dict) and "quantiles" in value:
                        _flatten_quantile_group(row, name, value)
                    else:
                        _flatten_distribution_group(row, name, value)
                elif name.endswith("distributions") or name in {"predictions_after_fit", "targets"}:
                    _flatten_distribution_group(row, name, value)
                else:
                    _flatten_stat(row, name, value)
            for name in QUANTILE_GROUPS:
                if name not in LEARNING_ARRAYS:
                    _flatten_quantile_group(row, name, learning.get(name))
            auxiliary = learning.get("auxiliary_heads")
            row["auxiliary_heads_enabled"] = bool(auxiliary) if auxiliary is not None else False
            row["auxiliary_loss_weight"] = learning.get("auxiliary_loss_weight")
            row["q_loss"] = learning.get("q_loss")
            row["auxiliary_loss"] = learning.get("auxiliary_loss")
            row["weighted_auxiliary_loss"] = learning.get("weighted_auxiliary_loss")
            _flatten_auxiliary_group(row, "auxiliary_targets", learning.get("auxiliary_targets"))
            _flatten_auxiliary_group(row, "auxiliary_predictions", learning.get("auxiliary_predictions"))
        if isinstance(gradients, dict):
            for name, value in gradients.items():
                row[f"gradient_norm_{name}"] = value
        convnext = record.get("convnext", {})
        if isinstance(convnext, dict):
            for name, value in convnext.items():
                row[f"convnext_{name}"] = value
        rows.append(row)
    return rows


def _mean(series):
    if len(series) == 0:
        return None
    return float(series.mean())


def _last(series):
    if len(series) == 0:
        return None
    return series.iloc[-1]


def _build_summary(pd, records, activation_df, episode_df, learning_df, checkpoint_label):
    starts = [record for record in records if record.get("event") == "training_start"]
    start = starts[0] if starts else {}
    run_metadata = start.get("run_metadata", {}) if isinstance(start.get("run_metadata"), dict) else {}
    env_config = start.get("env", {}) if isinstance(start.get("env"), dict) else {}
    reward_config = env_config.get("reward", {}) if isinstance(env_config.get("reward"), dict) else {}
    train_levels = env_config.get("train_levels", []) if isinstance(env_config.get("train_levels"), list) else []

    def num_col(df, name):
        if name not in df:
            return pd.Series([], dtype="float64")
        return pd.to_numeric(df[name], errors="coerce").dropna()

    def bool_col(df, name):
        if name not in df:
            return pd.Series([], dtype=bool)
        return df[name].dropna().astype(bool)

    feature_nan_total = 0
    feature_inf_total = 0
    q_nan_total = 0
    q_inf_total = 0
    for column in (
        "convnext_pixel_input_nan_count",
        "convnext_image_feature_nan_count",
        "bird_embedding_nan_count",
        "latent_nan_count",
    ):
        if column in activation_df:
            feature_nan_total += int(activation_df[column].fillna(0).sum())
    for column in (
        "convnext_pixel_input_inf_count",
        "convnext_image_feature_inf_count",
        "bird_embedding_inf_count",
        "latent_inf_count",
    ):
        if column in activation_df:
            feature_inf_total += int(activation_df[column].fillna(0).sum())
    for column in (
        "distributional_dueling_Q_network_nan_count",
        "quantile_dueling_Q_network_nan_count",
        "double_Q_network_nan_count",
        "default_Q_network_nan_count",
    ):
        if column in activation_df:
            q_nan_total += int(activation_df[column].fillna(0).sum())
    for column in (
        "distributional_dueling_Q_network_inf_count",
        "quantile_dueling_Q_network_inf_count",
        "double_Q_network_inf_count",
        "default_Q_network_inf_count",
    ):
        if column in activation_df:
            q_inf_total += int(activation_df[column].fillna(0).sum())

    activation_rows = int(len(activation_df))
    feature_std_min = None
    feature_std_mean = None
    if "convnext_image_feature_std" in activation_df and activation_rows:
        feature_std_min = float(activation_df["convnext_image_feature_std"].min())
        feature_std_mean = float(activation_df["convnext_image_feature_std"].mean())

    if activation_rows == 0:
        convnext_status = "missing_activation_rows"
        convnext_note = "No ConvNeXt activation rows were found in diagnostics."
    elif feature_nan_total or feature_inf_total:
        convnext_status = "attention_needed"
        convnext_note = "ConvNeXt-related tensors contain NaN or Inf values."
    elif feature_std_min is not None and feature_std_min <= 1e-6:
        convnext_status = "attention_needed"
        convnext_note = "ConvNeXt image features look collapsed or nearly constant."
    else:
        convnext_status = "healthy_technical_signal"
        convnext_note = (
            "ConvNeXt is producing finite, non-constant image features. "
            "That means the visual stem is technically working, but policy quality still needs evaluation."
        )

    loss_first = None
    loss_last = None
    loss_min = None
    loss_max = None
    if "loss" in learning_df and len(learning_df):
        losses = learning_df["loss"].dropna()
        if len(losses):
            loss_first = float(losses.iloc[0])
            loss_last = float(losses.iloc[-1])
            loss_min = float(losses.min())
            loss_max = float(losses.max())

    if loss_first is None:
        rl_status = "no_learning_updates_yet"
        rl_note = "Replay learning has not produced a logged learning_update yet."
    elif loss_last <= loss_first:
        rl_status = "learning_loss_decreased"
        rl_note = "RL loss is lower at this checkpoint than at the first logged update."
    else:
        rl_status = "learning_loss_not_lower_yet"
        rl_note = "RL loss is noisy or not lower yet; check longer training and evaluation."

    wins = episode_df["win"].astype(bool) if "win" in episode_df and len(episode_df) else pd.Series([], dtype=bool)
    level_values = []
    if "level" in episode_df:
        level_values.extend(num_col(episode_df, "level").astype(int).tolist())
    if "env_level" in activation_df:
        level_values.extend(num_col(activation_df, "env_level").astype(int).tolist())
    unique_levels_seen = len(set(level_values))
    num_train_levels = int(env_config.get("num_train_levels") or len(train_levels) or 0)
    level_coverage = (unique_levels_seen / num_train_levels) if num_train_levels else None

    proxy_bonus = num_col(activation_df, "env_proxy_bonus")
    tap_score_bonus = num_col(activation_df, "env_tap_score_bonus")
    tap_win_bonus = num_col(activation_df, "env_tap_win_bonus")
    pig_proxy_bonus = num_col(activation_df, "env_pig_proxy_bonus")
    best_score_bonus = num_col(activation_df, "env_best_score_bonus")
    best_score_improvement = num_col(activation_df, "env_best_score_improvement")
    score_regression_guarded = bool_col(activation_df, "env_score_regression_guarded")
    tap_ms = num_col(activation_df, "env_tap_ms")

    quantile_std = num_col(learning_df, "current_quantiles_quantile_std_mean")
    quantile_p10 = num_col(learning_df, "current_quantiles_quantile_p10_mean")
    quantile_p90 = num_col(learning_df, "current_quantiles_quantile_p90_mean")
    quantile_spread = None
    if len(quantile_p10) and len(quantile_p90):
        quantile_spread = float((quantile_p90.reset_index(drop=True) - quantile_p10.reset_index(drop=True)).mean())

    auxiliary_enabled = bool_col(learning_df, "auxiliary_heads_enabled")
    auxiliary_loss = num_col(learning_df, "auxiliary_loss")
    weighted_auxiliary_loss = num_col(learning_df, "weighted_auxiliary_loss")
    q_loss = num_col(learning_df, "q_loss")
    if len(auxiliary_enabled) and bool(auxiliary_enabled.any()):
        auxiliary_status = "auxiliary_heads_logged"
    elif run_metadata.get("auxiliary_heads_enabled"):
        auxiliary_status = "auxiliary_selected_but_no_learning_update_yet"
    else:
        auxiliary_status = "not_auxiliary_run"

    convnext_update_series = None
    for df in (learning_df, activation_df):
        if "convnext_update_enabled" in df and len(df["convnext_update_enabled"].dropna()):
            convnext_update_series = df["convnext_update_enabled"].dropna().astype(bool)
    convnext_update_last = bool(convnext_update_series.iloc[-1]) if convnext_update_series is not None else None
    finetune_at = start.get("convnext_finetune_at_step")
    if convnext_update_last is True:
        convnext_finetune_status = "backbone_gradients_enabled"
    elif finetune_at is not None:
        convnext_finetune_status = "backbone_frozen_until_scheduled_finetune"
    else:
        convnext_finetune_status = "backbone_frozen_or_not_tracked"

    if len(quantile_std):
        qr_status = "qr_quantiles_logged"
    elif run_metadata.get("q_head_preset") == "qr_rainbow":
        qr_status = "qr_selected_but_no_learning_update_yet"
    else:
        qr_status = "not_qr_run"

    if reward_config.get("reward_profile") == "shaped_proxy_v1":
        proxy_reward_status = "proxy_reward_logged" if len(proxy_bonus) else "proxy_reward_selected_no_step_rows_yet"
    else:
        proxy_reward_status = "not_proxy_reward_run"

    if level_coverage is None:
        level_coverage_status = "unknown"
    elif level_coverage >= 0.9:
        level_coverage_status = "broad_all_map_coverage"
    elif level_coverage >= 0.25:
        level_coverage_status = "partial_map_coverage"
    else:
        level_coverage_status = "early_low_map_coverage"

    summary = {
        "checkpoint_label": checkpoint_label,
        "model": start.get("model"),
        "stem_model_class": start.get("stem_model_class"),
        "q_network_class": start.get("q_network_class"),
        "training_size_preset": run_metadata.get("training_size_preset"),
        "rainbow_run_variant": run_metadata.get("rainbow_run_variant"),
        "q_head_preset": run_metadata.get("q_head_preset"),
        "honest_run_name": run_metadata.get("current_honest_name"),
        "reward_profile": reward_config.get("reward_profile"),
        "train_level_pool": env_config.get("train_level_pool"),
        "num_train_levels": num_train_levels,
        "unique_levels_seen": unique_levels_seen,
        "train_level_coverage": level_coverage,
        "level_coverage_status": level_coverage_status,
        "activation_rows": activation_rows,
        "episode_count": int(len(episode_df)),
        "win_count": int(wins.sum()) if len(wins) else 0,
        "win_rate": float(wins.mean()) if len(wins) else None,
        "avg_episode_return": _mean(episode_df["episode_return"].dropna()) if "episode_return" in episode_df else None,
        "avg_score": _mean(episode_df["score"].dropna()) if "score" in episode_df else None,
        "avg_shots": _mean(episode_df["shots"].dropna()) if "shots" in episode_df else None,
        "learning_updates": int(len(learning_df)),
        "loss_first": loss_first,
        "loss_last": loss_last,
        "loss_min": loss_min,
        "loss_max": loss_max,
        "last_epsilon": float(_last(num_col(activation_df, "epsilon"))) if len(num_col(activation_df, "epsilon")) else None,
        "last_action_entropy": float(_last(num_col(activation_df, "action_entropy"))) if len(num_col(activation_df, "action_entropy")) else None,
        "last_angle_entropy": float(_last(num_col(activation_df, "angle_entropy"))) if len(num_col(activation_df, "angle_entropy")) else None,
        "last_tap_entropy": float(_last(num_col(activation_df, "tap_entropy"))) if len(num_col(activation_df, "tap_entropy")) else None,
        "avg_tap_ms_logged_steps": _mean(tap_ms),
        "proxy_bonus_positive_rate": float((proxy_bonus > 0).mean()) if len(proxy_bonus) else None,
        "proxy_reward_status": proxy_reward_status,
        "avg_proxy_bonus": _mean(proxy_bonus),
        "avg_tap_score_bonus": _mean(tap_score_bonus),
        "tap_score_bonus_count": int((tap_score_bonus > 0).sum()) if len(tap_score_bonus) else 0,
        "tap_win_bonus_count": int((tap_win_bonus > 0).sum()) if len(tap_win_bonus) else 0,
        "pig_proxy_bonus_count": int((pig_proxy_bonus > 0).sum()) if len(pig_proxy_bonus) else 0,
        "avg_pig_proxy_bonus": _mean(pig_proxy_bonus),
        "best_score_bonus_count": int((best_score_bonus > 0).sum()) if len(best_score_bonus) else 0,
        "best_score_improvement_count": int((best_score_improvement > 0).sum()) if len(best_score_improvement) else 0,
        "score_regression_guarded_count": int(score_regression_guarded.sum()) if len(score_regression_guarded) else 0,
        "qr_num_quantiles": int(_last(learning_df["num_quantiles"].dropna())) if "num_quantiles" in learning_df and len(learning_df["num_quantiles"].dropna()) else None,
        "qr_status": qr_status,
        "qr_current_quantile_std_mean": _mean(quantile_std),
        "qr_current_quantile_spread_p90_p10_mean": quantile_spread,
        "auxiliary_status": auxiliary_status,
        "auxiliary_loss_mean": _mean(auxiliary_loss),
        "auxiliary_loss_last": float(_last(auxiliary_loss)) if len(auxiliary_loss) else None,
        "weighted_auxiliary_loss_mean": _mean(weighted_auxiliary_loss),
        "weighted_auxiliary_loss_last": float(_last(weighted_auxiliary_loss)) if len(weighted_auxiliary_loss) else None,
        "q_loss_mean": _mean(q_loss),
        "q_loss_last": float(_last(q_loss)) if len(q_loss) else None,
        "convnext_update_enabled_last": convnext_update_last,
        "convnext_finetune_status": convnext_finetune_status,
        "convnext_gradient_scale_last": float(_last(num_col(learning_df, "convnext_gradient_scale"))) if len(num_col(learning_df, "convnext_gradient_scale")) else None,
        "gradient_norm_convnext_backbone_last": float(_last(num_col(learning_df, "gradient_norm_convnext_backbone"))) if len(num_col(learning_df, "gradient_norm_convnext_backbone")) else None,
        "gradient_norm_convnext_backbone_mean": _mean(num_col(learning_df, "gradient_norm_convnext_backbone")),
        "gradient_norm_image_projection_mean": _mean(num_col(learning_df, "gradient_norm_image_projection")),
        "gradient_norm_distributional_q_network_mean": _mean(num_col(learning_df, "gradient_norm_distributional_q_network")),
        "convnext_feature_nan_total": feature_nan_total,
        "convnext_feature_inf_total": feature_inf_total,
        "q_output_nan_total": q_nan_total,
        "q_output_inf_total": q_inf_total,
        "convnext_image_feature_std_min": feature_std_min,
        "convnext_image_feature_std_mean": feature_std_mean,
        "convnext_status": convnext_status,
        "convnext_note": convnext_note,
        "rl_status": rl_status,
        "rl_note": rl_note,
    }
    return summary


def _plot_activation_health(plt, activation_df, out_path: Path) -> None:
    if activation_df.empty:
        return

    x = activation_df["loop_step"] if "loop_step" in activation_df else range(len(activation_df))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax = axes[0]
    for column, label in (
        ("convnext_pixel_input_mean", "pixel mean"),
        ("convnext_pixel_input_std", "pixel std"),
    ):
        if column in activation_df:
            ax.plot(x, activation_df[column], label=label)
    ax.set_title("ConvNeXt input image statistics")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    for column, label in (
        ("convnext_image_feature_mean", "feature mean"),
        ("convnext_image_feature_std", "feature std"),
        ("convnext_image_feature_max", "feature max"),
    ):
        if column in activation_df:
            ax.plot(x, activation_df[column], label=label)
    ax.set_title("ConvNeXt image feature health")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[2]
    for column, label in (
        ("distributional_dueling_Q_network_std", "C51 output std"),
        ("distributional_dueling_Q_network_max", "C51 output max"),
        ("quantile_dueling_Q_network_std", "QR output std"),
        ("quantile_dueling_Q_network_max", "QR output max"),
        ("latent_std", "latent std"),
    ):
        if column in activation_df:
            ax.plot(x, activation_df[column], label=label)
    ax.set_title("Latent / Rainbow head activation health")
    ax.set_xlabel("training loop step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.suptitle("ConvNeXt + Rainbow Activation Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_rl_health(plt, pd, activation_df, episode_df, learning_df, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    ax = axes[0]
    if not learning_df.empty and "loss" in learning_df:
        x = learning_df["loop_step"] if "loop_step" in learning_df else range(len(learning_df))
        ax.plot(x, learning_df["loss"], label="loss", color="#1f77b4")
        if (learning_df["loss"].dropna() > 0).all():
            ax.set_yscale("log")
        ax.set_title("RL learning loss")
        ax.set_xlabel("training loop step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No learning_update rows yet", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1]
    if not episode_df.empty and "episode_return" in episode_df:
        x = episode_df["loop_step"] if "loop_step" in episode_df else range(len(episode_df))
        returns = episode_df["episode_return"]
        ax.plot(x, returns, label="episode return", color="#2ca02c", alpha=0.5)
        window = min(20, max(1, len(episode_df) // 5))
        ax.plot(x, returns.rolling(window=window, min_periods=1).mean(),
                label=f"return rolling {window}", color="#006d2c")
        ax.set_title("Reward trend")
        ax.set_xlabel("training loop step")
        ax.set_ylabel("return")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No episode_end rows yet", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[2]
    if not episode_df.empty and "win" in episode_df:
        x = episode_df["loop_step"] if "loop_step" in episode_df else range(len(episode_df))
        wins = episode_df["win"].astype(float)
        window = min(20, max(1, len(episode_df) // 5))
        ax.plot(x, wins.rolling(window=window, min_periods=1).mean(),
                label=f"win rate rolling {window}", color="#d62728")
        ax.set_ylim(-0.05, 1.05)
        if not activation_df.empty and "epsilon" in activation_df:
            ax2 = ax.twinx()
            ax2.plot(activation_df["loop_step"], activation_df["epsilon"],
                     label="epsilon", color="#9467bd", alpha=0.6)
            ax2.set_ylabel("epsilon")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax.legend(loc="best")
        ax.set_title("Win trend and exploration")
        ax.set_xlabel("training loop step")
        ax.set_ylabel("rolling win rate")
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No win data yet", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle("Rainbow DQN Training Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_markdown_summary(out_path: Path, diagnostics_path: Path, summary: dict[str, Any], artifacts: dict[str, str]) -> None:
    def fmt(value):
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "# Training Diagnostics Report",
        "",
        f"Source diagnostics: `{diagnostics_path}`",
        f"Checkpoint: `{summary.get('checkpoint_label')}`",
        f"Model: `{summary.get('model')}`",
        f"Stem: `{summary.get('stem_model_class')}`",
        f"Q head: `{summary.get('q_network_class')}`",
        f"Run: {summary.get('honest_run_name')}",
        "",
        "## Plain-English Read",
        "",
        f"ConvNeXt status: **{summary.get('convnext_status')}**",
        "",
        summary.get("convnext_note", ""),
        "",
        f"RL status: **{summary.get('rl_status')}**",
        "",
        summary.get("rl_note", ""),
        "",
        "Important interpretation: finite non-collapsed ConvNeXt features mean the visual stem is technically working. "
        "They do not prove the policy is strong; use evaluation runs for that.",
        "",
        "## Key Numbers",
        "",
        f"- Activation rows: {fmt(summary.get('activation_rows'))}",
        f"- Episodes: {fmt(summary.get('episode_count'))}",
        f"- Wins: {fmt(summary.get('win_count'))}",
        f"- Win rate: {fmt(summary.get('win_rate'))}",
        f"- Average return: {fmt(summary.get('avg_episode_return'))}",
        f"- Average score: {fmt(summary.get('avg_score'))}",
        f"- Train level pool / coverage: {fmt(summary.get('train_level_pool'))} / "
        f"{fmt(summary.get('unique_levels_seen'))}/{fmt(summary.get('num_train_levels'))} "
        f"({fmt(summary.get('train_level_coverage'))})",
        f"- Learning updates: {fmt(summary.get('learning_updates'))}",
        f"- Loss first/last/min/max: {fmt(summary.get('loss_first'))} / {fmt(summary.get('loss_last'))} / "
        f"{fmt(summary.get('loss_min'))} / {fmt(summary.get('loss_max'))}",
        f"- Last epsilon: {fmt(summary.get('last_epsilon'))}",
        f"- Last action/angle/tap entropy: {fmt(summary.get('last_action_entropy'))} / "
        f"{fmt(summary.get('last_angle_entropy'))} / {fmt(summary.get('last_tap_entropy'))}",
        f"- ConvNeXt image feature std min/mean: {fmt(summary.get('convnext_image_feature_std_min'))} / "
        f"{fmt(summary.get('convnext_image_feature_std_mean'))}",
        f"- ConvNeXt fine-tune status: {fmt(summary.get('convnext_finetune_status'))}; "
        f"backbone grad norm last/mean: {fmt(summary.get('gradient_norm_convnext_backbone_last'))} / "
        f"{fmt(summary.get('gradient_norm_convnext_backbone_mean'))}",
        f"- QR status: {fmt(summary.get('qr_status'))}; quantiles: {fmt(summary.get('qr_num_quantiles'))}; "
        f"std mean: {fmt(summary.get('qr_current_quantile_std_mean'))}; "
        f"p90-p10 spread mean: {fmt(summary.get('qr_current_quantile_spread_p90_p10_mean'))}",
        f"- Auxiliary heads: {fmt(summary.get('auxiliary_status'))}; aux loss last/mean: "
        f"{fmt(summary.get('auxiliary_loss_last'))} / {fmt(summary.get('auxiliary_loss_mean'))}; "
        f"weighted aux last/mean: {fmt(summary.get('weighted_auxiliary_loss_last'))} / "
        f"{fmt(summary.get('weighted_auxiliary_loss_mean'))}",
        f"- Proxy reward status: {fmt(summary.get('proxy_reward_status'))}; positive rate: "
        f"{fmt(summary.get('proxy_bonus_positive_rate'))}; avg proxy bonus: {fmt(summary.get('avg_proxy_bonus'))}",
        f"- Proxy counts tap-score/tap-win/pig/best-score: "
        f"{fmt(summary.get('tap_score_bonus_count'))} / {fmt(summary.get('tap_win_bonus_count'))} / "
        f"{fmt(summary.get('pig_proxy_bonus_count'))} / {fmt(summary.get('best_score_bonus_count'))}",
        f"- ConvNeXt NaN/Inf total: {fmt(summary.get('convnext_feature_nan_total'))} / "
        f"{fmt(summary.get('convnext_feature_inf_total'))}",
        f"- Q output NaN/Inf total: {fmt(summary.get('q_output_nan_total'))} / "
        f"{fmt(summary.get('q_output_inf_total'))}",
        "",
        "## Added Components Compared With Model D",
        "",
        f"- QR-DQN head: {fmt(summary.get('qr_status'))}",
        f"- Low epsilon plus NoisyNet exploration: last epsilon {fmt(summary.get('last_epsilon'))}",
        f"- Scheduled ConvNeXt fine-tuning: {fmt(summary.get('convnext_finetune_status'))}",
        f"- Proxy reward shaping: {fmt(summary.get('proxy_reward_status'))}",
        f"- All-map training pool: {fmt(summary.get('level_coverage_status'))}",
        f"- Action/tap usage: action entropy {fmt(summary.get('last_action_entropy'))}, "
        f"tap entropy {fmt(summary.get('last_tap_entropy'))}",
        "",
        "## Files",
        "",
    ]
    for label, path in artifacts.items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_training_diagnostics_report(
    diagnostics_path: str | Path,
    output_dir: str | Path | None = None,
    checkpoint_label: str | None = None,
) -> dict[str, Any]:
    """Create CSVs, plots, and summaries from a diagnostics.jsonl file."""

    diagnostics_path = Path(diagnostics_path)
    if output_dir is None:
        output_dir = diagnostics_path.parent / "diagnostic_reports" / "latest"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt, pd = _load_plotting()
    records = _read_jsonl(diagnostics_path)
    activation_df = pd.DataFrame(_activation_rows(records))
    episode_df = pd.DataFrame(_episode_rows(records))
    learning_df = pd.DataFrame(_learning_rows(records))
    reward_action_columns = [
        column for column in activation_df.columns
        if column.startswith("env_")
        or column in {
            "model", "wall_time", "loop_step", "transition", "action", "action_text",
            "reward", "score", "terminal", "win", "game_over", "action_total",
            "action_unique", "action_entropy", "angle_entropy", "tap_entropy",
            "top_actions", "top_action_counts", "top_angle_bins", "top_tap_bins",
        }
    ] if not activation_df.empty else []
    reward_action_df = activation_df[reward_action_columns].copy() if reward_action_columns else pd.DataFrame()
    if not episode_df.empty and "level" in episode_df:
        level_summary_df = (
            episode_df.assign(win_numeric=episode_df["win"].astype(float))
            .groupby("level", dropna=True)
            .agg(
                episodes=("level", "size"),
                wins=("win_numeric", "sum"),
                win_rate=("win_numeric", "mean"),
                avg_score=("score", "mean"),
                max_score=("score", "max"),
                avg_return=("episode_return", "mean"),
                avg_shots=("shots", "mean"),
            )
            .reset_index()
            .sort_values(["win_rate", "episodes"], ascending=[True, False])
        )
    else:
        level_summary_df = pd.DataFrame()

    artifacts = {
        "convnext_rl_step_csv": "convnext_rl_step_diagnostics.csv",
        "episode_csv": "episode_diagnostics.csv",
        "learning_csv": "learning_diagnostics.csv",
        "reward_action_csv": "reward_action_diagnostics.csv",
        "level_summary_csv": "level_summary_diagnostics.csv",
        "activation_plot": "convnext_activation_health.png",
        "rl_plot": "rainbow_rl_training_health.png",
        "summary_json": "summary.json",
        "summary_markdown": "README.md",
    }

    activation_df.to_csv(output_dir / artifacts["convnext_rl_step_csv"], index=False)
    episode_df.to_csv(output_dir / artifacts["episode_csv"], index=False)
    learning_df.to_csv(output_dir / artifacts["learning_csv"], index=False)
    reward_action_df.to_csv(output_dir / artifacts["reward_action_csv"], index=False)
    level_summary_df.to_csv(output_dir / artifacts["level_summary_csv"], index=False)

    _plot_activation_health(plt, activation_df, output_dir / artifacts["activation_plot"])
    _plot_rl_health(plt, pd, activation_df, episode_df, learning_df, output_dir / artifacts["rl_plot"])

    summary = _build_summary(pd, records, activation_df, episode_df, learning_df, checkpoint_label)
    summary["diagnostics_path"] = str(diagnostics_path)
    summary["output_dir"] = str(output_dir)
    summary["artifacts"] = artifacts

    (output_dir / artifacts["summary_json"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown_summary(output_dir / artifacts["summary_markdown"], diagnostics_path, summary, artifacts)
    return summary


def find_latest_diagnostics(root: str | Path = "out/angry_birds") -> Path:
    root = Path(root)
    candidates = list(root.glob("*/diagnostics.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No diagnostics.jsonl files found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", nargs="?", help="Path to diagnostics.jsonl. Defaults to latest run.")
    parser.add_argument("--output-dir", help="Directory for CSV/PNG/summary outputs.")
    parser.add_argument("--checkpoint-label", default="manual", help="Label written into summary files.")
    args = parser.parse_args(argv)

    diagnostics_path = Path(args.diagnostics) if args.diagnostics else find_latest_diagnostics()
    summary = export_training_diagnostics_report(
        diagnostics_path=diagnostics_path,
        output_dir=args.output_dir,
        checkpoint_label=args.checkpoint_label,
    )
    print(json.dumps({
        "output_dir": summary["output_dir"],
        "convnext_status": summary["convnext_status"],
        "rl_status": summary["rl_status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
