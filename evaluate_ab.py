"""
모델 평가 스크립트
================================
  - TensorFlow SavedModel을 로드해 소스 코드 없이 모든 알고리즘(DQN/PPO/A3C 등) 평가
결과 저장:
  - out/angry_birds/<모델명>/evaluation_results.csv
  - out/angry_birds/<모델명>/evaluation_summary.txt
  - out/angry_birds/<모델명>/score_distribution.png
"""

import os
import csv
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.utils import setup_hardware
from src.envs.angry_birds import (
    ALL_400_TRAIN_LEVELS,
    AngryBirds,
    FILTERED_TRAIN_LEVELS,
    STATE_PIXEL_RES,
    BIRD_DIM,
    action_to_params,
)

# ─────────────────────────────────────────────────────────────
# 평가 설정
# ─────────────────────────────────────────────────────────────

# 평가할 모델 이름 (out/angry_birds/<모델명>/ 에 저장된 것)
# 터미널에서 AB_MODEL_NAME=... 로 덮어쓸 수 있습니다.
MODEL_NAME = os.environ.get(
    "AB_MODEL_NAME",
    "ab_agent_20260516_220759_rainbow_D_full_n3_noisy",
)

# ── 평가 레벨 ─────────────────────────────────────────────────
COMPETITION_LEVELS = None   # None이면 아래 NUM_EVAL_LEVELS개를 랜덤 샘플
                            # 맵 번호 지정 시: [12, 34, 67, 88, 103] 등으로 지정

# COMPETITION_LEVELS가 None일 때 사용할 랜덤 평가 레벨 수
NUM_EVAL_LEVELS = int(os.environ.get("AB_NUM_EVAL_LEVELS", "20"))
EVAL_LEVEL_POOL = os.environ.get(
    "AB_EVAL_LEVEL_POOL",
    os.environ.get("AB_TRAIN_LEVEL_POOL", "all400"),
).lower()

# 평가 시 게임 속도 (1=원래 속도, 100=빠른 속도)
SIM_SPEED = int(os.environ.get("AB_SIM_SPEED", "3"))

# ─────────────────────────────────────────────────────────────
# 시작
# ─────────────────────────────────────────────────────────────
setup_hardware(use_gpu=False)

# ─────────────────────────────────────────────────────────────
# SavedModel 로딩
# ─────────────────────────────────────────────────────────────

def load_saved_model(model_name: str):
    model_dir = f"out/angry_birds/{model_name}/"
    saved_model_path = model_dir + "saved_model"

    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(
            f"\n\n[오류] SavedModel을 찾을 수 없습니다.\n"
            f"경로: {saved_model_path}\n\n"
        )

    print(f"  SavedModel 로드 중: {saved_model_path}")
    loaded = tf.saved_model.load(saved_model_path)
    predict_fn = loaded.signatures["serving_default"]
    print("  모델 로드 완료!")
    return predict_fn


def get_action(predict_fn, image_arr: np.ndarray, bird_arr: np.ndarray) -> int:
    """
    SavedModel로 action index를 결정합니다.

    Args:
        predict_fn: load_saved_model()이 반환한 호출 가능 객체
        image_arr : (128, 128, 3) float32 ndarray
        bird_arr  : (5,)          float32 ndarray
    Returns:
        action index (int, 0~199)
    """
    image_t = tf.expand_dims(tf.cast(image_arr, tf.float32), 0)   # (1, 128, 128, 3)
    bird_t  = tf.expand_dims(tf.cast(bird_arr,  tf.float32), 0)   # (1, 5)
    output  = predict_fn(image=image_t, bird=bird_t)
    logits  = list(output.values())[0]                             # (1, 200)
    return int(tf.argmax(logits[0]).numpy())


# ─────────────────────────────────────────────────────────────
# 평가 루프
# ─────────────────────────────────────────────────────────────

def evaluate_model(model_name: str,
                   eval_levels=None,
                   num_eval_levels: int = 20,
                   sim_speed: int = 50):
    """
    Returns:
        results: list of dict, 각 레벨의 평가 결과
        summary: dict, 전체 요약 통계
    """
    print("=" * 60)
    print(f"  모델 평가 시작: {model_name}")
    print("=" * 60)

    # ── SavedModel 로드 ──
    print(f"\n[1/4] SavedModel 로드 중: {model_name}")
    predict_fn = load_saved_model(model_name)

    # ── 환경 초기화 ──
    print(f"\n[2/4] AngryBirds 환경 초기화 중...")
    env = AngryBirds(num_par_inst=1)
    env.set_sim_speed(sim_speed)
    print(f"      시뮬레이션 속도: {sim_speed}x")

    # ── 평가 레벨 목록 결정 ──
    if eval_levels is None:
        candidate_levels = (
            ALL_400_TRAIN_LEVELS
            if EVAL_LEVEL_POOL in {"all400", "all_400", "all"}
            else FILTERED_TRAIN_LEVELS
        )
        np.random.seed(42)
        eval_levels = sorted(np.random.choice(
            candidate_levels,
            size=min(num_eval_levels, len(candidate_levels)),
            replace=False
        ).tolist())
    else:
        eval_levels = list(eval_levels)

    print(f"\n[3/4] 평가 레벨: {len(eval_levels)}개")
    print(f"      레벨 목록: {eval_levels[:10]}{'...' if len(eval_levels) > 10 else ''}")

    # ── 레벨 평가 ──
    results = []

    print(f"\n[4/4] 레벨 평가 시작...\n")
    print(f"{'레벨':>6} | {'점수':>8} | {'결과':>9} | {'발사 수':>6} | {'소요 시간':>8}")
    print("-" * 55)

    for level in eval_levels:
        level_result = evaluate_single_level(
            predict_fn=predict_fn,
            env=env,
            level=level,
        )
        results.append(level_result)

        status_str = "Pass" if level_result["passed"] else "Fail"
        print(f"{level:>6} | {level_result['score']:>8,} | {status_str:>9} | "
              f"{level_result['num_shots']:>6} | {level_result['elapsed_sec']:>6.1f}s")

    env._cleanup()

    # ── 요약 통계 계산 ──
    summary = compute_summary(results)

    return results, summary


def evaluate_single_level(predict_fn, env: AngryBirds, level: int) -> dict:
    """
    Returns:
        dict: level, score, passed, num_shots, elapsed_sec
    """
    env.scores[:] = 0
    env.times[:] = 0
    env.game_overs[:] = False
    env.wins[:] = False
    env.load_specified_level(level)

    start_time = time.time()
    total_score = 0
    num_shots = 0
    game_over = False
    level_won = False

    while not game_over:
        # 현재 화면 상태 획득: [image(1,128,128,3), bird(1,5)]
        states = env.get_states()
        image_arr = states[0][0]   # (128, 128, 3)
        bird_arr  = states[1][0]   # (5,)

        # SavedModel로 action 결정 (greedy)
        action = get_action(predict_fn, image_arr, bird_arr)
        alpha, tap_ms = action_to_params(action)
        print(f"    shot#{num_shots+1}  action={action:>3}  alpha={alpha:>3}°  tap={tap_ms}ms")

        # 발사 실행
        reward, score, terminals, times, wins, game_overs = env.step([action])

        total_score = int(score[0])
        level_won = bool(wins[0])
        num_shots += 1
        game_over = bool(game_overs[0])

        # 무한루프 방지 (최대 20발)
        if num_shots >= 20:
            break

    elapsed = time.time() - start_time

    return {
        "level": level,
        "score": total_score,
        "passed": level_won,
        "num_shots": num_shots,
        "elapsed_sec": elapsed,
    }


def compute_summary(results: list) -> dict:
    """평가 결과 계산"""
    scores = [r["score"] for r in results]
    passed = [r["passed"] for r in results]
    shots  = [r["num_shots"] for r in results]

    return {
        "total_levels":          len(results),
        "passed_levels":         sum(passed),
        "failed_levels":         len(results) - sum(passed),
        "win_rate":              sum(passed) / len(results) * 100,
        "avg_score":             float(np.mean(scores)),
        "max_score":             int(np.max(scores)),
        "min_score":             int(np.min(scores)),
        "std_score":             float(np.std(scores)),
        "avg_shots":             float(np.mean(shots)),
    }


# ─────────────────────────────────────────────────────────────
# 결과 출력 / 저장
# ─────────────────────────────────────────────────────────────

def print_summary(summary: dict, model_name: str):
    print("\n" + "=" * 60)
    print(f"  평가 결과 요약: {model_name}")
    print("=" * 60)
    print(f"  평가 레벨 수       : {summary['total_levels']}개")
    print(f"  클리어 성공        : {summary['passed_levels']}개")
    print(f"  클리어 실패        : {summary['failed_levels']}개")
    print(f"  클리어율(Win Rate) : {summary['win_rate']:.1f}%")
    print(f"  평균 점수          : {summary['avg_score']:,.0f}점")
    print(f"  최고 점수          : {summary['max_score']:,}점")
    print(f"  최저 점수          : {summary['min_score']:,}점")
    print(f"  점수 표준편차      : {summary['std_score']:,.0f}점")
    print(f"  평균 발사 수       : {summary['avg_shots']:.1f}발/레벨")
    print("=" * 60)


def save_results_to_csv(results: list, model_name: str):
    """
    평가 결과를 CSV로 저장
    """
    out_dir = f"out/angry_birds/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = out_dir + "evaluation_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EvaluationIndex", "LevelIndex", "Score", "LevelStatus", "NumShots"])
        for i, r in enumerate(results, 1):
            status = "Pass" if r["passed"] else "Fail"
            writer.writerow([i, r["level"], r["score"], status, r["num_shots"]])

    print(f"\n  CSV 저장: {csv_path}")
    return csv_path


def save_summary_to_txt(summary: dict, model_name: str):
    """요약 통계를 텍스트 파일로 저장"""
    out_dir = f"out/angry_birds/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)
    txt_path = out_dir + "evaluation_summary.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"모델: {model_name}\n")
        f.write("=" * 40 + "\n")
        for key, val in summary.items():
            if isinstance(val, float):
                f.write(f"{key}: {val:.2f}\n")
            else:
                f.write(f"{key}: {val}\n")

    print(f"  요약 저장: {txt_path}")
    return txt_path


def plot_score_distribution(results: list, summary: dict, model_name: str):
    """레벨별 점수 분포를 막대 그래프로 시각화합니다."""
    out_dir = f"out/angry_birds/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)

    levels = [r["level"] for r in results]
    scores = [r["score"] for r in results]
    colors = ["#2ecc71" if r["passed"] else "#e74c3c" for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"AngryBirds 평가 결과 - {model_name}", fontsize=14, fontweight="bold")

    # 그래프 1: 레벨별 점수
    ax1 = axes[0]
    ax1.bar(range(len(levels)), scores, color=colors, alpha=0.85)
    ax1.axhline(y=summary["avg_score"], color="navy", linestyle="--", linewidth=1.5,
                label=f"평균: {summary['avg_score']:,.0f}점")
    ax1.set_xlabel("평가 순서")
    ax1.set_ylabel("점수")
    ax1.set_title(f"레벨별 점수 (Win Rate: {summary['win_rate']:.1f}%  |  "
                  f"평균 점수: {summary['avg_score']:,.0f}점  |  "
                  f"평균 발사: {summary['avg_shots']:.1f}발)")
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels([str(lv) for lv in levels], rotation=45, fontsize=8)
    pass_patch = mpatches.Patch(color="#2ecc71", label=f"Pass ({summary['passed_levels']}개)")
    fail_patch = mpatches.Patch(color="#e74c3c", label=f"Fail ({summary['failed_levels']}개)")
    ax1.legend(handles=[pass_patch, fail_patch, ax1.lines[0]], loc="upper right")

    # 그래프 2: 점수 분포 히스토그램
    ax2 = axes[1]
    ax2.hist(scores, bins=min(15, len(scores)), color="#3498db", edgecolor="white", alpha=0.8)
    ax2.axvline(x=summary["avg_score"], color="navy", linestyle="--", linewidth=1.5,
                label=f"평균: {summary['avg_score']:,.0f}점")
    ax2.set_xlabel("점수")
    ax2.set_ylabel("레벨 수")
    ax2.set_title("점수 분포 히스토그램")
    ax2.legend()

    plt.tight_layout()
    plot_path = out_dir + "score_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  그래프 저장: {plot_path}")


# ─────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, summary = evaluate_model(
        model_name=MODEL_NAME,
        eval_levels=COMPETITION_LEVELS,
        num_eval_levels=NUM_EVAL_LEVELS,
        sim_speed=SIM_SPEED,
    )

    print_summary(summary, MODEL_NAME)
    save_results_to_csv(results, MODEL_NAME)
    save_summary_to_txt(summary, MODEL_NAME)
    plot_score_distribution(results, summary, MODEL_NAME)

    print("\n평가 완료!")
