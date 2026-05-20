"""
커스텀 RL 알고리즘 SavedModel 내보내기 템플릿 (Tier 3)
========================================================

DQN이 아닌 알고리즘(PPO, A3C, SAC 등)을 직접 구현한 학생용 내보내기 가이드입니다.

필수 조건
---------
모든 모델은 아래 입출력 인터페이스를 반드시 준수해야 합니다:

  입력:
    - image : (batch, 128, 128, 3)  float32  — 게임 화면 스크린샷
    - bird  : (batch, 5)            float32  — 현재 새 종류 one-hot
                                               [BirdRed, BirdBlue, BirdYellow,
                                                BirdBlack, BirdWhite]

  출력:
    - logits/q-values : (batch, 200)  float32
      * 200 = ANGLE_RESOLUTION(20) × TAP_TIME_RESOLUTION(10)
      * 평가 시 argmax(output) → action index 로 사용
      * DQN: Q-values,  PPO/A3C: action logits (or probabilities) 모두 가능

사용 방법
---------
1. 아래 export_custom_model() 함수에 본인의 모델을 전달하세요.
2. 학습이 끝난 후, 평가 전에 한 번만 호출하면 됩니다.
3. 생성된 saved_model/ 폴더를 포함해 모델 폴더 전체를 제출하세요.

예시 (PPO actor):

    from src.agent.model.export_template import export_custom_model

    # 학습 완료 후
    model_dir = "out/angry_birds/my_ppo_agent/"
    export_custom_model(actor_model, model_dir)
    # → out/angry_birds/my_ppo_agent/saved_model/ 생성됨
"""

import os
import tensorflow as tf


# ──────────────────────────────────────────────────────────────────────────────
# 고정 상수 (수정 금지)
# ──────────────────────────────────────────────────────────────────────────────
_IMAGE_H  = 128
_IMAGE_W  = 128
_IMAGE_C  = 3
_BIRD_DIM = 5
_N_ACTIONS = 200


def export_custom_model(model: tf.keras.Model, model_dir: str):
    """
    커스텀 RL 모델을 TF SavedModel 형식으로 내보냅니다.

    Args:
        model     : Keras 모델.
                    반드시 inputs=[image(B,128,128,3), bird(B,5)],
                    output=logits(B,200) 을 갖는 구조여야 합니다.
        model_dir : 출력 디렉터리 (예: "out/angry_birds/my_ppo_agent/")
    """
    out_path = os.path.join(model_dir, "saved_model")

    @tf.function(input_signature=[
        tf.TensorSpec([None, _IMAGE_H, _IMAGE_W, _IMAGE_C], tf.float32, name="image"),
        tf.TensorSpec([None, _BIRD_DIM],                   tf.float32, name="bird"),
    ])
    def serving_fn(image, bird):
        return model([image, bird], training=False)

    tf.saved_model.save(model, out_path, signatures={"serving_default": serving_fn})
    print(f"SavedModel 내보내기 완료: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 동작 확인용 예시
# ──────────────────────────────────────────────────────────────────────────────

def _build_example_ppo_actor() -> tf.keras.Model:
    """
    PPO actor 예시 모델.
    image + bird_onehot → action logits (200-dim)
    실제 학습 시에는 이 구조를 참고해 본인 모델을 설계하세요.
    """
    image_input = tf.keras.Input(shape=(_IMAGE_H, _IMAGE_W, _IMAGE_C), name="image")
    bird_input  = tf.keras.Input(shape=(_BIRD_DIM,),                   name="bird")

    # ── CNN feature extractor ──
    x = tf.keras.layers.Conv2D(32, 4, strides=2, padding="same", activation="relu")(image_input)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    image_feat = tf.keras.layers.Dense(256, activation="relu")(x)

    # ── bird one-hot concat ──
    latent = tf.keras.layers.Concatenate()([image_feat, bird_input])

    # ── actor head: action logits (200-dim) ──
    logits = tf.keras.layers.Dense(256, activation="relu")(latent)
    logits = tf.keras.layers.Dense(_N_ACTIONS, name="action_logits")(logits)

    return tf.keras.Model(inputs=[image_input, bird_input], outputs=logits,
                          name="ppo_actor_example")


if __name__ == "__main__":
    import numpy as np

    # 예시: PPO actor를 빌드하고 내보내기
    actor = _build_example_ppo_actor()
    actor.summary()

    out_dir = "out/angry_birds/example_ppo/"
    os.makedirs(out_dir, exist_ok=True)
    export_custom_model(actor, out_dir)

    # ── 내보내기 검증 ──
    loaded = tf.saved_model.load(out_dir + "saved_model")
    predict_fn = loaded.signatures["serving_default"]

    dummy_image = np.zeros((1, 128, 128, 3), dtype=np.float32)
    dummy_bird  = np.array([[1, 0, 0, 0, 0]], dtype=np.float32)
    output = predict_fn(image=tf.constant(dummy_image),
                        bird=tf.constant(dummy_bird))
    logits = list(output.values())[0]
    action = int(tf.argmax(logits[0]).numpy())
    print(f"\n검증 완료: 출력 shape={logits.shape}, 선택된 action={action}")
