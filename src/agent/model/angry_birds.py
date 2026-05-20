from src.agent.model.stem import StemNetwork
from tensorflow.keras.layers import (Input, Flatten, Dense, ReLU,
                                     Convolution2D, MaxPool2D, Concatenate,
                                     Dropout, Lambda)
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.applications import ConvNeXtTiny
import tensorflow as tf


def _convnext_pixel_input(x):
    """Accept both 0..1 RL-preprocessed images and 0..255 raw SavedModel images."""
    x = tf.cast(x, tf.float32)
    return tf.cond(tf.reduce_max(x) <= 1.5, lambda: x * 255.0, lambda: x)


class ClassicConv(StemNetwork):
    """
    AngryBirds 전용 CNN Stem Network.
    128x128 RGB 스크린샷 + 현재 새 종류(5-dim one-hot)를 입력받아
    잠재 벡터를 출력합니다.

    구조: Conv→ReLU→Pool (×4) → Flatten → Dense(latent_dim)
          + bird one-hot 5-dim 벡터를 Concatenate
          → 최종 latent 크기: latent_dim + 5
    """

    def __init__(self, latent_dim):
        super().__init__(sequential=False)
        self.latent_dim = latent_dim

    def get_functional_graph(self, input_shapes, batch_size=None):
        """
        Keras Functional API로 모델 그래프를 생성합니다.

        Args:
            input_shapes: [(128, 128, 3), (5,)]
                          input_shapes[0] = 이미지 shape
                          input_shapes[1] = bird one-hot shape
                          (이전 버전처럼 (0,)이면 이미지만 사용)
        Returns:
            inputs (list): Keras Input 레이어 목록
            latent: 최종 잠재 벡터 텐서
        """
        image_shape = input_shapes[0]   # (128, 128, 3)
        bird_dim    = input_shapes[1][0] if len(input_shapes) > 1 else 0  # 5 or 0

        input_frame = Input(shape=image_shape, name="image_input")

        # ─── Conv Block 1 ───────────────────────────────────
        x = Convolution2D(32, (4, 4), strides=1, padding='same',
                          kernel_initializer=GlorotNormal,
                          use_bias=False, name="conv_1")(input_frame)
        x = ReLU()(x)
        x = MaxPool2D((2, 2))(x)   # (64, 64, 32)

        # ─── Conv Block 2 ───────────────────────────────────
        x = Convolution2D(64, (3, 3), strides=2, padding='same',
                          kernel_initializer=GlorotNormal,
                          use_bias=False, name="conv_2")(x)
        x = ReLU()(x)
        x = MaxPool2D((2, 2))(x)   # (16, 16, 64)

        # ─── Conv Block 3 ───────────────────────────────────
        x = Convolution2D(64, (2, 2), strides=1, padding='same',
                          kernel_initializer=GlorotNormal,
                          use_bias=False, name="conv_3")(x)
        x = ReLU()(x)
        x = MaxPool2D((2, 2))(x)   # (8, 8, 64)

        # ─── Conv Block 4 ───────────────────────────────────
        x = Convolution2D(128, (2, 2), strides=1, padding='same',
                          kernel_initializer=GlorotNormal,
                          use_bias=False, name="conv_4")(x)
        x = ReLU()(x)
        x = MaxPool2D((2, 2))(x)   # (4, 4, 128)

        # ─── Flatten → Dense (이미지 잠재 벡터) ─────────────
        x = Flatten(name='flat')(x)
        image_latent = Dense(self.latent_dim, activation="relu",
                             name="image_latent")(x)

        # ─── 새 종류 one-hot Concatenate ────────────────────
        inputs = [input_frame]
        if bird_dim > 0:
            input_bird = Input(shape=(bird_dim,), name="bird_input")
            inputs.append(input_bird)
            latent = Concatenate(name="latent")([image_latent, input_bird])
        else:
            latent = image_latent

        return inputs, latent

    def get_config(self):
        return {"latent_dim": self.latent_dim}


class PretrainedConvNeXtTiny(StemNetwork):
    """
    AngryBirds stem using Keras/TensorFlow ConvNeXtTiny ImageNet weights.

    ORIGINAL ClassicConv is kept above.  This class is added as a separate
    training experiment so older checkpoints can still restore their original
    architecture.

    Fused latent:
      ConvNeXtTiny image feature + raw bird one-hot + learned bird embedding.
    """

    def __init__(self,
                 image_feature_dim=512,
                 bird_embed_dim=32,
                 weights="imagenet",
                 trainable_backbone=False,
                 dropout_rate=0.0,
                 pretrained_source=None):
        super().__init__(sequential=False)
        self.image_feature_dim = image_feature_dim
        self.bird_embed_dim = bird_embed_dim
        self.weights = weights
        self.pretrained_source = pretrained_source if pretrained_source is not None else weights
        self.trainable_backbone = trainable_backbone
        self.dropout_rate = dropout_rate

    def get_functional_graph(self, input_shapes, batch_size=None):
        image_shape = input_shapes[0]
        bird_dim = input_shapes[1][0] if len(input_shapes) > 1 else 0

        input_frame = Input(shape=image_shape, name="image_input")

        # Agent training sends images after env.preprocess() as 0..1 floats.
        # SavedModel/evaluate paths may pass raw 0..255 screenshots.  This
        # layer normalizes both cases to what ConvNeXt preprocessing expects.
        x = Lambda(_convnext_pixel_input, name="convnext_pixel_input")(input_frame)

        convnext = ConvNeXtTiny(
            include_top=False,
            include_preprocessing=True,
            weights=self.weights,
            input_shape=image_shape,
            pooling="avg",
        )
        convnext.trainable = self.trainable_backbone
        image_feature = convnext(x)

        if self.image_feature_dim is not None and self.image_feature_dim > 0:
            image_feature = Dense(self.image_feature_dim, activation="relu",
                                  name="convnext_image_feature")(image_feature)

        if self.dropout_rate > 0:
            image_feature = Dropout(self.dropout_rate, name="convnext_feature_dropout")(image_feature)

        inputs = [input_frame]
        if bird_dim > 0:
            input_bird = Input(shape=(bird_dim,), name="bird_input")
            inputs.append(input_bird)

            if self.bird_embed_dim is not None and self.bird_embed_dim > 0:
                bird_embedding = Dense(self.bird_embed_dim, activation="relu",
                                       name="bird_embedding")(input_bird)
                latent = Concatenate(name="latent")(
                    [image_feature, input_bird, bird_embedding]
                )
            else:
                latent = Concatenate(name="latent")([image_feature, input_bird])
        else:
            latent = image_feature

        return inputs, latent

    def get_config(self):
        return {
            "image_feature_dim": self.image_feature_dim,
            "bird_embed_dim": self.bird_embed_dim,
            # Checkpoint restore should not need internet or a Keras weight
            # download.  The checkpoint itself supplies all model weights.
            "weights": None,
            "pretrained_source": self.pretrained_source,
            "trainable_backbone": self.trainable_backbone,
            "dropout_rate": self.dropout_rate,
        }
