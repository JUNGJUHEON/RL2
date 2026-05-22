from src.agent.model.q_network import (
    QNetwork,
    DoubleQNetwork,
    VanillaQNetwork,
    DistributionalDuelingQNetwork,
    QuantileDuelingQNetwork,
)
from src.agent.model.stem import StemNetwork
from src.agent.model.generic import StemNetwork2D1D, StemNetwork2DLarge, StemNetwork2D, ConvLSTM,\
    Rainbow, RainbowImproved, StemNetwork2DSmallNoDense
from src.agent.model.noisy import NoisyDense
from src.agent.model.angry_birds import ClassicConv, PretrainedConvNeXtTiny


def get_class_from_name(name: str):
    return eval(name)
