from .base import Vanilla
from .KD import KD
from .DKD import DKD
distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "DKD": DKD
}