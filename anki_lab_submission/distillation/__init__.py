from .base import Vanilla
from .KD import KD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
}