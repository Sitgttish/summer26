"""ETAP — ESM3-based Transformer Attention Protein classifier for binary protein sequence classification."""

__version__ = '0.1.2'

from .model import ETA
from .train import run_training
from .eval import run_eval
