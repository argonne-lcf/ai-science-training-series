# Test imports for the AI4S Sim+AI examples on ALCF Polaris
import sys

try:
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from tqdm.notebook import tqdm

    # Parsl
    import parsl

    # DragonHPC
    import dragon

    # Example dependencies
    import ase
    import rdkit
    from xtb.ase.calculator import XTB

    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline

    import torch

    # Custom dependencies
    import chemfunctions

    print("All packages imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)