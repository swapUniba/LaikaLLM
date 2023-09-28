import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    """
    Function which fixes the random state of each library used by this repository with the seed
    specified when invoking `pipeline.py`

    Returns:
        The integer random state set via command line argument

    """

    # seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    print(f"Random seed set as {seed}")

    return seed
