import numpy as np
import pytest
from pathlib import Path
from iMiner.md.common import read_single_gro


def test_read_gro():
    crds = read_single_gro(Path(__file__).parent / "data" / "test.gro")
    assert np.allclose(
        crds,
        [[5.490, 3.870, 7.505], [5.495, 3.918, 7.619]]
    )