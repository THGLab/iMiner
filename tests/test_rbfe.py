'''
This file is deprecated!
'''
from pathlib import Path
import pytest
from iMiner.md.rbfe import GromacsTopologyFilePerturb


@pytest.mark.parametrize(
    "sys1,sys2,mapping",
    [
        ("methane", "cholromethane", [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
        ("methane", "methanol", [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [-1, 5]]),
        ("methane", "ethane", [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [-1, 5], [-1, 6], [-1, 7]]),
        ("ethane", "methane", [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, -1], [6, -1], [7, -1]])
    ]
)
def test_pert(sys1, sys2, mapping):
    datadir = Path(__file__).parent / "data"
    output = Path(f"{sys1}~{sys2}.top")
    structA = GromacsTopologyFilePerturb(str(datadir / f"{sys1}.top"))
    structB = GromacsTopologyFilePerturb(str(datadir / f"{sys2}.top"))
    structA.perturb(structB, mapping)
    structA.write(output)
    output.unlink()
    