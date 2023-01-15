import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iMiner.md.gbsa import GBSA


if __name__ == "__main__":
    gbsa = GBSA("gbsa", use_mpi=False)
    gbsa.set_params(
        complex_file="prod.tpr",
        traj_file="prod.xtc",
        topol_file="processed.top",
        index_file="index.ndx",
        mmpbsa_params={"modes": "gb", "indi": 4.0, "exdi": 80.0}
    )
    gbsa.run()