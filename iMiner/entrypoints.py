from typing import Optional, Union
import argparse
from pathlib import Path
import os


def parse_args():    
    parser = argparse.ArgumentParser(description="iMiner: A platform to mine inhibitors")
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    gbsa_parser = subparsers.add_parser("gbsa")
    gbsa_parser.add_argument("-t", "--task_name", dest='task_name', default=None)
    gbsa_parser.add_argument("-w", "--working_dir", dest='wdir', default=None)
    gbsa_parser.add_argument("-m", "--use_mpi", dest='use_mpi', default=False)
    gbsa_parser.add_argument("-n", "--num_threads", dest='num_threads', default=1)

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


def gbsa(task_name: Optional[str] = None, wdir: Optional[os.PathLike] = None, use_mpi: bool = False, num_threads: int = 1):
    assert (task_name is not None) or (wdir is not None), "either task_name or working_dir must be specified"
    if wdir is not None:
        wdir = Path(wdir).resolve()
    else:
        wdir = Path.cwd() / "md" / task_name / "prod"
    
    from iMiner.cmd import set_directory
    from iMiner.md.gbsa import GBSA
    
    with set_directory(wdir):
        gbsa = GBSA(
            workdir="gbsa", 
            use_mpi=use_mpi,
            num_threads=num_threads
        )
        gbsa.set_params(
            complex_file="prod.tpr",
            traj_file="prod_align.xtc",
            topol_file="../processed.top",
            index_file="index.ndx",
            mmpbsa_params={"modes": "gb", "igb": 2, "indi": 4.0, "exdi": 80.0}
        )
        gbsa.run()


def main():
    args = parse_args()
    if args['command'] == "gbsa":
        gbsa(**args)

