from typing import Optional
import os

from iMiner.cmd import run_command, find_executable


def gmx_extract_and_align_traj(
    ref_file: os.PathLike,
    traj_file: os.PathLike,
    index_file: Optional[os.PathLike] = None,
    center_grp_name: str = "MOL",
    align_grp_name: str = "MOL_Protein",
    output_grp_name: str = "MOL_Protein",
):
    gmx = find_executable(["gmx_mpi", "gmx"])
    cmds = [gmx, "trjconv", "-s", ref_file, "-f", traj_file]
    if index_file is not None:
        cmds += ["-n", index_file]
    
    # remove pbc and centering
    run_command(
        cmds + ['']
    )