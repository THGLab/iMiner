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
        cmds + ["-center -pbc mol"], 
        input=" ".join([center_grp_name, output_grp_name])
    )
    #
    
def gmx_rms(ref_file: os.PathLike,
    traj_file: os.PathLike,
    index_file: Optional[os.PathLike] = None,
    output: Optional[os.PathLike] = None,
    **kwargs
    ):
    """
    GROMACS command that computes the root mean square deviation of atom distances, 
    which has the advantage that no fit is needed
    
    =====
    output: xvgr/xmgr file
    """
    gmx = find_executable(["gmx_mpi", "gmx"])
    cmds = [gmx, "rmsdist", "-s", ref_file, "-f", traj_file]
    if index_file is not None:
        cmds += ["-n", index_file]
    if output is not None:
        cmds += ["-o", output]
    #for key in kwargs:
    #    cmds += [f"-{key}", kwargs[key]]
    # ligand group 2 MOL
    run_command(
        #" ".join(["echo 'MOL' |"] + cmds + ["-pbc"]), shell=True
        cmds + ["-pbc"], input="MOL"
    )
    
def read_xvg(xvg_file: os.PathLike, tunit: str = "ps", dunit: str = "nm"):
    tu = 1.
    du = 1.
    if tunit == "ns":
        tu = 0.0001
    if dunit == "A":
        du = 10.
    tlist = []
    rmslist = []
    with open(xvg_file, "r+") as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            elif line.startswith("@"):
                continue
            else:
                t, rms = line.strip().split()
                tlist.append(float(t)*tu)
                rmslist.append(float(rms)*du)
    return tlist, rmslist