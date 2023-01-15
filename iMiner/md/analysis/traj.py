"""
Author: Oufan Zhang, Eric Wang
Date: 01/14/2023

Codes for manipulating MD trajectories and RMSD-related
"""
from typing import Optional, Union, Tuple
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

from iMiner.cmd import run_command, find_executable


def gmx_genidx(input_file: os.PathLike, output_file: Optional[os.PathLike] = None):
    gmx = find_executable(["gmx_mpi", "gmx"])
    cmds = [gmx, 'make_ndx', '-f', input_file]
    if output_file:
        cmds += ['-o', output_file]
    run_command(cmds)
        
    
def gmx_extract_and_align_traj(
    ref_file: os.PathLike,
    traj_file: os.PathLike,
    index_file: Optional[os.PathLike] = None,
    output_file: Optional[os.PathLike] = None,
    center_grp: Union[str, int] = "MOL",
    align_grp: Optional[Union[str, int]] = None,
    output_grp: Union[str, int] = "MOL_Protein",
    gen_short_dt: int = 0,
):
    """
    Remove water/ions and align GROMACS traj file
    
    Parameters
    ----------
    ref_file: os.PathLike
        Reference file, '-s' option in `gmx trjconv`
    traj_file: os.PathLike
        Trajectory file, '-f' option in `gmx trjconv`
    index_file: os.PathLike
        Index file, '-n' option in `gmx_trjconv`. Default it None.
    center_grp: str or int
        Group name or index for centering
    align_grp: str or int
        Group name or index for aligning. If None, the align step is skipped.
    output_grp: str or int
        Group name or index for output
    gen_short_dt: int
        Generate short md traj with large time intervals, in ps
    """
    gmx = find_executable(["gmx_mpi", "gmx"])
    traj_file = Path(traj_file)
    tmpout = traj_file.with_(f'{traj_file.stem}_pbc{traj_file.suffix}')
    
    base_cmds = [gmx, "trjconv", "-s", ref_file, "-f", traj_file]
    if index_file is not None:
        base_cmds += ["-n", index_file]
     
    # Remove PBC and centering
    cmds = base_cmds.copy()
    if align_grp is not None:
        cmds += ["-o", tmpout]
    elif output_file is not None:
        cmds += ["-o", output_file]
    
    # remove pbc and centering
    run_command(
        cmds + ["-center -pbc mol"], 
        input=f"{center_grp}\n{output_grp}"
    )
    
    if align_grp is not None:
        cmds = base_cmds.copy()
        output_file = traj_file.with_name(f'{traj_file.stem}_align{traj_file.suffix}')
        cmds += ['-o', output_file, '-fit', 'rot+trans']
        run_command(cmds, input=f"{output_grp}\n{output_grp}")
        
        cmds = base_cmds.copy()
        cmds += ['-o', output_file.with_suffix('.gro'), '-dump', 0] 
        run_command(cmds, input=str(output_grp))
    
    if gen_short_dt:
        output_file = traj_file.with_name(f'{traj_file.stem}_align{gen_short_dt // 1000}ns{traj_file.suffix}')
        cmds = base_cmds.copy()
        cmds += ['-o', output_file, '-dt', gen_short_dt]
        run_command(cmds, input=str(output_grp))

        
def gmx_rms(
    ref_file: os.PathLike,
    traj_file: os.PathLike,
    output_file: os.PathLike,
    index_file: Optional[os.PathLike] = None,
    align_grp: Union[str, int] = "Protein-H",
    output_grp: Union[str, int] = "MOL"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run `gmx rms` command
    """
    gmx = find_executable(['gmx_mpi', 'gmx'])
    cmds = [gmx, 'rms', '-s', ref_file, '-f', traj_file, '-o', output_file]
    if index_file:
        cmds += ['-n', index_file]
    run_command(cmds, input=f"{align_grp}\n{output_grp}")
    

def xtc_to_pdb(ref, traj, trajdir, dt):
    """
    Convert xtc to pdbs
    """
    run_command(
        f"gmx trjconv -s {Path(ref).resolve()} -f {Path(traj).resolve()} -o {Path(trajdir).resolve() / 'prod.pdb'} -sep -dt {dt}",
        raise_error=True,
        input="0"
    )
    pdbs = []
    for pdb in trajdir.glob('*.pdb'):
        with open(pdb, 'r') as f:
            contents = f.readlines()
        for i, line in enumerate(contents):
            if line.startswith("MODEL"):
                contents[i] = "MODEL        1\n"
        with open(pdb, 'w') as f:
            f.write("".join(contents))
        pdbs.append(str(pdb))
    return pdbs


def gmx_rmsdist(
    ref_file: os.PathLike,
    traj_file: os.PathLike,
    index_file: Optional[os.PathLike] = None,
    output: Optional[os.PathLike] = None,
    **kwargs
):
    """
    Run `gmx rmsdist` command
    
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
    """
    Read GROMACS ouput .xvg file
    """
    tu = 1.
    du = 1.
    if tunit == "ns":
        tu = 0.0001
    if dunit == "A":
        du = 10.
    data = np.loadtxt(xvg_file, comments=["#", "@"])
    tlist = data[:, 0] * tu
    rmslist = data[:, 1] * du
    return tlist, rmslist


def plot_rmsd(
    tlist: np.ndarray, 
    rmslist: np.ndarray, 
    name: Optional[str] = None,
    tunit: str = "ns",
    dunit: str = "A"
):
    """
    Plot RMSD
    """
    if dunit in ["A", "Angstrom", "angstrom"]:
        dunit = "\mathring{A}"
    dunit = "$\mathrm{" + dunit + "}$"
        
    avg_rmsd = np.mean(rmslist)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8,5))
    ax.set_xlabel(f"Time ({tunit})")
    ax.set_ylabel(f"RMSD ({dunit})")
    ax.set_xlim(tlist.min(), tlist.max())
    if name is None:
        ax.set_title(f"Average RMSD: {avg_rmsd:.2f}" + dunit)
    else:
        ax.set_title(f"{name}  Average RMSD: {avg_rmsd:.2f}" + dunit)
    ax.set_ylim(0, np.max(rmslist) * 1.1)
    ax.legend()
    return fig, ax