"""
Author: Eric Wang
Date Created: 10/24/2022

This package contains functions to run gromacs
"""
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import gromacs
gromacs.config.setup(Path(gromacs.__file__).parent / "templates/gromacswrapper.cfg")
from gromacs.fileformats.mdp import MDP
from iMiner.cmd import run_command, find_executable, set_directory
from iMiner.log import LOGGER


MAXWARN = 10


def run_preprocess_workflow(
    top: os.PathLike,
    gro: os.PathLike,
    wdir: os.PathLike = Path.cwd(),
    verbose: bool = False
):
    """
    Run preprocess workflow: add box with 1 nm buffer region, add water, add ions

    Parameters
    ----------
    top: os.PathLike
        topology file (-p)
    gro: os.PathLike
        coordinate file (-c)
    wdir: os.PathLike
        working directory where `gmx` is executed
    verbose: bool
        Whether print detailed information
    """
    gmx = find_executable(['gmx_mpi', 'gmx'])
    mdp = Path(__file__).with_name("em.mdp").resolve()
    with set_directory(wdir):
        if verbose: LOGGER.info("Adding box...")
        run_command([gmx, "editconf", "-f", gro, "-o", "newbox.gro", "-c", "-d", str(1.0), "-bt", "dodecahedron"])
        if verbose: LOGGER.info("Adding water...")
        run_command([gmx, "solvate", "-cp", "newbox.gro", "-cs", "spc216.gro", "-o", "solv.gro", "-p", top])
        if verbose: LOGGER.info("Add ions...")
        run_command([gmx, "grompp", "-f", mdp, "-c", "solv.gro", "-p", top, "-o", "ions.tpr", "-maxwarn", MAXWARN])
        run_command([gmx, "genion", "-s", "ions.tpr", "-o", "ions.gro", "-p", top, "-pname", "NA", "-nname", "CL", "-neutral"], input="SOL")
        run_command([gmx, "grompp", "-c", "ions.gro", "-f", mdp, "-p", top, "-pp", "processed.top"])


def run_md(
    top: os.PathLike,
    gro: os.PathLike,
    mdp: os.PathLike,
    deffnm: str,
    wdir: os.PathLike = Path.cwd(),
    cpt: Optional[os.PathLike] = None,
    restr_gro: Optional[os.PathLike] = None,
    restart: bool = True,
    params: Dict[str, Any] = dict(),
    enforce_gpu: bool = False,
):
    """
    Run molecular dynamics with GROMACS

    Parameters
    ----------
    top: os.PathLike
        topology file (-p)
    gro: os.PathLike
        coordinate file (-c)
    mdp: os.PathLike
        template parameter file (-f)
    deffnm: str
        default file name (-deffnm)
    wdir: os.PathLike
        working directory where `gmx` is executed
    cpt: os.PathLike, optional
        checkpoint file (-t)
    restr_gro: os.PathLike, optional
        restraint reference coordinate file (-r)
    restart: bool
        If restart from existing cpt file
    params: Dict[str, Any]
        parameters to update template mdp file
    enforce_gpu: bool
        whether to explicitly enforce "-update gpu -nb gpu -bonded gpu"
    """
    gmx = find_executable(["gmx_mpi", "gmx"])

    with set_directory(wdir):
        mdp_file = MDP(mdp)
        for key, value in params.items():
            mdp_file[key] = value
        # update mdp
        if os.path.isfile(mdp):
            shutil.copyfile(mdp, Path(mdp).with_name(f'{mdp.stem}.mdp.backup'))
        mdp_file.write(str(mdp))
        # run grompp
        grompp_cmds = [gmx, "grompp", '-c', str(gro), "-f", str(mdp)]
        if restr_gro is not None:
            grompp_cmds.append('-r')
            grompp_cmds.append(str(restr_gro))
        if cpt is not None:
            grompp_cmds.append('-t')
            grompp_cmds.append(str(cpt))
        grompp_cmds.append('-p')
        grompp_cmds.append(str(top))
        grompp_cmds.append('-o')
        grompp_cmds.append(f"{deffnm}.tpr")
        grompp_cmds.append('-maxwarn')
        grompp_cmds.append(str(MAXWARN))

        run_command(grompp_cmds, True)

        # run md
        mdrun_cmds = [gmx, 'mdrun']
        if restart and os.path.isfile(f"{deffnm}.cpt"):
            mdrun_cmds += ['-s', f'{deffnm}.tpr', '-cpi', f'{deffnm}.cpt']
        if enforce_gpu:
            mdrun_cmds += ['-update', 'gpu', '-nb', 'gpu', '-bonded', 'gpu']
        mdrun_cmds += ['-deffnm', deffnm]

        run_command(mdrun_cmds, True)


def run_md_workflow(
    top: os.PathLike,
    gro: os.PathLike,
    wdir: os.PathLike = Path.cwd(),
    restart: bool = True,
    params: Dict[str, Dict[str, Any]] = {"em": {}, "nvt": {}, "npt": {}, "prod": {}},
    enforce_gpu: bool = False,
    verbose: bool = False
):
    stages = ["em", "nvt", "npt", "prod"]
    with set_directory(wdir, mkdir=True) as w:
        for i, stage in enumerate(stages):
            Path.mkdir(w / stage, parents=True, exist_ok=True)
            shutil.copyfile(Path(__file__).with_name(f"{stage}.mdp"), w / stage / f"{stage}.mdp")
            if verbose: LOGGER.info(f"Running {stage}...")
            run_md(
                top = Path(top).resolve(),
                gro = Path(gro).resolve(),
                mdp = w / stage / f"{stage}.mdp",
                deffnm = stage,
                wdir = Path.resolve(w / stage),
                cpt = f"../{stages[i-1]}/{stages[i-1]}.cpt" if i >= 2 else None,
                restr_gro = f"../{stages[i-1]}/{stages[i-1]}.gro" if (i > 0 and i < 3) else None, 
                restart = bool(i) if restart else False, # em don't restart
                params = params[stage],
                enforce_gpu = bool(i) if enforce_gpu else False
            )
            gro = Path(w / stage / f"{stage}.gro")