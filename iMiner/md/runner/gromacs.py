"""
Author: Eric Wang
Date Created: 10/24/2022

This package contains functions to run gromacs
"""
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
from logging import Logger

import gromacs
gromacs.config.setup(Path(gromacs.__file__).parent / "templates/gromacswrapper.cfg")
from gromacs.fileformats.mdp import MDP
from iMiner.utils import to_relpath
from iMiner.cmd import run_command, find_executable, set_directory

MAXWARN = 10


def run_preprocess_workflow(
    top: os.PathLike,
    gro: os.PathLike,
    wdir: os.PathLike = Path.cwd(),
    box_type: str = "dodecahedron",
    buffer: float = 1.0,
    verbose: bool = False,
    logger: Optional[Logger] = None,
):
    """
    Run preprocess workflow: add box with buffer region, add water, add ions

    Parameters
    ----------
    top: os.PathLike
        topology file (-p)
    gro: os.PathLike
        coordinate file (-c)
    wdir: os.PathLike
        working directory where `gmx` is executed
    box_type: str
        box type: triclinic, cubic, dodecahedron, octahedron. Default: dodecahedron
    buffer: float
        distance between the solute and the box. Default: 1.0
    verbose: bool
        whether print detailed information
    logger: logging.Logger
        logging.Logger object for logging output.
    """
    gmx = find_executable(['gmx_mpi', 'gmx'])
    mdp = Path(__file__).with_name("em.mdp").resolve()
    posre_mdp = Path(__file__).with_name("nvt.mdp").resolve()
    with set_directory(wdir):
        if logger and verbose: logger.info("Adding box...")
        run_command([gmx, "editconf", "-f", gro, "-o", "newbox.gro", "-c", "-d", str(buffer), "-bt", box_type])
        if logger and verbose: logger.info("Adding water...")
        run_command([gmx, "solvate", "-cp", "newbox.gro", "-cs", "spc216.gro", "-o", "solv.gro", "-p", top])
        if logger and verbose: logger.info("Add ions...")
        run_command([gmx, "grompp", "-f", mdp, "-c", "solv.gro", "-p", top, "-o", "ions.tpr", "-maxwarn", MAXWARN])
        run_command([gmx, "genion", "-s", "ions.tpr", "-o", "ions.gro", "-p", top, "-pname", "NA", "-nname", "CL", "-neutral"], input="SOL")
        run_command([gmx, "grompp", "-c", "ions.gro", "-f", mdp, "-p", top, "-pp", "processed.top", '-maxwarn', MAXWARN])
        run_command([gmx, "grompp", "-c", "ions.gro", "-f", posre_mdp, "-r", "ions.gro", "-p", top, "-pp", "processed_posre.top", '-maxwarn', MAXWARN])


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
    return_commands_only: bool = False
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
    return_commands_only: bool
        if this is set, return commands only instead of running them
    """
    gmx = find_executable(["gmx_mpi", "gmx"])
    if return_commands_only:
        gmx = gmx.split("/")[-1]

    with set_directory(wdir):
        mdp_file = MDP(mdp)
        for key, value in params.items():
            mdp_file[key] = value
        # update mdp
        if os.path.isfile(mdp):
            shutil.copyfile(mdp, Path(mdp).with_name(f'{mdp.stem}.mdp.backup'))
        mdp_file.write(str(mdp))

        mdp = to_relpath(mdp, wdir)
        top = to_relpath(top, wdir)
        gro = to_relpath(gro, wdir)

        # run grompp
        grompp_cmds = [gmx, "grompp", '-c', str(gro), "-f", str(mdp)]
        if restr_gro is not None:
            restr_gro = to_relpath(restr_gro, wdir)
            grompp_cmds.append('-r')
            grompp_cmds.append(str(restr_gro))
        if cpt is not None:
            cpt = to_relpath(cpt, wdir)
            grompp_cmds.append('-t')
            grompp_cmds.append(str(cpt))
        grompp_cmds.append('-p')
        grompp_cmds.append(str(top))
        grompp_cmds.append('-o')
        grompp_cmds.append(f"{deffnm}.tpr")
        grompp_cmds.append('-maxwarn')
        grompp_cmds.append(str(MAXWARN))

        if not return_commands_only:
            run_command(grompp_cmds, True)

        # run md
        if not return_commands_only:
            mdrun_cmds = [] 
            mdrun_cmds += [gmx, 'mdrun']
            if restart and os.path.isfile(f"{deffnm}.cpt"):
                mdrun_cmds += ['-s', f'{deffnm}.tpr', '-cpi', f'{deffnm}.cpt']
            if enforce_gpu:
                mdrun_cmds += ['-update', 'gpu', '-nb', 'gpu', '-bonded', 'gpu']
            mdrun_cmds += ['-deffnm', deffnm]
            run_command(mdrun_cmds, True)
        else:
            mdrun = [gmx, 'mdrun', '-deffnm', deffnm]
            mdrun_restart = [gmx, 'mdrun', '-s', f'{deffnm}.tpr', '-cpi', f"{deffnm}.cpt", '-deffnm', deffnm]
            if restart:
                mdrun_cmds = [f"if [ -f {deffnm}.cpt ]; then\n  "] + mdrun_restart + ['\nelse\n  '] + mdrun + ['\nfi']
            else:
                mdrun_cmds = mdrun
            
    if return_commands_only:
        return [' '.join(grompp_cmds), ' '.join(mdrun_cmds)]
    else:
        return [True]


def run_md_workflow(
    top: os.PathLike,
    gro: os.PathLike,
    wdir: os.PathLike = Path.cwd(),
    restart: bool = True,
    params: Dict[str, Dict[str, Any]] = {"em": {}, "nvt": {}, "npt": {}, "prod": {}},
    enforce_gpu: bool = False,
    verbose: bool = True,
    logger: Optional[Logger] = None,
    mdp_dir: Optional[os.PathLike] = None,
    top_posre: Optional[os.PathLike] = None,
    return_commands_only: bool = False,
) -> Union[bool, str]:
    stages = ["em", "nvt", "npt", "prod"]
    if mdp_dir is None:
        mdp_dir = Path(__file__).parent
    else:
        mdp_dir = Path(mdp_dir).resolve()
    
    commands = []
    with set_directory(wdir, mkdir=True) as w:
        for i, stage in enumerate(stages):
            commands.append(f"cd {stage}")
            Path.mkdir(w / stage, parents=True, exist_ok=True)
            shutil.copyfile(mdp_dir / f"{stage}.mdp", w / stage / f"{stage}.mdp")
            if verbose and logger and (not return_commands_only): logger.info(f"Running {stage}...")
            if stage in ['nvt', 'npt'] and top_posre is not None:
                top_use = Path(top_posre).resolve()
            else:
                top_use = Path(top).resolve()
            commands += run_md(
                top = top_use,
                gro = Path(gro).resolve(),
                mdp = w / stage / f"{stage}.mdp",
                deffnm = stage,
                wdir = Path.resolve(w / stage),
                cpt = f"../{stages[i-1]}/{stages[i-1]}.cpt" if i >= 2 else None,
                restr_gro = f"../{stages[i-1]}/{stages[i-1]}.gro" if (i > 0 and i < 3) else None, 
                restart = bool(i) if restart else False, # em don't restart
                params = params[stage],
                enforce_gpu = bool(i) if enforce_gpu else False,
                return_commands_only=return_commands_only
            )
            gro = Path(w / stage / f"{stage}.gro")
            commands.append("cd ..")
    
    if return_commands_only:
        return '\n'.join(commands)
    else:
        return True