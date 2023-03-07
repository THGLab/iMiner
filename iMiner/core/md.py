"""
Author: Eric Wang
Data Created: 10/24/2022

This package contains class for running molecular dynamics project
"""
import os, sys
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import json

import numpy as np
import matplotlib.pyplot as plt

from iMiner.utils import timer
from iMiner.cmd import run_command, set_directory, find_executable, CommandExecuteError
from iMiner.core.project import BaseProject


class MDProject(BaseProject):
    def __init__(
        self, 
        project_name: Optional[str] = None, 
        project_path: Optional[os.PathLike] = None, 
        verbose: bool = True, 
        logger: Optional[os.PathLike] = "iMiner.log",
        temp_path: os.PathLike = "/tmp",
        engine: str = "gromacs"
    ) -> None:
        '''
        Initialize a molecule dynamics project

        Parameters
        ----------
        project_name: str
            Name of the project. If None, the name of the project will be determined from `project_path`. Default is None.
        project_path: os.PathLike or None
            Path of the project. If None, the path of the project will be `pwd/project_name`. Default is None.
            If not None and `project_path/.iminer/meta.json` exists, the project will be initialized using the meta data.
            Note that `project_path` and `project_name` cannot be set to None simultaneously.
        verbose: bool
            Whether to print verbose information. Default is True.
        logger: os.PathLike
            Path to log file. If relative path, the log file will be `project_path/logger`. If None, will not have log file.
            Default is iMiner.log
        temp_path: os.PathLike
            Path to store temporary files. Default is `/tmp`
        engine: str
            MD engine. Only "gromacs" supported currently.
        '''

        super().__init__(project_name, project_path, verbose, logger, temp_path)
        self.md_path = Path(self.project_path) / "md"
        self.md_path.mkdir(exist_ok=True, parents=True)
        assert engine in ['gromacs'], f"Not supported MD engine: {engine}"
        self.params = {
            "md": {
                "protein_ff": "ff14SB",
                "enforce_gpu": True,
                "em": {},
                "nvt": {},
                "npt": {},
                "prod": {},
                "interaction_analysis": {
                    "dt": 200, # ps
                    "use_mpi": True,
                    "chunksize": 1,
                    "gen_short_dt": 1000
                }
            }
        }
        self.step_cnt = 0

    @property
    def md_params(self) -> Dict[str, Any]:
        return self.params['md']
    
    def read_params_json(self, fname: os.PathLike):
        with open(fname, 'r') as f:
            jdata = json.load(f)
        
        # enforce 'nstxout-compressed/nstxout' == 'nstlog/nstenergy'
        for stage in ['em', 'nvt', 'npt', 'prod']:
            for key in ['nstenergy', 'nstlog']:
                if key not in jdata['md'][stage]:
                    if "nstxout-compressed" in jdata['md'][stage]:
                        jdata['md'][stage][key] = jdata['md'][stage]['nstxout-compressed']
                    elif "nstxout" in jdata['md'][stage]:
                        jdata['md'][stage][key] = jdata['md'][stage]['nstxout']
        
        for key in jdata['md']:
            if isinstance(self.md_params[key], dict):
                self.md_params[key].update(jdata['md'][key])
            else:
                self.md_params[key] = jdata['md'][key]

    def parametrize_ligand(self, name: str, wdir: Path, **kwargs):
        """
        Parametrize ligand
        """
        from iMiner.md.prep.ligand import run_acpype

        prep_path = wdir.resolve() / "ligand"
        # delete all existing acpype files, otherwise acpype reuses old files
        if prep_path.exists() and prep_path.is_dir():
            shutil.rmtree(prep_path) 
        prep_path.mkdir(parents=True)
        shutil.copyfile(self.get_ligand_with_name(name), prep_path / "ligand.sdf")
        with set_directory(prep_path):
            obabel = find_executable(['obabel'])
            try:
                run_acpype("ligand.sdf", **kwargs)
            except CommandExecuteError:
                self.logger.error(f"Error in preparing ligand {self.get_ligand_with_name(name)}. See details in acpype log file.")
                sys.exit(1)
            try:
                run_command([obabel, 'ligand.sdf', '-O', 'MOL.gro'])
            except CommandExecuteError:
                self.logger.error(f"Error in converting ligand {self.get_ligand_with_name(name)} with obabel.")
                sys.exit(1)
        return True
    
    def parametrize_protein(self, name: str, wdir: Path):
        """
        Parametrize protein
        """
        from iMiner.md.prep.protein import run_tleap, fix_hydrogen
        from iMiner.md.prep.ligand import run_acpype

        prep_path = wdir.resolve() / "protein"
        prep_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.get_protein_with_name(name), prep_path / "protein.pdb")
        with set_directory(prep_path):
            try:
                fix_hydrogen("protein.pdb", "protein_processed.pdb")
            except CommandExecuteError as e:
                self.logger.error(f"Error in fixing hydrogen atoms: {e}")
                sys.exit(1)

            try:
                run_tleap("protein_processed.pdb", protein_ff=self.md_params['protein_ff'])
            except CommandExecuteError as e:
                self.logger.error(f"Error in preparing protein {self.get_protein_with_name(name)}. See details in tleap log file.")
                sys.exit(1)
            try:
                run_acpype(args=["-p", "protein.prmtop", "-x", "protein.inpcrd"])
            except CommandExecuteError as e:
                self.logger.error(f"Error in preparing protein {self.get_protein_with_name(name)}. See details in acpype log file.")
                sys.exit(1)
        return True
    
    def make_complex(self, wdir: Path):
        """
        Make protein-ligand complex
        TODO: error handlings
        """
        from iMiner.md.prep.complex import make_complex

        prep_path = wdir.resolve() / "complex"
        prep_path.mkdir(parents=True, exist_ok=True)
        make_complex(
            wdir.resolve() / "protein" / "protein.amb2gmx" / "protein_GMX.top",
            wdir.resolve() / "ligand" / "MOL.acpype" / "MOL_GMX.itp",
            wdir.resolve() / "protein" / "protein.amb2gmx" / "protein_GMX.gro",
            wdir.resolve() / "ligand" / "MOL.gro",
            wdir.resolve() / 'protein' / 'protein.amb2gmx' / 'posre_protein.itp',
            wdir.resolve() / 'ligand' / 'MOL.acpype' / "posre_MOL.itp",
            prep_path
        )
    
    def remove_pbc_workflow(self, wdir: Path, version: int = 1):
        """
        Remove PBC
        """
        from iMiner.md.common import (
            preprocess_index_file, read_single_gro, parse_index_file, mk_index_file, gmx_genidx
        )

        prod_dir = wdir.resolve() / "prod"
        index_file = prod_dir / "index.ndx"
        traj_nopbc_file = prod_dir / "prod_align.xtc"
        
        # Generate index file
        gmx_genidx(prod_dir / "prod.gro", index_file)
        r_grp_idx, l_grp_idx, c_grp_idx = preprocess_index_file(index_file, index_file)
        self.logger.info(f"Index file generated: {index_file}")
        self.logger.info(f"Receptor group index: {r_grp_idx}")
        self.logger.info(f"Ligand group index: {l_grp_idx}")
        self.logger.info(f"Complex group index: {c_grp_idx}")
        
        if version == 2:
            coords = read_single_gro(wdir / "complex" / "complex.gro")
            center_pos = np.mean(coords, axis=0)
            center_idx = np.argmin(
                np.linalg.norm(coords - center_pos, ord=2, axis=1)
            ) + 1
            group_dict = parse_index_file(index_file)
            group_dict["center"] = [center_idx]
            mk_index_file(group_dict, index_file)

        # Remove PBC
        base_cmds = [find_executable(['gmx_mpi', 'gmx']), 'trjconv', '-n', index_file]
        if version == 1:
            # Step 1: make all the molecules as a whole and center the protein
            self.logger.info("Remove PBC Step 1: make all the molecules as a whole and center the protein")
            tmpf1 = prod_dir / "prod_whole_center.xtc"
            cmds = base_cmds.copy()
            cmds += ['-s', prod_dir / "prod.tpr", '-f', prod_dir / "prod.xtc", '-o', tmpf1, '-pbc', 'whole', '-center']
            run_command(cmds, input=f"Protein\n{c_grp_idx}")
            
            # Step 2: make all the molecules within the box
            self.logger.info("Remove PBC Step 2: make all the molecules within the box")
            tmpf2 = prod_dir / "prod_whole_center_nojump.xtc"
            cmds = base_cmds.copy()
            cmds += ['-s', prod_dir.parent / 'complex' / "complex.gro", '-f', tmpf1, '-o', tmpf2, '-pbc', 'nojump']
            run_command(cmds, input=str(c_grp_idx))
        
        elif version == 2:
            # Step 1: make all the molecules as a whole
            self.logger.info("Remove PBC Step 1: make all the molecules as a whole")
            tmpf1 = prod_dir / "prod_whole.xtc"
            cmds = base_cmds.copy()
            cmds += ['-s', prod_dir / "prod.tpr", '-f', prod_dir / "prod.xtc", '-o', tmpf1, '-pbc', 'whole']
            run_command(cmds, input=f"{c_grp_idx}")

            #Step 2: make all the molecules within the box by centering the central atom
            self.logger.info("Remove PBC Step 2: make all the molecules within the box by centering the central atom")
            tmpf2 = prod_dir / "prod_nojump_center.xtc"
            cmds = base_cmds.copy()
            cmds += ['-s', wdir / "complex" / "newbox.gro", '-f', tmpf1, '-o', tmpf2, '-pbc', 'nojump', '-center']
            run_command(cmds, input=f"center\n{c_grp_idx}")

        else:
            raise NotImplementedError(f"Invalid remove PBC workflow version: {version}")
        
        # Step 3: align
        self.logger.info("Remove PBC Step 3: align")
        cmds = base_cmds.copy()
        cmds += ['-s', prod_dir / 'prod.tpr', '-f', tmpf2, '-o', traj_nopbc_file, '-fit', 'rot+trans']
        run_command(cmds, input=f'{c_grp_idx}\n{c_grp_idx}')
        
        # Generate short trajectory for visualization
        self.logger.info("Remove PBC Step 4: generate short trajectory for visualization")
        base_cmds += ['-s', prod_dir / 'prod.tpr', '-f', traj_nopbc_file]
        cmds = base_cmds.copy()
        cmds += ['-o', traj_nopbc_file.with_suffix('.pdb'), '-dump', 0] # dump 1st frame
        run_command(cmds, input=str(c_grp_idx))
        short_dt = self.md_params['interaction_analysis']['gen_short_dt']
        if short_dt:
            output_file = prod_dir / f'{traj_nopbc_file.stem}_dt{short_dt // 1000}ns.xtc'
            cmds = base_cmds.copy()
            cmds = base_cmds + ['-o', output_file, '-dt', short_dt]
            run_command(cmds, input=str(c_grp_idx))
        
        # Generate sub tpr file for complex
        self.logger.info("Generate sub tpr file for dry complex structure")
        run_command(
            [
                find_executable(['gmx_mpi', 'gmx']),
                "convert-tpr",
                "-s", prod_dir / "prod.tpr",
                "-o", prod_dir / "prod_align.tpr",
                "-n", index_file
            ], input=str(c_grp_idx)
        )
        # Clean tmp file
        tmpf1.unlink()
        tmpf2.unlink()
           
    def analyze_md_traj(self, wdir: Path, lig_name: str):
        """
        Analyze MD trajectories: calculating RMSD and do interaction analysis
        """ 
        from iMiner.md.analysis.interaction import analyze_multiple_frames, plot_interact
        from iMiner.md.analysis.traj import (
            plot_rmsd,
            read_xvg,
            gmx_rms,
            xtc_to_pdb,
        )

        prod_dir = wdir.resolve() / "prod"
        ref_tpr_align = prod_dir / "prod_align.tpr"
        index_file = prod_dir / "index.ndx"
        traj_nopbc_file = prod_dir / "prod_align.xtc"

        # rmsd
        f_xvg = prod_dir / "prod_rmsd.xvg"
        f_rmsd_png = prod_dir / "prod_rmsd.png"  
        gmx_rms(
            ref_tpr_align,
            traj_nopbc_file,
            f_xvg,
            index_file,
        )
        tlist, rmslist = read_xvg(f_xvg, tunit='ns', dunit='A')
        if np.any(rmslist > 30):
            self.logger.warning("Large RMSD found! PBC may not be fixed properly.")
        fig, ax = plot_rmsd(tlist, rmslist, name=lig_name)
        fig.savefig(f_rmsd_png, dpi=300)
        plt.close(fig)
        self.logger.info(f"RMSD Calculated: {f_rmsd_png}")

        # analyze interaction
        self.logger.info("Analyze interaction...")
        with timer("Analyze interaction", self.logger):
            trajdir = prod_dir / "traj"
            f_csv = prod_dir / "interaction.csv"
            f_interact_png = prod_dir / 'interaction.png'
            if not trajdir.is_dir():
                trajdir.mkdir(exist_ok=True)
                pdbs = xtc_to_pdb(
                    ref_tpr_align, traj_nopbc_file, trajdir, 
                    self.md_params['interaction_analysis']['dt']
                )
                self.logger.info(f"Convert trajectory to seperate pdb files: {trajdir}")
            else:
                self.logger.warning(f"Found trajector directory: {trajdir}, xtc_to_pdb conversion is skipped.")
                pdbs = list(trajdir.glob("*.pdb"))
            df = analyze_multiple_frames(
                pdbs,
                f_csv,
                add_hydrogen=False,
                resnr_renum=None,
                use_mpi=self.md_params['interaction_analysis']['use_mpi'], 
                chunksize=self.md_params['interaction_analysis']['chunksize']
            )
            fig, ax = plot_interact(f_csv, title=lig_name)
            fig.savefig(f_interact_png, dpi=300)
            plt.close(fig)
            self.logger.info(f"Intearction analysis result save to: {f_interact_png}")
        
        return

    def prep_md(self, wdir: Path):
        """
        Run MD preparation workflow
        """
        from iMiner.md.runner.gromacs import run_preprocess_workflow

        complex_dir = wdir.resolve() / "complex"
        try:
            run_preprocess_workflow("topol.top", "complex.gro", complex_dir, verbose=True, logger=self.logger)
        except CommandExecuteError:
            self.logger.info("Error in gromacs prep steps. See complex folder.")
            sys.exit(1)
        shutil.copyfile(complex_dir / "ions.gro", wdir / "ions.gro")
        shutil.copyfile(complex_dir / "processed.top", wdir / 'processed.top')
        return True
    
    def run_md(self, wdir: Path):
        """
        Run molecular dynamics workflow
        """
        from iMiner.md.runner.gromacs import run_md_workflow

        try:
            run_md_workflow(
                "processed.top", 
                "ions.gro", 
                wdir, 
                restart=True, 
                params=self.md_params, 
                enforce_gpu=self.md_params['enforce_gpu'],
                verbose=True,
                logger=self.logger
            )
        except CommandExecuteError as e:
            self.logger.error("Error in running gromacs. See complex folder.")
            sys.exit(1)
        return True
    
    def clean(self, wdir: Path):
        """
        Clean GROMACS intermediate/duplicated files
        """
        for tmpfile in Path(wdir).glob("*/#*#"):
            tmpfile.unlink()
    
    def log_step(self, msg: str):
        self.step_cnt += 1
        self.logger.info(f"===== Step {self.step_cnt}: {msg.capitalize()} =====")
    
    def run(self, lig_name: str, prot_name: str, task_name: Optional[str] = None, 
            lig_charge = "auto"):
        """
        Run iMiner Molecular Dynamics Workflow
        
        Returns pre-md topology and md production run trajectory, to be used as trajectory
        rmsd calculation inputs
        """
        task_name = f"{lig_name}_{prot_name}" if task_name is None else task_name
        self.logger.info(f"Running md for {task_name}")
        wdir = self.md_path / task_name
        
        self.log_step("Parametrize Protein")
        self.parametrize_protein(prot_name, wdir)

        self.log_step("Parametrize Ligand")
        self.parametrize_ligand(lig_name, wdir, net_charge=lig_charge)

        self.log_step("Make Complex")
        self.make_complex(wdir)
        
        self.log_step("MD Preparation")
        self.prep_md(wdir)
        
        self.log_step("Run MD")
        self.run_md(wdir)
        
        self.log_step("Remove PBC of MD Trajectory")
        self.remove_pbc_workflow(wdir)
        
        self.log_step("Analyze RMSD and Interactions")
        self.analyze_md_traj(wdir, lig_name)
        
        self.log_step("Clean working directory")
        self.clean(wdir)
        
        complex_dir = wdir.resolve() / "complex"
        prod_dir = wdir.resolve() / "prod"
        
        return complex_dir / "ions.tpr", prod_dir/ "prod.xtc" 

    def show_interaction(self, task_name: str):
        from IPython.display import Image
        return Image(str(self.project_path / "md" / task_name / "prod" / "interaction.png"))

    def show_rmsd(self, task_name: str):
        from IPython.display import Image
        return Image(str(self.project_path / "md" / task_name / "prod" / "prod_rmsd.png"))