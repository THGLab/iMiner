"""
Author: Eric Wang
Data Created: 02/27/2023

This package contains class for running molecular dynamics project
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import json

import numpy as np
from iMiner.cmd import run_command, set_directory, find_executable
from iMiner.core.md import MDProject
from iMiner.md.runner.gromacs import run_preprocess_workflow, run_md_workflow


class RbfeProject(MDProject):
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

        super().__init__(project_name, project_path, verbose, logger, temp_path, engine)
        assert engine in ['gromacs'], f"Not supported MD engine: {engine}"
        self.params = {
            "rbfe": {
                "protein_ff": "ff14SB",
                "enforce_gpu": False,
                "em": {},
                "nvt": {},
                "npt": {},
                "prod": {},
                "interaction_analysis": {
                    "dt": 200, # ps
                    "use_mpi": True,
                    "chunksize": 1,
                    "gen_short_dt": 1000
                },
                "fep-lambdas": [
                    0.000, 0.020, 0.050, 0.100, 0.189, 0.278, 0.367, 0.456, 
                    0.544, 0.633, 0.722, 0.811, 0.900, 0.950, 0.980, 1.000
                ]
            }
        }
        self.update_lambdas()

    def update_lambdas(self):
        lambda_keys = [
            'fep-lambdas', 'coul-lambdas', 'vdw-lambdas', 'bonded-lambdas',
            'restraint-lambdas', 'mass-lambdas', 'temperature-lambdas'
        ]
        self.num_lambdas = None
        for key in self.md_params:
            if key in lambda_keys:
                if self.num_lambdas is None: self.num_lambdas = len(self.md_params[key])
                assert len(self.md_params[key]) == self.num_lambdas, "Number of lambdas inconsisent"
                for stage in ["em", "nvt", "npt", "prod"]:
                    self.md_params[stage][key] = " ".join(f'{x:.3f}' for x in self.md_params[key])

    @property
    def md_params(self) -> Dict[str, Any]:
        return self.params['rbfe']
    
    def read_params_json(self, fname: os.PathLike):
        with open(fname, 'r') as f:
            jdata = json.load(f)
        
        # enforce 'nstxout-compressed/nstxout' == 'nstlog/nstenergy'
        for stage in ['em', 'nvt', 'npt', 'prod']:
            for key in ['nstenergy', 'nstlog', 'nstdhdl']:
                if key not in jdata['rbfe'][stage]:
                    if "nstxout-compressed" in jdata['rbfe'][stage]:
                        jdata['rbfe'][stage][key] = jdata['rbfe'][stage]['nstxout-compressed']
                    elif "nstxout" in jdata['rbfe'][stage]:
                        jdata['rbfe'][stage][key] = jdata['rbfe'][stage]['nstxout']
        
        for key in jdata['rbfe']:
            if isinstance(self.md_params[key], dict):
                self.md_params[key].update(jdata['rbfe'][key])
            else:
                self.md_params[key] = jdata['rbfe'][key]
        
        self.update_lambdas()
                
    def parametrize_ligand(self, names: Tuple[str, str], wdir: Path, remove_cache: bool = True):
        """
        Parametrize ligand
        """
        from iMiner.md.prep.ligand import run_acpype
        
        for tag, name in zip(["A", "B"], names):
            self.logger.info(f"Paramterizing ligand {tag}: {name}")
            prep_path = wdir.resolve() / f"ligand{tag}"
            # delete all existing acpype files, otherwise acpype reuses old files
            if remove_cache and prep_path.is_dir():
                shutil.rmtree(prep_path)
            prep_path.mkdir(parents=True)
            shutil.copyfile(self.get_ligand_with_name(name), prep_path / "ligand.sdf")
            with set_directory(prep_path):
                obabel = find_executable(['obabel'])
                run_acpype("ligand.sdf")
                run_command([obabel, 'ligand.sdf', '-O', 'MOL.gro'])
    
    def make_perturbed_topology(self, wdir: Path, mapping: np.ndarray):
        """
        Make perturbed topology
        """
        from iMiner.md.rbfe.topology import GromacsTopologyFilePerturb
        
        structA = GromacsTopologyFilePerturb(
            str(wdir / "ligandA" / "MOL.acpype" / "MOL_GMX.top"),
            xyz=str(wdir / "ligandA" / "MOL.gro")
        )
        structB = GromacsTopologyFilePerturb(
            str(wdir / "ligandB" / "MOL.acpype" / "MOL_GMX.top"),
            xyz=str(wdir / "ligandB" / "MOL.gro")
        )
        structA.reset_box() # MOL.gro with box vector 0
        structA.perturb(structB, mapping)
        structA.write(str(wdir / "merged.top"), itp=False)
        structA.save(str(wdir / "merged.gro"), overwrite=True)
        # Generate Restraint File
        fp = open(wdir / "posre_merged.itp", "w")
        fp.write("[ position_restraints ]\n")
        fp.write("; atom  type    fx    fy    fz\n")
        for i, atom in enumerate(structA.atoms):
            if "H" not in atom.name:
                fp.write(f"{i+1:>6}     1  1000  1000 1000\n")
        fp.write("\n\n")
        fp.close()
    
    def make_systems(self, wdir: Path):
        """
        Make protein-ligand complex and ligand solvated system
        """
        from iMiner.md.prep.complex import make_complex, make_solvated

        self.logger.info("Making protein-ligand complex system...")
        complex_path = wdir.resolve() / "complex"
        complex_path.mkdir(parents=True, exist_ok=True)
        make_complex(
            wdir.resolve() / "protein" / "protein.amb2gmx" / "protein_GMX.top",
            wdir.resolve() / "merged.top",
            wdir.resolve() / "protein" / "protein.amb2gmx" / "protein_GMX.gro",
            wdir.resolve() / "merged.gro",
            wdir.resolve() / 'protein' / 'protein.amb2gmx' / 'posre_protein.itp',
            wdir.resolve() / "posre_merged.itp",
            complex_path
        )

        self.logger.info("Making ligand solvated system...")
        solvated_path = wdir.resolve() / "solvated"
        solvated_path.mkdir(parents=True, exist_ok=True)
        make_solvated(
            wdir.resolve() / "merged.top",
            wdir.resolve() / "merged.gro",
            wdir.resolve() / "posre_merged.itp",
            solvated_path
        )

    def prep_md(self, wdir: Path):
        """
        Run MD preparation workflow
        """
        self.logger.info("Prepare MD for solvated system...")
        run_preprocess_workflow("topol.top", "ligand.gro", wdir.resolve() / "solvated", verbose=True, logger=self.logger)
        self.logger.info("Prepare MD for complex system...")
        run_preprocess_workflow("topol.top", "complex.gro", wdir.resolve() / "complex", verbose=True, logger=self.logger)
        
    def run_md(self, wdir: Path):
        """
        Run molecular dynamics workflow
        """
        import iMiner.md.rbfe as rbfe
        
        for tag in ["solvated", "complex"]:
            for i in range(self.num_lambdas):
                self.logger.info(f"Running MD for {tag}/lambda{i}...")
                md_dir = wdir / tag / f"lambda{i}"
                md_dir.mkdir(parents=True, exist_ok=True)
                for stage in ['em', 'nvt', 'npt', 'prod']:
                    self.md_params[stage]['init-lambda-state'] = i
                run_md_workflow(
                    wdir / tag / "processed.top", 
                    wdir / tag / "ions.gro", 
                    md_dir, 
                    restart=True, 
                    params=self.md_params, 
                    enforce_gpu=self.md_params['enforce_gpu'],
                    verbose=True,
                    logger=self.logger,
                    mdp_dir=Path(rbfe.__path__[0]).resolve()
                )
    
    def clean(self, wdir: Path):
        """
        Clean GROMACS intermediate/duplicated files
        """
        for tmpfile in Path(wdir).glob("*/lambda*/*/#*#"):
            tmpfile.unlink()
    
    def run(
        self, 
        lig_names: Tuple[str, str], 
        prot_name: str, 
        mapping: Union[os.PathLike, np.ndarray],
        task_name: Optional[str] = None
    ):
        """
        Run iMiner Relative Binding Free Energy Calculation Workflow
        """
        task_name = f"{lig_names[0]}~{lig_names[1]}" if task_name is None else task_name
        self.logger.info(f"Running RBFE calculation for {task_name}")

        self.md_path = Path(self.project_path) / "rbfe"
        self.md_path.mkdir(exist_ok=True, parents=True)

        wdir = self.md_path / task_name
        
        self.log_step("Parametrize Ligand")
        self.parametrize_ligand(lig_names, wdir)
        
        self.log_step("Parametrize Protein")
        self.parametrize_protein(prot_name, wdir)
        
        self.log_step("Make Perturbed Topology")
        if isinstance(mapping, str) or isinstance(mapping, Path):
            mapping = np.loadtxt(mapping, dtype=int)
        else:
            mapping = np.array(mapping, dtype=int)
        self.make_perturbed_topology(wdir, mapping)
        
        self.log_step("Make Solvated and Complex Systems")
        self.make_systems(wdir)
        
        self.log_step("MD Preparation")
        self.prep_md(wdir)
        
        self.log_step("Run MD")
        self.run_md(wdir)        
        
        self.log_step("Clean working directory")
        self.clean(wdir)
        
        return True