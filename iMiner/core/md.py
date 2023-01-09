"""
Author: Eric Wang
Data Created: 10/24/2022

This package contains class for running molecular dynamics project
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import json

from iMiner.cmd import run_command, set_directory, find_executable
from iMiner.core.project import BaseProject
from iMiner.md.prep.ligand import run_acpype
from iMiner.md.prep.protein import run_tleap
from iMiner.md.prep.complex import make_complex
from iMiner.md.runner.gromacs import run_preprocess_workflow, run_md_workflow
from iMiner.log import LOGGER


class MDProject(BaseProject):
    def __init__(self, project_name:str, project_path: Optional[os.PathLike] = None, engine: str = "gromacs") -> None:
        '''
        Initialize a MD project with a project name and a project path

        When project_path is None, the project will be initialized in the current working directory,
        using the project_name as the project folder name

        :param engine: molecular dynamics engine, only support "gromacs" currently
        :type engine: str
        '''
        super().__init__(project_name, project_path)
        self.md_path = Path(self.project_path) / "md"
        self.md_path.mkdir(exist_ok=True, parents=True)
        assert engine in ['gromacs'], f"Not supported MD engine: {engine}"
        self.params = {
            "md": {
                "enforce_gpu": True,
                "em": {},
                "nvt": {},
                "npt": {},
                "prod": {}
            }
        }
    
    @property
    def md_params(self) -> Dict[str, Any]:
        return self.params['md']
    
    def read_params_json(self, fname: os.PathLike):
        with open(fname, 'r') as f:
            self.params.update(json.load(f))

    def parametrize_ligand(self, name: str, wdir: Path):
        """
        Parametrize ligand
        """
        prep_path = wdir.resolve() / "ligand"
        prep_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.get_ligand_with_name(name), prep_path / "ligand.sdf")
        with set_directory(prep_path):
            obabel = find_executable(['obabel'])
            mol = Chem.SDMolSupplier("ligand.sdf", removeHs=False)[0]
            net_charge = sum([at.GetFormalCharge() for at in mol.GetAtoms()])
            run_acpype("ligand.sdf", net_charge=net_charge)
            run_command([obabel, 'ligand.sdf', '-O', 'MOL.gro'])
    
    def parametrize_protein(self, name: str, wdir: Path):
        """
        Parametrize protein
        """
        prep_path = wdir.resolve() / "protein"
        prep_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.get_protein_with_name(name), prep_path / "protein.pdb")
        with set_directory(prep_path):
            run_tleap("protein.pdb")
            run_acpype(args=["-p", "protein.prmtop", "-x", "protein.inpcrd"])
    
    def make_complex(self, wdir: Path):
        """
        Make protein-ligand complex
        """
        prep_path = wdir.resolve() / "complex"
        prep_path.mkdir(parents=True, exist_ok=True)
        make_complex(
            wdir.resolve() / "protein" / "protein.amb2gmx" / "protein_GMX.top",
            wdir.resolve() / "ligand" / "MOL.acpype" / "MOL_GMX.itp",
            wdir.resolve() / "protein" / "protein.amb2gmx" / "protein_GMX.gro",
            wdir.resolve() / "ligand" / "MOL.gro",
            prep_path
        )
    
    def prep_md(self, wdir: Path):
        """
        Run MD preparation workflow
        """
        complex_dir = wdir.resolve() / "complex"
        run_preprocess_workflow("topol.top", "complex.gro", complex_dir, verbose=True)
        shutil.copyfile(complex_dir / "ions.gro", wdir / "ions.gro")
        shutil.copyfile(complex_dir / "processed.top", wdir / 'processed.top')
    
    def run_md(self, wdir: Path):
        """
        Run molecular dynamics workflow
        """
        run_md_workflow(
            "processed.top", 
            "ions.gro", 
            wdir, 
            restart=True, 
            params=self.md_params, 
            enforce_gpu=self.md_params['enforce_gpu'],
            verbose=True
        )
    
    def run(self, lig_name: str, prot_name: str, task_name: Optional[str] = None):
        """
        Run iMiner Molecular Dynamics Workflow
        """
        task_name = f"{lig_name}_{prot_name}" if task_name is None else task_name
        wdir = self.md_path / task_name

        def log_step(n: int, msg: str):
            LOGGER.info(f"===== Step {n}: {msg.capitalize()} =====")
        
        log_step(1, "Parametrize Ligand")
        self.parametrize_ligand(lig_name, wdir)
        log_step(2, "Parametrize Protein")
        self.parametrize_protein(prot_name, wdir)
        log_step(3, "Make Complex")
        self.make_complex(wdir)
        log_step(4, "MD Preparation")
        self.prep_md(wdir)
        log_step(5, "Run MD")
        self.run_md(wdir)
