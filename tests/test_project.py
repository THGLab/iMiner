import pytest
import shutil
from pathlib import Path
from iMiner.core.project import BaseProject
from iMiner.core.md import MDProject


def test_base_project():
    proj_name = "project_test"
    proj_path = Path(__file__).parent / proj_name
    proj = BaseProject(project_name=proj_name, project_path=proj_path, verbose=False)
    proj.add_protein(
        name="ala", 
        protein_file_path=Path(__file__).parent / "data" / "ala-dipeptide.pdb"
    )
    proj.add_ligand(smiles_or_path="C", name="methane")
    proj.add_ligand(smiles_or_path="CC", name="ethane")
    
    assert Path.is_file(proj_path / ".iminer" / "meta.json")
    assert Path.is_file(proj_path / "proteins" / "ala.pdb")
    assert Path.is_file(proj_path / "ligands" / "methane.sdf")
    assert Path.is_file(proj_path / "ligands" / "ethane.sdf")

    proj_loaded = BaseProject(project_path=proj_path, verbose=False)
    assert proj_loaded.project_name == proj_name
    assert proj_loaded.ligands["methane"] == proj_path / "ligands" / "methane.sdf"
    assert proj_loaded.ligands['ethane'] == proj_path / "ligands" / 'ethane.sdf'
    assert proj_loaded.proteins['ala'] == proj_path / "proteins" / 'ala.pdb'

    shutil.rmtree(proj_path)


def test_md_project():
    proj_name = "project_md_test"
    proj_path = Path(__file__).parent / proj_name
    proj = MDProject(project_name=proj_name, project_path=proj_path, verbose=False)
    proj.add_protein(
        name='ala',
        protein_file_path=Path(__file__).parent / "data/ala-dipeptide.pdb"
    )
    proj.add_ligand(smiles_or_path="C", name='methane')

    proj_loaded = MDProject(project_path=proj_path, verbose=False)
    proj_loaded.add_protein(
        name="ala-2", 
        protein_file_path=Path(__file__).parent / "data/ala-dipeptide.pdb"
    )
    proj.add_ligand(smiles_or_path="CC", name='ethane')
    shutil.rmtree(proj_path)