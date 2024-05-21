from pathlib import Path
from iMiner.md.amber.prep import create_solvated_box


def test_create_ligand_solvated_box():
    wdir = Path(__file__).parent / 'data/methane'
    create_solvated_box(
        ligand_mol2=wdir / "MOL_gas_gaff2.mol2",
        ligand_lib=wdir / "MOL_AC.lib",
        ligand_frcmod=wdir / "MOL_AC.frcmod",
        wdir=wdir,
        ligand_name="ligand",
        water_ff="tip3p",
        ligand_ff='gaff2',
        buffer=10.0,
        ionic_strength=0.0,
        box_resize=0.75
    )


def test_create_protein_solvated_box():
    wdir = Path(__file__).parent / 'data/protein'
    wdir.mkdir(exist_ok=True)
    create_solvated_box(
        protein_pdb=wdir.parent / "ala-dipeptide.pdb",
        wdir=wdir,
        protein_name='protein',
        water_ff='tip3p',
        protein_ff='ff14SB',
        buffer=10.0,
        ionic_strength=0.0,
        box_resize=0.75
    )


def test_create_complex_solvated_box():
    wdir = Path(__file__).parent / 'data/complex'
    wdir.mkdir(exist_ok=True)
    create_solvated_box(
        ligand_mol2=wdir.parent / "methane/MOL_gas_gaff2.mol2",
        ligand_lib=wdir.parent / "methane/MOL_AC.lib",
        ligand_frcmod=wdir.parent / "methane/MOL_AC.frcmod",
        protein_pdb=wdir.parent / "ala-dipeptide.pdb",
        protein_name="protein",
        protein_ff='ff14SB',
        wdir=wdir,
        ligand_name="ligand",
        water_ff="tip3p",
        ligand_ff='gaff2',
        buffer=10.0,
        ionic_strength=0.0,
        box_resize=0.75
    )