import pytest
import shutil
from typing import Dict
from pathlib import Path
from iMiner.cmd import set_directory
from iMiner.md.prep.protein import run_tleap
from iMiner.md.prep.complex import split_top, make_complex, make_solvated
from iMiner.md.runner.gromacs import run_md_workflow


@pytest.mark.parametrize(
    "protein_ff",
    ["ff19SB", "ff14SB", "ff03", "ff99SB", "ff99"]
)
def test_parametrize_protein(protein_ff):
    wdir = Path(f"tmp_{protein_ff}")
    wdir.mkdir(exist_ok=True)
    pdb = wdir / 'protein.pdb'
    shutil.copyfile(
        Path(__file__).parent / "data" / "ala-dipeptide.pdb",
        pdb
    )
    run_tleap(pdb, protein_ff=protein_ff)
    shutil.rmtree(wdir)


@pytest.mark.parametrize(
    "fname,info",
    [("MOL.top", {}), ("protein.top", {"SOL": 53, "protein": 1})],
)
def test_split_top(fname: str, info: Dict[str, int]):
    datadir = Path(__file__).parent / "data"
    top = datadir / fname
    atp = top.with_suffix(".atp")
    itp = top.with_suffix(".itp")
    test_info = split_top(top, atp, itp)
    for k in info:
        assert info[k] == test_info[k]
    with open(itp, 'r') as f:
        for line in f:
            assert "[ atomtypes ]" not in line
            assert "SOL" not in line
            assert "WAT" not in line
    
    atp.unlink()
    itp.unlink()


def test_make_complex():
    datadir = Path(__file__).parent / "data"
    wdir = datadir / 'complex'
    wdir.mkdir(exist_ok=True)
    wdir.resolve()
    with set_directory(datadir):
        make_complex(
            protein_top="protein.top",
            ligand_top="MOL.top",
            protein_gro="protein.gro",
            ligand_gro='MOL.gro',
            protein_posre="posre_protein.itp",
            ligand_posre='posre_MOL.itp',
            complex_dir=wdir,
            complex_top_name="topol.top",
            complex_gro_name="complex.gro"
        )
    assert Path.is_file(wdir / "topol.top")
    assert Path.is_file(wdir / "complex.gro")
    shutil.rmtree(wdir)
 

def test_make_solvated():
    datadir = Path(__file__).parent / "data"
    wdir = datadir / 'solvated'
    wdir.mkdir(exist_ok=True)
    wdir = wdir.resolve()
    with set_directory(datadir):
        make_solvated(
            ligand_top="MOL.top",
            ligand_gro='MOL.gro',
            ligand_posre='posre_MOL.itp',
            wdir=wdir,
            solvated_top_name="topol.top",
        )
    assert Path.is_file(wdir / "topol.top")
    shutil.rmtree(wdir)


def test_make_commands():
    datadir = Path(__file__).parent / "data"
    wdir = datadir / 'test_make_commands'
    commands = run_md_workflow(
        top = datadir / 'MOL.top',
        gro = datadir / 'MOL.gro',
        wdir = wdir,
        enforce_gpu=True,
        restart=True,
        return_commands_only=True
    )
    shutil.rmtree(wdir)
