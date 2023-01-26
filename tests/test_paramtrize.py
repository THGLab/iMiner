import pytest
import shutil
from pathlib import Path
from iMiner.md.prep.protein import run_tleap


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
