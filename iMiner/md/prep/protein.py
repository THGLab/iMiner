import os
from pathlib import Path
from typing import List, Optional

from iMiner.cmd import run_command, find_executable, set_directory
from iMiner.utils import file_abspath


PROTEIN_FF_KEYS = {
    "ff14SB": "leaprc.protein.ff14SB",
    "ff19SB": "leaprc.protein.ff19SB",
    "ff03": "oldff/leaprc.ff03",
    "ff99SB": "oldff/leaprc.ff99SB",
    "ff99": "oldff/leaprc.ff99"
}

def fix_protein(
    in_pdb: os.PathLike, 
    out_pdb: os.PathLike,
    add_hydrogen: bool = True,
    add_hydrogen_pH: float = 7.0,
    remove_heterogens: bool = True,
    keep_water: bool = False,
    keep_chain_ids: Optional[List[str]] = None
):
    """
    Fix protein with `pdbfixer`

    Parameters
    ----------
    in_pdb: os.PathLike
        input pdbfile
    out_pdb: os.PathLike
        output pdbfile
    add_hydrogen: bool
        whether to add missing hydrogens
    add_hydrogen_pH: float
        pH when adding hydrogens
    remove_heterogens: bool
        whether to remove heterogens
    keep_water: bool
        whether to keep water
    keep_chain_ids: List of str, optional
        Chain ids to keep. If None, will keep all the chains
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(in_pdb)
    chainIds = set(c.id for c in fixer.topology.chains())
    if keep_chain_ids:
        remove_chain_ids = [c for c in chainIds if c not in keep_chain_ids]
    else:
        remove_chain_ids = []
    fixer.removeChains(chainIds=remove_chain_ids)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    if add_hydrogen:
        fixer.addMissingHydrogens(add_hydrogen_pH)
    if remove_heterogens:
        fixer.removeHeterogens(keepWater=keep_water)
    with open(out_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    

def fix_hydrogen(in_pdb: os.PathLike, out_pdb: Optional[os.PathLike] = None):
    """
    Fix hydrogen atom types using pdb4amber program
    """
    in_pdb = Path(in_pdb).resolve()
    if out_pdb is None:
        out_pdb = in_pdb.with_name(f"{in_pdb.stem}_processed.pdb")
    tmp_pdb = in_pdb.with_name(f"{in_pdb.stem}_no_h.pdb")
    pdb4amber = find_executable("pdb4amber")
    # remove hydrogen first
    run_command([pdb4amber, "-i", in_pdb, '-o', tmp_pdb, "-y"])
    # use reduce to add hydrogen
    run_command([pdb4amber, "-i", tmp_pdb, "-o", out_pdb, '--reduce'])
    # fix wrong hydrongen name in CYM
    fix_cym(out_pdb, out_pdb)
    return out_pdb


def fix_cym(in_pdb: os.PathLike, out_pdb: os.PathLike):
    """
    Fix wrong hydrogen name in CYM
    """
    in_pdb = Path(in_pdb).resolve()
    lines = []
    with open(in_pdb, 'r') as f:
        for line in f:
            if "CYM" in line:
                line = line.replace("HN1", "H  ")
                line = line.replace("HB1", "HB3")
            lines.append(line)
    with open(out_pdb, 'w') as f:
        f.write("".join(lines))


def run_tleap(
    pdb: os.PathLike,
    prmtop: os.PathLike = "protein.prmtop",
    inpcrd: os.PathLike = "protein.inpcrd",
    protein_ff: str = "ff14SB",
    water_ff: str = "tip3p"
):
    """
    Run tleap to parametrize protein

    Parameter
    ---------
    pdb: os.PathLike
        protein pdb file
    prmtop: os.PathLike
        output amber parameter file
    inpcrd: os.PathLike
        output amber input coordinate file
    protein_ff: str
        protein forcefield
    water_ff: str
        water forcefield
    """
    tleap = find_executable("tleap")
    with open(Path(__file__).with_name("leap.in"), 'r') as f:
        leap_in = f.read()
    
    leap_in = leap_in.format(
        pdb=file_abspath(pdb), 
        prmtop=prmtop, 
        inpcrd=inpcrd, 
        protein_ff=PROTEIN_FF_KEYS[protein_ff], 
        water_ff=water_ff
    )

    with set_directory(Path(pdb).parent) as wdir:
        with open("leap.in", 'w') as f:
            f.write(leap_in)
        return_code, err, out = run_command([tleap, '-f', "leap.in"], raise_error=True)
