import os
import shutil
from pathlib import Path
from typing import Optional

from iMiner.cmd import set_directory


def split_top(top: os.PathLike, atp: os.PathLike, itp: os.PathLike):
    atpf = open(atp, 'w')
    itpf = open(itp, 'w')
    with open(top, 'r') as f:
        read_atp = False
        read_itp = False
        for line in f:
            if line.startswith("[ atomtypes ]"):
                read_atp = True
                read_itp = False
            elif line.startswith("[ moleculetype ]"):
                read_atp = False
                read_itp = True
            elif line.startswith("[ system ]"):
                read_atp = False
                read_itp = False
            if read_atp:
                atpf.write(line)
            if read_itp:
                itpf.write(line)
    atpf.close()
    itpf.close()


def merge_gro(pgro: os.PathLike, lgro: os.PathLike, cgro: os.PathLike):
    out = open(cgro, 'w')
    pgrof = open(pgro, 'r')
    lgrof = open(lgro, 'r')
    out.write("Complex\n")
    pgrof.readline()
    lgrof.readline()
    natoms_prot = int(pgrof.readline().strip())
    natoms_lig = int(lgrof.readline().strip())
    numatoms = natoms_prot + natoms_lig
    out.write(str(numatoms) + '\n')
    ligstr = [line for line in lgrof.readlines()][:-1]
    protstr = [line for line in pgrof.readlines()]
    out.write("".join(ligstr))
    #out.write("\n")
    out.write("".join(protstr))
    out.close()
    pgrof.close()
    lgrof.close()


def make_complex(
    protein_top: os.PathLike, 
    ligand_top: os.PathLike, 
    protein_gro: os.PathLike,
    ligand_gro: os.PathLike,
    protein_posre: os.PathLike,
    ligand_posre: os.PathLike,
    complex_dir: os.PathLike,
    complex_top_name: str = "topol.top",
    complex_gro_name: str = "complex.gro",
):
    """
    Make protein-ligand complex topology and coordinate
    """
    protein_top = Path(protein_top).resolve()
    ligand_top = Path(ligand_top).resolve()
    protein_gro = Path(protein_gro).resolve()
    ligand_gro = Path(ligand_gro).resolve()
    protein_posre = Path(protein_posre).resolve()
    ligand_posre = Path(ligand_posre).resolve()
    with set_directory(complex_dir):
        split_top(protein_top, "protein.atp", "protein.itp")
        split_top(ligand_top, "MOL.atp", "MOL.itp")
        merge_gro(protein_gro, ligand_gro, complex_gro_name)
        shutil.copyfile(protein_posre, "posre_protein.itp")
        shutil.copyfile(ligand_posre, "posre_MOL.itp")
        shutil.copyfile(Path(__file__).with_name("topol_template.top"), complex_top_name)
        shutil.copyfile(Path(__file__).with_name("water_and_ions.atp"), "water_and_ions.atp")


def make_solvated(
    ligand_top: os.PathLike,
    ligand_gro: os.PathLike,
    ligand_posre: os.PathLike,
    wdir: os.PathLike,
    solvated_top_name: str = "topol.top"
):
    """
    Make solvated system
    """
    ligand_top = Path(ligand_top).resolve()
    ligand_gro = Path(ligand_gro).resolve()
    ligand_posre = Path(ligand_posre).resolve()
    with set_directory(wdir):
        split_top(ligand_top, "MOL.atp", "MOL.itp")
        shutil.copyfile(ligand_posre, "posre_MOL.itp")
        shutil.copyfile(ligand_gro, "ligand.gro")
        shutil.copyfile(Path(__file__).with_name("solvated_template.top"), solvated_top_name)
        shutil.copyfile(Path(__file__).with_name("water_and_ions.atp"), "water_and_ions.atp")
