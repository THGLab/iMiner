import os
import shutil
from pathlib import Path
from typing import List

from iMiner.cmd import set_directory, find_executable, run_command


def split_top(top: os.PathLike, atp: os.PathLike, itp: os.PathLike):
    atpf = []
    itpf: List[List] = []
    mols = {}
    with open(top, 'r') as f:
        read_atp = False
        read_itp = False
        read_molecule = False
        for line in f:
            if line.startswith("[ atomtypes ]"):
                read_atp = True
                read_itp = False
            elif line.startswith("[ moleculetype ]"):
                read_atp = False
                read_itp = True
                itpf.append([])
            elif line.startswith("[ system ]"):
                read_atp = False
                read_itp = False
            elif line.startswith("[ molecules ]"):
                read_molecule = True
                continue
            if read_atp: 
                atpf.append(line)
            if read_itp: 
                itpf[-1].append(line)
            if read_molecule:
                if not line.startswith(';'):
                    tmp = line.strip().split()
                    name, count = tmp[0], int(tmp[1])
                    # change water name to SOL, compatitable with GROMACS
                    if name == "WAT": name = "SOL" 
                    mols[name] = count 
    
    with open(atp, 'w') as f:
        f.write(''.join(atpf))
    
    with open(itp, 'w') as f:
        for ls in itpf:
            is_water = any([("WAT" in x) or ("SOL" in x) for x in ls])
            if not is_water:
                f.write(''.join(ls))
    
    return mols

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
        prot_info = split_top(protein_top, "protein.atp", "protein.itp")
        lig_info = split_top(ligand_top, "MOL.atp", "MOL.itp")
        if len(lig_info) == 0: lig_info = {"MOL": 1} # default number of lig to 1 with name "MOL"
        
        merge_gro(protein_gro, ligand_gro, complex_gro_name)
        
        # Restraints
        run_command([find_executable(['gmx_mpi', 'gmx']), 'genrestr', '-f', protein_gro, '-o', 'posre_protein.itp'], input="Protein-H")
        shutil.copyfile(ligand_posre, "posre_MOL.itp")

        with open(Path(__file__).with_name("complex_template.top")) as f:
            tstr = f.read()
        
        if "SOL" in prot_info:
            tstr.replace('#include "water.atp"', "")
        
        # add [ molecules ] info
        molstr = ['[ molecules ]\n']
        for k, v in lig_info.items():
            molstr.append(f" {k:<8}{v:<8}\n")
        for k, v in prot_info.items():
            molstr.append(f" {k:<8}{v:<8}\n")
        molstr = ''.join(molstr)
        tstr += molstr

        with open(complex_top_name, 'w') as f:
            f.write(tstr)
        
        # copy water and ions atom type defs
        shutil.copyfile(Path(__file__).with_name("water.atp"), "water.atp")
        shutil.copyfile(Path(__file__).with_name("ions.atp"), "ions.atp")


def make_solvated(
    ligand_top: os.PathLike,
    ligand_gro: os.PathLike,
    ligand_posre: os.PathLike,
    wdir: os.PathLike,
    solvated_top_name: str = "topol.top",
    solvated_gro_name: str = "ligand.gro"
):
    """
    Make solvated system
    """
    ligand_top = Path(ligand_top).resolve()
    ligand_gro = Path(ligand_gro).resolve()
    ligand_posre = Path(ligand_posre).resolve()
    with set_directory(wdir):
        lig_info = split_top(ligand_top, "MOL.atp", "MOL.itp")
        if len(lig_info) == 0: lig_info = {"MOL": 1} # default number of lig to 1 with name "MOL"
        shutil.copyfile(ligand_posre, "posre_MOL.itp")
        shutil.copyfile(ligand_gro, solvated_gro_name)

        with open(Path(__file__).with_name("solvated_template.top")) as f:
            tstr = f.read()
        
        # add [ molecules ] info
        molstr = ['[ molecules ]\n']
        for k, v in lig_info.items():
            molstr.append(f" {k:<8}{v:<8}\n")

        molstr = ''.join(molstr)
        tstr += molstr

        with open(solvated_top_name, 'w') as f:
            f.write(tstr)
        
        # copy water and ions atom type defs
        shutil.copyfile(Path(__file__).with_name("water.atp"), "water.atp")
        shutil.copyfile(Path(__file__).with_name("ions.atp"), "ions.atp")
