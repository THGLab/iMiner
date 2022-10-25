import os
from typing import Optional


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


# split_top("../protein/protein.amb2gmx/protein_GMX.top", "protein.atp", "protein.itp")
# split_top("MOL.acpype/MOL_GMX.itp", "MOL.atp", "MOL.itp")
# merge_gro("../protein/protein.amb2gmx/protein_GMX.gro", "MOL.gro", "complex.gro")
