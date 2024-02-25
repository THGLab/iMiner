import os
from pathlib import Path
import shutil
from typing import Optional
from iMiner.cmd import run_command, find_executable, set_directory


WATERBOX_IONPARM_NAMES = {
    "tip3p": ("TIP3PBOX", "ionsjc_tip3p")
}


def run_tleap(inp: os.PathLike):
    tleap = find_executable("tleap")
    code, _, _ = run_command([tleap, '-f', inp], raise_error=False)
    if code != 0:
        raise RuntimeError(f"TLEaP fails. See log: {Path.cwd() / 'leap.log'}")


def get_waterbox_ions_names(water_ff: str):
    """
    Get name of waterbox and ions parameter in tleap
    """
    if water_ff in WATERBOX_IONPARM_NAMES:
        return WATERBOX_IONPARM_NAMES[water_ff]
    else:
        raise NotImplementedError(f"Water force field not supported: {water_ff}")


def get_num_ions(volume: float, ionic_strength: float):
    raise NotImplementedError()


def create_solvated_box(
    wdir: os.PathLike,
    ligand_mol2: Optional[os.PathLike] = None,
    ligand_lib: Optional[os.PathLike] = None,
    ligand_frcmod: Optional[os.PathLike] = None,
    ligand_ff: str = "gaff2",
    ligand_name: str = "ligand",
    protein_pdb: Optional[os.PathLike] = None,
    protein_ff: str = "ff14SB",
    protein_name: str = 'protein',
    water_ff: str = "tip3p",
    buffer: float = 10.0,
    ionic_strength: float = 0.0,
    box_resize: float = 0.75
):
    # copy files
    wdir = Path(wdir).resolve()

    if (ligand_mol2 is not None) and (protein_pdb is not None):
        system_name = "complex"
    elif (protein_pdb is not None):
        system_name = protein_name
    elif (ligand_mol2 is not None):
        system_name = ligand_name
    else:
        raise RuntimeError("Must specify either ligand or protein")

    if ligand_mol2 is not None:
        shutil.copyfile(ligand_mol2, wdir / f"{ligand_name}.mol2")
        shutil.copyfile(ligand_frcmod, wdir / f"{ligand_name}.frcmod")
        shutil.copyfile(ligand_lib, wdir / f"{ligand_name}.lib")
    if protein_pdb is not None:
        shutil.copyfile(protein_pdb, wdir / f"{protein_name}.pdb")

    waterbox, ionparm = get_waterbox_ions_names(water_ff)
    scripts = [
        f"source leaprc.water.{water_ff}",
        f"loadAmberParams frcmod.{ionparm}",
        f"source leaprc.protein.{protein_ff}",
    ]

    if ligand_mol2 is not None:
        scripts += [
            f"source leaprc.{ligand_ff}",
            f"loadoff {ligand_name}.lib",
            f"loadamberparams {ligand_name}.frcmod",
            f"{ligand_name} = loadmol2 {ligand_name}.mol2"
        ]
    if protein_pdb is not None:
        scripts.append(f"{protein_name} = loadpdb {protein_name}.pdb")
    
    if system_name == "complex":
        scripts.append((
            f"{system_name} = combine "
            "{" 
            f" {ligand_name} {protein_name} "
            "}"
        ))
    
    scripts += [
        # create proteins in solution
        f"solvatebox {system_name} {waterbox} {buffer} {box_resize}",

        # Neutralize
        f"addions {system_name} Na+ 0",
        f"addions {system_name} Cl- 0",
    ]
    
    if ionic_strength != 0.0:
        raise NotImplementedError("Adding ions according to ionic strength is not supported yet.")
        # TODO: this number may be not correct. calcualte the exact number with volume
        # num_ions = 12
        # scripts.append(f"addionsrand {system_name} Na+ {num_ions}")
        # scripts.append(f"addionsrand {system_name} Na+ {num_ions}")
    
    scripts += [
        f"savepdb {system_name} {system_name}_solvated.pdb",
        f"saveamberparm {system_name} {system_name}_solvated.prmtop {system_name}_solvated.inpcrd",
        "quit"
    ]
    
    with set_directory(wdir):
        if os.path.isfile('leap.log'): os.remove('leap.log')
        with open("leap.in", 'w') as f:
            f.write('\n'.join(scripts))
        run_tleap("leap.in")


def create_fep_simulation_box(
    wdir: os.PathLike,
    ligand_mol2: Optional[os.PathLike] = None,
    ligand_lib: Optional[os.PathLike] = None,
    ligand_frcmod: Optional[os.PathLike] = None,
    ligand_ff: str = "gaff2",
    ligand_name: str = "ligand",
    protein_pdb: Optional[os.PathLike] = None,
    protein_ff: str = "ff14SB",
    protein_name: str = 'protein',
    water_ff: str = "tip3p",
    buffer: float = 10.0,
    ionic_strength: float = 0.0,
    box_resize: float = 0.75
):
    """
    Create solvated box for ligands and protein-ligand complex
    """