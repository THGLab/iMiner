from typing import Union, List
from pathlib import Path
import warnings
from iMiner.cmd import find_executable, run_command


def run_acpype(input: Union[str, Path, None] = None,
               basename: str = "MOL",
               charge_method: str = "bcc",
               atom_type: str = "gaff2",
               net_charge: Union[int, str] = "auto",
               args: Union[None, List[str]] = None):
    """
    Run acpype

    Parameters
    ----------
    input : str or Path or None
        input file name with extension that `acpype -i` support
    basename : str
        a basename for the project, `acpype -b` option
    charge method : str
        gas, bcc (default), user (user's charges in mol2 file)
    atom_type : str
        atom type, can be 'gaff', 'gaff2', 'amber' (AMBER14SB) or 'amber2' (AMBER14SB + GAFF2), default is gaff2
    net_charge : int or "guess"
        net molecular charge, default is "auto". If "auto" and input is mol/sdf/mol2, iMiner will compute the net charge
        based on input file using RDKit. If "guess", acpype will guess a charge.
    args : List[str] or None
        arguments used to run acpype. if `args` is not None, all other arguments are ignored
    """
    acpype = find_executable("acpype")
    if args is not None:
        cmd = [acpype] + args
    else:
        assert input is not None, "Input is None."
        cmd = [acpype, "-i", str(input), "-b", basename, "-c", charge_method, "-a", atom_type]            
        if net_charge == "auto":
            suffix = Path(input).suffix
            from rdkit import Chem
            if suffix == ".mol":
                mol = Chem.MolFromMolFile(input, removeHs=False)
            elif suffix == ".sdf":
                mol = Chem.SDMolSupplier(input, removeHs=False)[0]
            elif suffix == ".mol2":
                mol = Chem.MolFromMol2File(input, removeHs=False)
            else:
                mol = None
            if mol:
                net_charge = sum([at.GetFormalCharge() for at in mol.GetAtoms()])
            else:
                warnings.warn(f"Fail to parse input file {Path(input).resolve()}. iMiner will let acpype to determine net charge")
        if not isinstance(net_charge, str):
            cmd = cmd.extend(["-n", str(net_charge)])
    return_code, out, err = run_command(cmd, raise_error=True)
    return 