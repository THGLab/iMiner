import os
from typing import Tuple

from iMiner.cmd import run_command, find_executable, set_directory
from iMiner.md import parse_index_file, mk_index_file
from iMiner.log import init_logger


LOGGER = init_logger("iMiner.log")


try:
    MPIRUN_EXEC = find_executable("mpirun")
except:
    MPIRUN_EXEC = None
    

def preprocess_index_file(
    f_ndx_ori: os.PathLike,
    f_ndx_new: os.PathLike,
    overwrite: bool = True,
    ligand_name: str = "MOL"
) -> Tuple[int, int]:
    """
    Pre-process index file
    
    Combine protein with ions except [Na+] and [Cl-] and get group id of 
    receptor and ligand
    
    Parameters
    ----------
    f_ndx_ori: os.PathLike
        Path to original index file
    f_ndx_new: os.PathLike
        Path to pre-processed index file
    overwrite: bool
        Whether to overwrite index file, default True
    ligand_name: str
        Group name of ligand, default "MOL"
    
    Return
    ------
    (r_grp_idx, l_grp_idx): Tuple[int, int]
        Group index of receptor and ligand
    """
    receptor_name = 'Receptor_GBSA'
    group_dict = parse_index_file(f_ndx_ori)
    group_dict[receptor_name] = group['Protein'].copy()
    for ndx in group_dict.get('Ion', []):
        if ndx in group_dict.get('NA', []):
            continue
        if ndx in group_dict.get('CL', []):
            continue
        group_dict[receptor_name].append(ndx)
    
    mk_index_file(group_dict, f_ndx_new, overwrite)
    keys = list(group_dict.keys())
    r_grp_idx = keys.index(receptor_name)
    l_grp_idx = keys.index(ligand_name)
    return r_grp_idx, l_grp_idx

    
class GBSA:
    def __init__(self):
        pass
    
    def set_params(
        self,
        complex_file: os.PathLike, 
        traj_file: os.PathLike, 
        topol_file: os.PathLike, 
        index_file: os.PathLike, 
        pbsa_params: Optional[Dict[str, Any]] = None, 
        mmpbsa_in_file: Optional[os.PathLike] = None, 
        num_threads: int = 1
    ):
        """
        Set GBSA Parameters
        
        Parameters
        ----------
        complex_file: os.PathLike
            Path to complex strcture file, argument '-cs' in gmx_MMPBSA  
        traj_file: os.PathLike
            Path to MD trajectory file, argument '-ct' in gmx_MMPBSA
        topol_file: os.PathLike
            Path to complex topology, argument '-cp' in gmx_MMPBSA
        index_file: os.PathLike
            Path to index file, argument '-ci' in gmx_MMPBSA
        mmpbsa_params: Dict[str, Any] or None
            Parameters to control gmx_MMPBSA calculation, default None
            See more, please refer to https://valdes-tresanco-ms.github.io/gmx_MMPBSA/v1.5.7/input_file/  
        mmpbsa_in_file: os.PathLike or None
            Path to gmx_MMPBSA input file, default None
            If None, iMiner will use default parameters and parameters specified in `mmpbsa_params`
        num_threads: int
            Number of cores in mpirun, default 1
        """
        
    
    def run_gmx_mmpbsa_exec(self,
    
    
    