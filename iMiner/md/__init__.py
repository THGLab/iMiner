import os
from typing import Dict, List, Optional

 
def parse_index_file(fname: os.PathLike) -> Dict[str, List[int]]:
    """
    Parse GROMACS index file
    
    Parameters
    ----------
    fname: os.PathLike
        Path to GROMACS index file (.ndx)
    
    Return
    ------
    groupdict: Dict[str, List[int]]
        Dict with groups' names and corresponding atom indices
    """
    groupdict = {}
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith("["):
                name = line[1:].strip(']\n').strip()
                groupdict[name] = list()
                continue
            
            for x in line.strip().split():
                groupdict[name].append(int(x))
    return groupdict


def mk_index_file(groupdict: Dict[str, List[int]], fname: os.PathLike, overwrite: bool = True):
    """
    Make a GROMACS index file (.ndx)
    
    Parameters
    ----------
    groupdict: Dict[str, List[int]]
        Dict with groups' names and corresponding atom indices
    fname: os.PathLike
        Path to GROMACS index file (.ndx)
    overwrite: bool
        Whether to overwrite and existing file
    """
    if os.path.isfile(fname) and (not overwrite):
        raise FileExistsError(f"{fname} exists")
    
    num_per_line = 15
    with open(fname, 'w') as f:
        for name, indices in groupdict.items():
            f.write(f'[ {name} ]\n')
            for i in range(0, len(indices), num_per_line):
                f.write(" ".join(f'{x:>4}' for x in indices[i: i + num_per_line]))
                f.write('\n')