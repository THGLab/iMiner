import os, sys
from pathlib import Path
from io import StringIO
import xml.etree.ElementTree as ET
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import logging
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plip.structure.preparation import PDBComplex, logger as PLIP_LOGGER
from plip.exchange.report import StructureReport
from plip.basic import config as PLIP_CONFIG

PLIP_LOGGER.setLevel(logging.ERROR)


def analyze_single_frame(
    pdbpath: os.PathLike, 
    add_hydrogen: bool = False,
    resnr_renum: Optional[Dict[int, int]] = None
) -> Dict[str, int]:
    """
    Analyze a single ligand-complex structure

    Parameters
    ----------
    pdbpath: os.PathLike
        Path to the pdbfile to be analyzed
    add_hydrogen: bool
        Whether to add hydrogen to the pdb structure. Default is False
    
    Return
    ------
    interact_count_frame: Dict[str, int]
        A dict with interaction type as the key, and the number of interaction as value
        The key is in the format "{name}/{restype}/{resnr}/{chain}". For example,
        'hydrophobic_interaction/ALA/123/A'
    """
    
    if add_hydrogen:
        PLIP_CONFIG.NOHYDRO = False
    else:
        PLIP_CONFIG.NOHYDRO = True
        
    pdb = PDBComplex()
    pdb.load_pdb(str(pdbpath))
    pdb.analyze()
    report = StructureReport(pdb)

    tmp = sys.stdout
    xmlstr = StringIO()
    sys.stdout = xmlstr
    report.write_xml(True)
    sys.stdout = tmp
    xmlstr.seek(0)

    xmlobj = ET.fromstring(xmlstr.read())

    binding_sites = xmlobj.findall("./bindingsite")
    bs = [bs for bs in binding_sites if bs.findall("identifiers/longname")[0].text == "MOL"][0]
    itypes = bs.findall("interactions/")
    interact_count_frame = {}
    for itype in itypes:
        for item in itype:
            name = item.tag
            restype = item.find("restype").text
            resnr = item.find("resnr").text
            if resnr_renum is not None:
                resnr = resnr_renum[int(resnr)]
            chain = item.find("reschain").text
            sig = f"{name}/{restype}/{resnr}/{chain}"
            cnt = interact_count_frame.get(sig, 0)
            if cnt == 0:
                interact_count_frame.update({sig: cnt+1})

    return interact_count_frame


def analyze_multiple_frames(
    pdbpaths: List[os.PathLike], 
    f_csv: os.PathLike = "interaction.csv",
    add_hydrogen: bool = False,
    resnr_renum: Optional[Dict[int, int]] = None,
    use_mpi: bool = True, 
    chunksize: int = 1
) -> pd.DataFrame:
    """
    Analyze multiple frames and write results to a csv file
    """
    interacts_frames = []
    analyze_single_frame_func = partial(
        analyze_single_frame, 
        add_hydrogen=add_hydrogen,
        resnr_renum=resnr_renum
    )   
    if not use_mpi:
        for pdbpath in tqdm(pdbpaths):
            frame_data = analyze_single_frame_func(pdbpath)
            interacts_frames.append(frame_data)
    else:
        import multiprocessing as mp
        import math
        num_cores = mp.cpu_count()
        chunksize = math.ceil(len(pdbpaths) / num_cores) if chunksize == "auto" else int(chunksize)
        pool = mp.Pool(processes=num_cores)
        for frame_data in tqdm(
            pool.imap(func=analyze_single_frame_func, iterable=pdbpaths, chunksize=chunksize),
            total=len(pdbpaths)
        ):
            interacts_frames.append(frame_data)
                
    interacts = {}
    for frame_data in interacts_frames:
        for sig, cnt in frame_data.items():
            val = interacts.get(sig, 0)
            interacts.update({sig: val+cnt})
            
    interact_df = []
    for sig, val in interacts.items():
        name, resname, resnr, chain = tuple(sig.split("/"))
        ratio = val / len(pdbpaths)
        interact_df.append({
            "interaction": name,
            "resname": resname,
            "resnr": resnr,
            "chain": chain,
            "ratio": ratio
        })
    interact_df = pd.DataFrame(interact_df)
    interact_df.to_csv(str(f_csv))
    return interact_df


def plot_interact(f_csv, threshold=0.1, title=None):
    df = pd.read_csv(str(f_csv), index_col=0)
    newdf = pd.DataFrame()
    df = df.sort_values(['resnr', 'chain'])
    df.index = list(range(df.shape[0]))
    for i in range(df.shape[0]):
        resname = df.loc[i, 'resname']
        chain = df.loc[i, 'chain']
        resnr = df.loc[i, 'resnr']
        restag = f"{resname}{resnr}{chain}"
        itype = df.loc[i, 'interaction']
        newdf.loc[restag, itype] = df.loc[i, 'ratio']

    newdf = newdf.fillna(0.0)
    newdf[newdf < threshold] = 0.0
    newdf = newdf.loc[(newdf != 0).any(axis=1), :]
    ind = np.arange(newdf.shape[0])
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(15, 5))
    bottom = np.zeros(newdf.shape[0])
    interacts = sorted(list(newdf.columns))
    
    color_map = {
        "hydrogen_bond": "C0",
        "hydrophobic_interaction": "C1",
        "pi_stack": "C2",
        "salt_bridge": "C3",
        "pi_cation_interaction": "C4",
        "halogen_bond": "C5"
    }
    
    for interact in interacts:
        ax.bar(ind, newdf[interact], label=interact, bottom=bottom, color=color_map[interact])
        bottom += newdf[interact]
        for x, y, value in zip(ind, bottom, newdf[interact]):
            ax.text(x, y, f"{value:.2f}", ha='center', va='bottom')
    
    ax.set_xticks(ind)
    ax.set_xticklabels(list(newdf.index), rotation=50)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(0, max(1, np.max(bottom) * 1.05))
    return fig, ax