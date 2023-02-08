import sys
from pathlib import Path
from io import StringIO
import xml.etree.ElementTree as ET
from tqdm import tqdm

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from plip.structure.preparation import PDBComplex
from plip.exchange.report import StructureReport

from iMiner.cmd import run_command


def analyze_single_frame(pdbpath, mol_name="MOL"):
    pdb = PDBComplex()
    pdb.load_pdb(pdbpath)
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
    try:
        bs = [bs for bs in binding_sites if bs.findall("identifiers/longname")[0].text == mol_name][0]
        itypes = bs.findall("interactions/")
    except IndexError: 
        itypes = []
        bs = [bs for bs in binding_sites]
        #print(pdbpath)
        #print([bs.findall("identifiers/longname")[0].text for bs in binding_sites])
        #with open(pdbpath, "r") as f:
        #    c = f.read()
        #    print("\n".join(c.split("TER\n")[1:]))
        for site in bs:
            itypes += list(site.findall("interactions/"))
            
    interact_count_frame = {}
    for itype in itypes:
        for item in itype:
            name = item.tag
            restype = item.find("restype").text
            resnr = item.find("resnr").text
            chain = item.find("reschain").text
            sig = f"{name}/{restype}/{resnr}/{chain}"
            cnt = interact_count_frame.get(sig, 0)
            if cnt == 0:
                interact_count_frame.update({sig: cnt+1})

    return interact_count_frame


def analyze_multiple_frames(pdbpaths):
    interacts = {}
    for pdbpath in tqdm(pdbpaths):
        frame_data = analyze_single_frame(pdbpath)
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
    return interact_df


def analyze_gmx_traj(ref, traj, dt=200):
    trajdir = Path(traj).parent / "traj"
    trajdir.mkdir(exist_ok=True)
    run_command(
        f"gmx trjconv -s {Path(ref).resolve()} -f {Path(traj).resolve()} -o {Path(trajdir).resolve() / 'prod.pdb'} -sep -dt {dt}",
        raise_error=True,
        input="0"
    )
    pdbs = []
    for pdb in trajdir.glob('*.pdb'):
        with open(pdb, 'r') as f:
            contents = f.readlines()
        for i, line in enumerate(contents):
            if line.startswith("MODEL"):
                contents[i] = "MODEL        1\n"
        with open(pdb, 'w') as f:
            f.write("".join(contents))
        pdbs.append(str(pdb))
    df = analyze_multiple_frames(pdbs)
    return df


def plot_interact(df, threshold=0.1, title=None, resnr_renum=None):
    newdf = pd.DataFrame()
    df = df.sort_values(['resnr', 'chain'])
    df.index = list(range(df.shape[0]))
    for i in range(df.shape[0]):
        resname = df.loc[i, 'resname']
        if resnr_renum is not None:
            resnr = resnr_renum[df.loc[i, 'resnr']]
        else:
            resnr = df.loc[i, 'resnr']
        chain = df.loc[i, 'chain']
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
    
    ax.set_xticks(ind)
    ax.set_xticklabels(list(newdf.index), rotation=50)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(0, max(1, np.max(bottom) * 1.05))
    return fig