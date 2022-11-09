"""
This example is only temporary and the API is subject to change
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from iMiner.docking.autodock import LigandPDBQT


def convert_pdbqt_to_sdf(template_sdf, template_pdbqt, input_pdbqt, output_sdf):
    in_pdbqt = LigandPDBQT(template=template_sdf)
    out_pdbqt = LigandPDBQT(template=template_sdf)

    in_pdbqt.read_file(template_pdbqt)
    out_pdbqt.read_file(input_pdbqt)

    out_pdbqt.write_sdf(output_sdf, mapping=in_pdbqt.get_mapping(), reset_h=True)


if __name__ == "__main__":
    convert_pdbqt_to_sdf(
        "ligand.sdf",
        "ligand.pdbqt",
        "ligand_out_ligand_01.pdbqt",
        "ligand_out_ligand_01.sdf"
    )