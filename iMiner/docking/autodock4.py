"""
Author: Oliver Sun
Date Created: Oct 28 2022

Implementation of AutoDock4
"""

from iMiner.docking.base import BaseDocking
from pathlib import Path
import shutil
import os
import subprocess


class AD4Docking(BaseDocking):
    """
    Run AutoDock4 with predefined binding pocket for ligands.
    """
    def __init__(self, protein_pdb, docking_box):
        """
        Initialize autodock4 with a protein and a docking box

        @param protein_pdb: str, path to the protein pdb file
        @param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        """
        super().__init__(protein_pdb, docking_box)
        protein_path = Path(protein_pdb)
        protein_folder = protein_path.parent
        self.protein_name = protein_path.stem
        self.ad4dir = os.mkdiros.path.join(protein_folder, "{}-ad4".format(self.protein_name))
        self.config_path = self.write_docking_config(protein_folder, docking_box)
    
    def process_protein(self, protein_pdb):
        """
        Using the AutoDock Tools Suite to prepare the protein and convert it to a pdbqt file.

        @param protein_pdb: str, path to protein pdb file
        """



        raise NotImplementedError()
    
    def write_docking_config(path_to_write, docking_box):
        """
        write autodock4 configuration file with the given docking box infomation

        @param path_to_write: str, path to write the config file in
        @param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        """
        xmin, ymin, zmin, xmax, ymax, zmax = docking_box
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center_z = (zmin + zmax) / 2
        size_x = xmax - xmin
        size_y = ymax - ymin
        size_z = zmax - zmin

        fp = os.path.join(path_to_write, "config.txt")
        with open(fp, "w") as f:
            f.write("center_x = " + str(center_x) + "\n")
            f.write("center_y = " + str(center_y) + "\n")
            f.write("center_z = " + str(center_z) + "\n")
            f.write("size_x = " + str(size_x) + "\n")
            f.write("size_y = " + str(size_y) + "\n")
            f.write("size_z = " + str(size_z) + "\n")
        
        return fp

        
        
    
