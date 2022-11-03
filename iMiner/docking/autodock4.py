"""
Author: Oliver Sun
Date Created: Oct 28 2022

Automate the use of AutoDock4-GPU Docking Software 
"""

from iMiner.docking.autodock import AutoDockBaseDocking
from pathlib import Path
import shutil
import os
import re
import subprocess

supported_atypes = set(['HD', 'C', 'A', 'N', 'NA', 'OA', 'F', 'P', 'SA', 'S',
                    'Cl', 'Br', 'I', 'Mg', 'Ca', 'Mn', 'Fe', 'Zn'])

gpf = """npts NPTS_X NPTS_Y NPTS_Z
gridfld PREFIX.maps.fld
spacing 0.375
receptor_types RECTYPES
ligand_types HD C A N NA OA F P SA S Cl Br I
receptor REC
gridcenter CENTER_X CENTER_Y CENTER_Z
smooth 0.5
map         PREFIX.HD.map
map         PREFIX.C.map
map         PREFIX.A.map
map         PREFIX.N.map
map         PREFIX.NA.map
map         PREFIX.OA.map
map         PREFIX.F.map
map         PREFIX.P.map
map         PREFIX.SA.map
map         PREFIX.S.map
map         PREFIX.Cl.map
map         PREFIX.Br.map
map         PREFIX.I.map
elecmap     PREFIX.e.map
dsolvmap    PREFIX.d.map
dielectric -0.1465
"""

class AD4Docking(AutoDockBaseDocking):
    """
    Run AutoDock4 with predefined binding pocket for ligands.
    """
    def __init__(self, protein_pdb, protein_pdbqt, docking_box):
        """
        Initialize autodock4 with a protein and a docking box

        @param protein_pdb: str, path to the protein pdb file
        @param protein_pdbqt: str, path to the protein pdbqt file
        @param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        """
        super().__init__(protein_pdb, docking_box)
        self.convert_pdb_to_pdbqt(protein_pdb, protein_pdbqt)
        self.protein_path = Path(protein_pdbqt).resolve()
        self.protein_folder = self.protein_path.parent
        self.protein_name = self.protein_path.stem
        self.ad4dir = os.mkdir(os.path.join(self.protein_folder, "{}-ad4".format(self.protein_name)))

    def write_gpf_file(self, docking_box, spacing = 0.375):
        """
        write autodock4 configuration file with the given docking box infomation

        @param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        @param spacing: float, spacing of the protein grid, default 0.375 angstroms

        @return: str, filepath to the successfully written gpf
        """
        # convert the box information into gpf-required information
        xmin, ymin, zmin, xmax, ymax, zmax = docking_box
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        center_z = (zmin + zmax) / 2.0
        npts_x = 2 * int((xmax - xmin) / (2*spacing))
        npts_y = 2 * int((ymax - ymin) / (2*spacing))
        npts_z = 2 * int((zmax - zmin) / (2*spacing))

        # extract the receptor types from the pdbqt file
        command = 'cut -c 77-79 %s | sort -u' % self.protein_path
        try:
            out = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, text = True)
            out_types = out.communicate()[0]
            pdbqt_types = re.split("\s+", out_types.strip())
            rectypes = ' '.join(list(set(pdbqt_types) & supported_atypes))
        except subprocess.CalledProcessError as e:
            return e.output

        # fill in the necessary information
        gpf = gpf.replace('RECTYPES',   rectypes)
        gpf = gpf.replace('PREFIX',     self.protein_name)
        gpf = gpf.replace('REC',        self.protein_path)
        gpf = gpf.replace('NPTS_X',     '%d' % npts_x)
        gpf = gpf.replace('NPTS_Y',     '%d' % npts_y)
        gpf = gpf.replace('NPTS_Z',     '%d' % npts_z)
        gpf = gpf.replace('CENTER_X',   '%.3f' % center_x)
        gpf = gpf.replace('CENTER_Y',   '%.3f' % center_y)
        gpf = gpf.replace('CENTER_Z',   '%.3f' % center_z)
    
        # write everything to a config gpf file for autogrid
        gpf_path = os.path.join(self.ad4dir, "{}.gpf".format(self.protein_name))
        with open(gpf_path, "w") as f:
            f.write(gpf)
        
        return gpf_path

        
        
    
