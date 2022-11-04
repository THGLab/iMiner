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

autogrid_path = '/global/home/groups/co_armada2/local/ADFRsuite/bin/autogrid4'

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
    def __init__(self, protein_pdb, ligand_fd, protein_ad4_fd, docking_box, name = None):
        """
        Initialize autodock4 with a protein and a docking box

        @param protein_pdb: str, path to the protein pdb file
        @param ligand_fd: str, path to the folder that stores all ligands.
        @param protein_ad4_fd: str, path to the autodock4 prepared protein folder
        @param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        @param name: str or none, name of the protein 
        """
        super().__init__(protein_pdb, docking_box)
        self.ad4dir = Path(protein_ad4_fd).resolve()
        if self.ad4dir.exists() and self.ad4dir.is_dir():
            shutil.rmtree(self.ad4dir)  
        self.ad4dir.mkdir(parents = True, exist_ok = True)
        
        # create folders corresponding to the project
        self.ligands_path = self.ad4dir / "ligands"
        self.ligands_path.mkdir(parents=True)
        self.grid_path = self.ad4dir / "protein-grid"
        self.grid_path.mkdir(parents=True)
        self.result_path = self.ad4dir / "result"
        self.result_path.mkdir(parents=True)

        # define name and convert pdb to pdbqt
        if name == None:
            self.protein_name = Path(protein_pdb).resolve().stem
        else:
            self.protein_name = name
        self.protein_path = self.grid_path / "{}.pdbqt".format(self.protein_name)
        self.convert_pdb_to_pdbqt(protein_pdb, self.protein_path)

        # convert ligands into their pdbqts
        for file in os.listdir(ligand_fd):
            if file.endswith(".sdf"):
                ligand_input = Path(file).resolve()
                ligand_name = ligand_input.stem
                ligand_output = self.ligands_path / "{}.pdbqt".format(ligand_name)
                self.convert_sdf_to_pdbqt(ligand_input, ligand_output)
        
        # docking box information
        self.docking_box = docking_box

    def write_gpf_file(self, spacing = 0.375):
        """
        write autodock4 configuration file with the given docking box infomation

        @param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        @param spacing: float, spacing of the protein grid, default 0.375 angstroms

        @return: path, filepath to the successfully written gpf
        """
        # convert the box information into gpf-required information
        xmin, ymin, zmin, xmax, ymax, zmax = self.docking_box
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        center_z = (zmin + zmax) / 2.0
        npts_x = 2 * int((xmax - xmin) / (2*spacing))
        npts_y = 2 * int((ymax - ymin) / (2*spacing))
        npts_z = 2 * int((zmax - zmin) / (2*spacing))

        # allowed recptor types
        supported_atypes = set(['HD', 'C', 'A', 'N', 'NA', 'OA', 'F', 'P', 'SA', 'S',
                    'Cl', 'Br', 'I', 'Mg', 'Ca', 'Mn', 'Fe', 'Zn'])
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
        gpf_final = gpf.replace('RECTYPES',   rectypes)
        gpf_final = gpf_final.replace('PREFIX',     str(self.grid_path/self.protein_name))
        gpf_final = gpf_final.replace('REC',        str(self.protein_path))
        gpf_final = gpf_final.replace('NPTS_X',     '%d' % npts_x)
        gpf_final = gpf_final.replace('NPTS_Y',     '%d' % npts_y)
        gpf_final = gpf_final.replace('NPTS_Z',     '%d' % npts_z)
        gpf_final = gpf_final.replace('CENTER_X',   '%.3f' % center_x)
        gpf_final = gpf_final.replace('CENTER_Y',   '%.3f' % center_y)
        gpf_final = gpf_final.replace('CENTER_Z',   '%.3f' % center_z)
    
        # write everything to a config gpf file for autogrid
        self.gpf_path = self.grid_path / "{}.gpf".format(self.protein_name)
        with open(self.gpf_path, "w") as f:
            f.write(gpf_final)

        return True
    
    def run_autogrid4(self):
        """
        running autogrid4 to generate calculated grid based on the gpf file.

        @param gpf_path: path, path to the gpf file for grid generation

        @return fld_path: path, path to the prepared protein file for AD4. 
        """
        try:
            out = subprocess.run([autogrid_path, '-p', self.gpf_path])
        except subprocess.CalledProcessError as e:
            return e.output
        
        fld_file = self.grid_path / "{}.maps.fld".format(self.protein_name)
        if fld_file.is_file():
            self.fld_file = fld_file
        else:
            fld_file = self.grid_path.glob('*.maps.fld')[0]
            self.fld_file = fld_file

        return True

    def run_autodock(self):
        """
        Run autodock 4 with the protein and ligands
        """
        raise NotImplementedError()


        
        
    
