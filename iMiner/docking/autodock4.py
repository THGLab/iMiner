"""
Author: Oliver Sun
Date Created: Oct 28 2022

Automate the use of AutoDock4-GPU Docking Software 
"""

from iMiner.docking.autodock import AutoDockBaseDocking
from iMiner.cmd import set_directory
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import os
import re
import subprocess

autogrid_path = '/global/home/groups/co_armada2/local/ADFRsuite/bin/autogrid4'
ad4gpu_path = '/global/home/groups/co_armada2/local/AutoDock-GPU/bin/autodock_gpu_128wi'

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
    Run AutoDock4 with predefined binding pocket for ligands
    """
    def __init__(self, protein_pdb, docking_box, temp_path = None, name = None, **kwargs):
        """
        Initialize autodock4 with a protein and a docking box

        @param protein_pdb: str, path to the protein pdb file
        @param protein_ad4_fd: str, path to the autodock4 prepared protein folder
        @param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        @param name: str or None, name of the protein 
        """
        super().__init__(protein_pdb, docking_box)

        # define name and convert pdb to pdbqt
        if name == None:
            self.protein_name = Path(protein_pdb).resolve().stem
        else:
            self.protein_name = name
        
        self.ad4dir = Path(temp_path) / "{}-ad4".format(self.protein_name)
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
        self.ligand_name_smile_dict = {}

        self.protein_path = self.grid_path / "{}.pdbqt".format(self.protein_name)
        self.convert_pdb_to_pdbqt(protein_pdb, self.protein_path)
        
        # docking box information
        self.docking_box = docking_box

    def write_gpf_file(self, spacing):
        """
        write autodock4 configuration file with the given docking box infomation

        @param spacing: float, spacing of the protein grid, default 0.375 angstroms

        @return: path of the gpf file, when the run is successful
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
        gpf_final = gpf_final.replace('PREFIX',     str(self.protein_name))
        gpf_final = gpf_final.replace('REC',         "{}.pdbqt".format(self.protein_name))
        gpf_final = gpf_final.replace('NPTS_X',     '%d' % npts_x)
        gpf_final = gpf_final.replace('NPTS_Y',     '%d' % npts_y)
        gpf_final = gpf_final.replace('NPTS_Z',     '%d' % npts_z)
        gpf_final = gpf_final.replace('CENTER_X',   '%.3f' % center_x)
        gpf_final = gpf_final.replace('CENTER_Y',   '%.3f' % center_y)
        gpf_final = gpf_final.replace('CENTER_Z',   '%.3f' % center_z)
    
        # write everything to a config gpf file for autogrid
        gpf_file = "{}.gpf".format(self.protein_name)
        with open(self.grid_path / gpf_file, "w") as f:
            f.write(gpf_final)

        return gpf_file
    
    def run_autogrid4(self, gpf_file):
        """
        running autogrid4 to generate calculated grid based on the gpf file

        @param gpf_file: path or str, path to the gpf file for grid generation

        @return: file path to the fld file, when the run is successful
        """
        with set_directory(self.grid_path):
            try:
                out = subprocess.run([autogrid_path, '-p', gpf_file], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                return e.output
    
        # find the maps.fld file for autodock4
        fld_file = list(self.grid_path.glob('*.maps.fld'))[0]
        return fld_file

    def write_batch_dock_file(self, fld_file):
        """
        write the batch docking file to the ad4 folder.

        @param fld_file: str, file path to the prepared protein fld file

        @return: file path to the batch docking file
        """
        batch_lst = []
        batch_lst.append(str(fld_file))

        # write all ligands' input and output paths
        for file in os.listdir(self.ligands_path):
            if file.endswith("pdbqt"):
                ligand_input = self.ligands_path / file
                batch_lst.append(str(ligand_input))
                output_res = self.result_path / file.split('.')[0]
                batch_lst.append(str(output_res))
        
        # write everything to a config gpf file for autogrid
        batch_path = self.grid_path / "batch.txt"
        with open(batch_path, "w") as f:
            f.write("\n".join(batch_lst))

        return str(batch_path)
        
    def run_autodock(self, spacing, nrun):
        """
        Run autodock 4 with the protein and ligands

        @param spacing: grid space for autodock4, default = 0.375
        @param nrun: number of runs for each ligand, default = 200

        @return: True if the run is successful
        """
        gpf_file = self.write_gpf_file(spacing = spacing)
        fld_file = self.run_autogrid4(gpf_file)
        batch_file = self.write_batch_dock_file(fld_file)

        try:
            out = subprocess.run([ad4gpu_path, '--filelist', batch_file,\
                "--nrun", str(nrun), "-x", "0"], stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            return e.output
        
        return True
    
    def dlg_analysis(self, dlg_file, output_dir):
        """
        Take in a dlg filepath and return necessary information for the best docked pose

        @param dlg_file: str or path, path to the dlg file from running autodock4

        @return a row of dataframe that has information of the best pose of a ligand
        """
        ligand_name = Path(dlg_file).stem
        smile = self.ligand_name_smile_dict[ligand_name]
        with open(dlg_file, "r") as f:
            file = f.read()
        
        # Universal Patterns/Names
        rmsd_pattern =\
        "_____|______|______|___________|_________|_________________|___________\n"
        rmsd_columns = ["Rank","Sub-Rank", "Run", "Binding Energy", \
                        "Cluster RMSD","Reference RMSD", "Grep Pattern"]
        rmsd_table = file.split(rmsd_pattern)[-1].split("\n")[:-4]
        
        # write out the rmsd dataframe and extract lowest energy run
        with open("rmsd_table.txt","w") as f:
            f.write("\n".join(rmsd_table))
        rmsd_df = pd.read_csv("rmsd_table.txt", sep='\s+', \
                            names = rmsd_columns, engine = 'python')
        os.remove("rmsd_table.txt")
        best_score = rmsd_df.set_index('Run')['Binding Energy'].min()
        num_run = rmsd_df.set_index('Run')['Binding Energy'].idxmin()

        # convert the output file to sdf and extract the pose from the best run
        converted_sdf = self.result_path / "{}.sdf".format(ligand_name)
        self.convert_adresult_to_sdf(dlg_file, converted_sdf)
        with open(converted_sdf, "r") as f:
            all_sdf = f.read().split("$$$$\n")
        output_dir = Path(output_dir).resolve()
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        ligand_best_pose = output_dir / "{}-best-pose.sdf".format(ligand_name)
        with open(ligand_best_pose, "w") as f1:
            f1.write(all_sdf[num_run-1])
    
        return [smile, best_score, ligand_best_pose]

    def dock(self, ligands, output_dir, single_job_timeout=120, spacing = 0.375, nrun = 200):
        """
        All-together function that runs autodock and generate a pandas
        dataframe with necessary information for further analysis

        @param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        @param output_dir: str, path to the output directory
        @param single_job_timeout: int, timeout for each job in seconds
        @param spacing: grid space for autodock4, default = 0.375
        @param nrun: number of runs for each ligand, default = 200
        
        @return: dataframe that has necessary information of ad4result.
        """
        # convert ligands into their pdbqts
        for file in ligands:
            ligand_input = Path(file)
            ligand_name = ligand_input.stem
            ligand_output = self.ligands_path / "{}.pdbqt".format(ligand_name)
            self.convert_sdf_to_pdbqt(ligand_input, ligand_output)
            self.ligand_name_smile_dict[ligand_name] = self.convert_sdf_to_smiles(str(ligand_input))
        
        self.run_autodock(spacing = spacing, nrun = nrun)

        analysis_df = pd.DataFrame(columns=['smiles', 'score', 'path'])
        for file in os.listdir(self.result_path):
            if file.endswith(".dlg"):
                analysis_df.loc[len(analysis_df.index)] = \
                    self.dlg_analysis(self.result_path / file, output_dir)
        
        return analysis_df