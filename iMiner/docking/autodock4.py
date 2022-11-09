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
    def __init__(self, protein_pdb, ligand_fd, protein_ad4_fd, docking_box, name = None):
        """
        Initialize autodock4 with a protein and a docking box

        @param protein_pdb: str, path to the protein pdb file
        @param ligand_fd: str, path to the folder that stores all ligands
        @param protein_ad4_fd: str, path to the autodock4 prepared protein folder
        @param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        @param name: str or None, name of the protein 
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
        self.best_pose_path = self.ad4dir / "best-pose"
        self.best_pose_path.mkdir(parents=True)

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
                ligand_input = Path(ligand_fd) / file
                ligand_name = ligand_input.stem
                ligand_output = self.ligands_path / "{}.pdbqt".format(ligand_name)
                self.convert_sdf_to_pdbqt(ligand_input, ligand_output)
        
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
        gpf_file = self.grid_path / "{}.gpf".format(self.protein_name)
        with open(gpf_file, "w") as f:
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
    
    def dlg_analysis_rmsd(self, dlg_file):
        """
        Take in a dlg filepath and return a rmsd dataframe with x, y, z center of the ligand

        @param dlg_file: str or path, path to the dlg file from running autodock4

        @return dataframe that carries information for all poses of a single ligand
        """
        ligand_name = dlg_file.stem
        with open(dlg_file, "r") as f:
            file = f.read()
        
        # Universal Patterns/Names
        rmsd_pattern =\
        "_____|______|______|___________|_________|_________________|___________\n"
        rmsd_columns = ["Rank","Sub-Rank", "Run", "Binding Energy", \
                        "Cluster RMSD","Reference RMSD", "Grep Pattern"]
        coor_columns = ["Atom","Ligand Name", "Num", "x", \
                        "y","z", "vdw", "Electrostatic", "q", "Type"]
        rmsd_table = file.split(rmsd_pattern)[-1].split("\n")[:-4]
        
        # write out the rmsd dataframe for further analysis
        with open("rmsd_table.txt","w") as f:
            f.write("\n".join(rmsd_table))
        rmsd_df = pd.read_csv("rmsd_table.txt", sep='\s+', \
                            names = rmsd_columns, engine = 'python')
        os.remove("rmsd_table.txt")
        rmsd_df['center_x'] = 0
        rmsd_df['center_y'] = 0
        rmsd_df['center_z'] = 0
        
        # generate pdbqt files & calculate center x,y,z coordinates for docked poses
        docked_lst = file.split\
        ("    FINAL DOCKED STATE:\n    ________________________")[1:]
        
        for pose in docked_lst:
            coordinates = re.findall\
            ("DOCKED: MODEL[\s\S]+DOCKED: TER", pose)[0].split("\n")
            num_run = int(re.findall("Run = (\d+)", pose)[0])
            mol_coor = [i[12:] for i in coordinates if "DOCKED: ATOM" in i]
            
            # calculate the geometric average center of the molecule
            with open("mol_coor.txt", "w") as f:
                f.write("\n".join(mol_coor))
            coor = pd.read_csv("mol_coor.txt", sep = '\s+', \
                        engine = 'python', index_col = 0, names = coor_columns)
            os.remove("mol_coor.txt")
            
            center = coor[['x', 'y', 'z']].mean()
            rmsd_df.loc[rmsd_df['Run'] == num_run, 'center_x'] = center['x']
            rmsd_df.loc[rmsd_df['Run'] == num_run, 'center_y'] = center['y']
            rmsd_df.loc[rmsd_df['Run'] == num_run, 'center_z'] = center['z']
            
            # write out the all docked poses of the ligand
            all_ligands = self.result_path / ligand_name
            all_ligands.mkdir(exist_ok=True, parents=True)
            path = all_ligands / "docked_{}.pdbqt".format(num_run)
            pdbqt = [i[8:] for i in coordinates]
            with open(path, "w") as f:
                f.write("\n".join(pdbqt))
    
        return rmsd_df

    def ad4result_analysis(self, spacing = 0.375, nrun = 200):
        """
        All-together function that runs autodock and generate a pandas
        dataframe with necessary information for further analysis

        @param spacing: grid space for autodock4, default = 0.375
        @param nrun: number of runs for each ligand, default = 200
        
        @return: dataframe that has necessary information of ad4result.
        """
        self.run_autodock(spacing = spacing, nrun = nrun)
        analysis_df = pd.DataFrame(columns=['Index', 'Smile', 'Best Energy', 'Path to Best Pose'])
        for file in os.listdir(self.result_path):
            if file.endswith(".dlg"):
                ligand_name = Path(file).stem
                df = self.dlg_analysis_rmsd(self.result_path / file)
                run = df.set_index('Run')['Binding Energy'].idxmin()
                best_pose_path = self.best_pose_path / "{}-best-pose.pdbqt".format(ligand_name)
                shutil.copyfile(self.result_path / ligand_name/ 'docked_{}.pdbqt'.format(run), best_pose_path)
                shutil.rmtree(self.result_path / ligand_name)
                best_score = df['Binding Energy'].min()
                analysis_df.loc[len(analysis_df.index)] = [ligand_name, '', best_score, best_pose_path]
        
        return analysis_df