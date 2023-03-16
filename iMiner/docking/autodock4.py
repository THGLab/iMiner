"""
Author: Oliver Sun, Oufan Zhang
Date Created: Oct 28 2022

Automate the use of AutoDock4-GPU Docking Software 
"""

from iMiner.docking.autodock import AutoDockBaseDocking
from iMiner.cmd import set_directory
from iMiner.pathlib import *
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import os
import re
#import time
import subprocess



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
    def __init__(self, protein_pdb, docking_box, temp_path = None, name = None, logger = None, **kwargs):
        """
        Initialize autodock4 with a protein and a docking box
        @param protein_pdb: str, path to the protein pdb file
        @param temp_path: str, path to the autodock4 prepared protein folder
        @param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        @param name: str or None, name of the protein 
        """
        super().__init__(protein_pdb, docking_box, logger=logger)
        
        # define name and convert pdb to pdbqt
        if name == None:
            self.protein_name = Path(protein_pdb).resolve().stem
        else:
            self.protein_name = name
        
        self.ad4dir = Path(temp_path).resolve() / "{}-ad4".format(self.protein_name)
        if self.ad4dir.exists() and self.ad4dir.is_dir():
            shutil.rmtree(self.ad4dir)  
        self.ad4dir.mkdir(parents = True, exist_ok = True)
        
        # create folders corresponding to the project
        self.ligands_path = self.ad4dir / "ligands"
        self.ligands_path.mkdir(parents=True, exist_ok = True)
        self.grid_path = self.ad4dir / "protein-grid"
        self.grid_path.mkdir(parents=True, exist_ok = True)
        self.result_path = self.ad4dir / "result"
        self.result_path.mkdir(parents=True, exist_ok = True)

        # convert pdb to pdbqt
        self.protein_path = self.grid_path / "{}.pdbqt".format(self.protein_name)
        
        if not os.path.exists(self.protein_path):
            if protein_pdb.endswith('.pdb'):
                self.convert_pdb_to_pdbqt(protein_pdb, self.protein_path)
            elif protein_pdb.endswith('.pdbqt'):
                shutil.copy(protein_pdb, self.protein_path)
        
        # support for flexible docking
        self.flex_docking = False
        if 'flex' in kwargs:
            self.flex_docking = True
            if not protein_pdb.endswith('.pdbqt'):
                raise RuntimeError('Using flexible docking. Your protein input must be a processed rigid protein pdbqt.')
            flexres = self.grid_path / "{}_flex.pdbqt".format(self.protein_name)
            if not os.path.exists(flexres):
                if kwargs['flex'].endswith('.pdbqt'):
                    shutil.copy(kwargs['flex'], flexres)
                else:
                    raise RuntimeError('flexible protein file not supported')
        
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

    def write_batch_dock_file(self, fld_file, liglist):
        """
        write the batch docking file to the ad4 folder.

        @param fld_file: str, file path to the prepared protein fld file
        @param liglist: list of path of ligands to dock

        @return: file path to the batch docking file
        """
        batch_lst = []
        batch_lst.append(str(fld_file))

        # write all ligands' input and output paths
        for f in liglist:
            file = str(f)
            if file.endswith(".pdbqt"):
                batch_lst.append(file)
                output_res = self.result_path / file.split('/')[-1].split('.')[0]
                batch_lst.append(str(output_res))
        
        # write everything to a config gpf file for autogrid
        batch_path = self.grid_path / "batch.txt"
        with open(batch_path, "w") as f:
            f.write("\n".join(batch_lst))

        return str(batch_path)
        
    def run_autodock(self, spacing, nrun, liglist):
        """
        Run autodock 4 with the protein and ligands

        @param spacing: grid space for autodock4, default = 0.375
        @param nrun: number of runs for each ligand
        @param liglist: list of path of ligands to dock

        @return: True if the run is successful
        """
        gpf_file = self.grid_path / "{}.gpf".format(self.protein_name)
        fld_file = self.grid_path / '{}.maps.fld'.format(self.protein_name)
        batch_file = self.grid_path / "batch.txt"
        
        if not os.path.exists(gpf_file):
            gpf_file = self.write_gpf_file(spacing = spacing)
        if not os.path.exists(fld_file):
            fld_file = self.run_autogrid4(gpf_file)
        batch_file = self.write_batch_dock_file(fld_file, liglist)
        
        cmd = [ad4gpu_path, '--filelist', batch_file,\
                "--nrun", str(nrun), "-x", "0", "--rlige", "1"]
        if self.flex_docking:
            cmd += ['-F', '%s/%s_flex.pdbqt'%(self.grid_path, self.protein_name)]
            
        try:
            out = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            return e.output
        
        return True
    
    def dlg_analysis(self, dlg_file, output_dir, rescore=False, write_best_pose = True, write_best_cluster_pose = True):
        """
        Take in a dlg filepath and return necessary information for the best docked pose

        @param dlg_file: str or path, path to the dlg file from running autodock4

        @return a row of dataframe that has information of the best pose of a ligand
        """
        if not os.path.isfile(dlg_file): 
            print("DLG File %s not found."%dlg_file)
            return

        ligand_name = Path(dlg_file).stem
        smile = self.read_smiles_from_dlg(dlg_file)
        with open(dlg_file, "r") as f:
            file = f.read()

        if rescore:
            keyword = "INPUT-LIGAND-PDBQT: USER    Estimated Free Energy of Binding    ="
            line = file.split(keyword)[1].split("\n")[0]
            energy = float(line.strip().split()[0])
            return [ligand_name, smile, energy]
        
        # Universal Patterns/Names
        rmsd_pattern =\
        "_____|______|______|___________|_________|_________________|___________\n"
        cluster_pattern1 =\
        "_____|___________|_____|___________|_____|____:____|____:____|____:____|____:___\n"
        cluster_pattern2 =\
        "_____|___________|_____|___________|_____|______________________________________\n"
        
        rmsd_columns = ["Rank","Sub-Rank", "Run", "Binding Energy", \
                        "Cluster RMSD","Reference RMSD", "Grep Pattern"]
        rmsd_table = file.split(rmsd_pattern)[-1].split("\n")[:-4]
        cluster_data = file[file.find(cluster_pattern1):\
                            file.find(cluster_pattern2)].split("\n")[1:-1]
        best_cluster = cluster_data[np.argmax([s.count("#") for s in cluster_data])]\
                        .replace("|","").split()
        score_cluster, run_cluster = float(best_cluster[1]), int(best_cluster[2])
        # write out the rmsd dataframe and extract lowest energy run
        with open(f"rmsd_table_{ligand_name}.txt","w") as f:
            f.write("\n".join(rmsd_table))
        rmsd_df = pd.read_csv(f"rmsd_table_{ligand_name}.txt", sep='\s+', \
                            names = rmsd_columns, engine = 'python')
        os.remove(f"rmsd_table_{ligand_name}.txt")

        try:
            best_score = rmsd_df.set_index('Run')['Binding Energy'].min()
            num_run = rmsd_df.set_index('Run')['Binding Energy'].idxmin()
        except TypeError:
            print('Errors in binding energy for %s'%dlg_file)
            return []

        # convert the output file to sdf and extract the pose from the best run
        converted_sdf = self.result_path / "{}.sdf".format(ligand_name)
        succ = self.convert_adresult_to_sdf(dlg_file, converted_sdf)
        if not (succ and os.path.exists(converted_sdf)):
            return [ligand_name, smile, best_score, None]
            
        with open(converted_sdf, "r") as f:
            all_sdf = f.read().split("$$$$\n")
        output_dir = Path(output_dir).resolve()
        
        if write_best_pose:
            ligand_best_pose = output_dir / "{}.sdf".format(ligand_name)
            with open(ligand_best_pose, "w") as f1:
                f1.write(all_sdf[num_run-1])
        
        if write_best_cluster_pose:
            ligand_cluster_pose = output_dir / "cluster-score-{}-{}.sdf".format(score_cluster, ligand_name)
            with open(ligand_cluster_pose, "w") as f1:
                f1.write(all_sdf[run_cluster-1])
            
        return [ligand_name, smile, best_score, ligand_best_pose]
    
    def convert_ligand(self, file):
        '''
        Convert a ligand into pdbqt
        
        :return: path to pdbqt
        '''
        ligand_input = Path(file)
        ligand_name = ligand_input.stem
        ligand_output = self.ligands_path / "{}.pdbqt".format(ligand_name)
        succ = self.convert_sdf_to_pdbqt(ligand_input, ligand_output)
        if not (succ and os.path.exists(ligand_output)):
            return False
        return ligand_output
        
    def dock_parallel(self, ligands, output_dir, n_jobs=1, single_job_timeout=120, verbose=True, save_df_freq=500):
        '''
        Dock a list of ligands to the pocket in the protein in parallel for AutoDock GPU.
        *overwrites the base function*

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        :param output_dir: str, path to the output directory
        :param n_jobs: int, number of jobs to run in parallel
        :param verbose: bool, whether to show progress bar

        :return: pd.DataFrame with columns ["original_name", "smiles", "score", "path"], path is the path to the docked conformation
        '''
        import multiprocessing
        from tqdm import tqdm
        from functools import partial
        from iMiner.docking.base import unpack_helper
        
        pool = multiprocessing.Pool(n_jobs)
        
        # generates pdbqts in parallel
        lig_outs = []
        ligands = [[l] for l in ligands]
        if verbose:
            pbar = tqdm(total=len(ligands))
        for l in pool.imap(partial(unpack_helper, self.convert_ligand), ligands):
            if l:
                lig_outs.append(l)
            if verbose:
                pbar.update(1)
        if self.logger is not None:
            self.logger.info(f"Finished preparing pdbqt inputs.")
        
        # run autodock in batch mode
        #st = time.time()
        self.run_autodock(spacing = 0.375, nrun = 10, liglist = lig_outs)
        if self.logger is not None:
            self.logger.info(f"Docking finished.")
        os.makedirs(output_dir, exist_ok=True)
        
        # process docking results 
        dlgs = [self.result_path / (str(file.stem) + ".dlg") for file in lig_outs]
        zipped_args = zip(dlgs, [output_dir] * len(dlgs))   
        counter = 0
        results = pd.DataFrame(columns=['original_name', 'smiles', 'score', 'path'])
        if verbose:
            pbar = tqdm(total=len(dlgs))
        for result in pool.imap(partial(unpack_helper, self.dlg_analysis), zipped_args):
            counter += 1
            if len(result) > 0:
                results.loc[len(results.index)] = result
            if verbose:
                pbar.update(1)
        if self.logger is not None:
            self.logger.info(f"Outputs processed.")
        return results
        

    def dock(self, ligands, output_dir, single_job_timeout=120, spacing = 0.375, nrun = 10):
        """
        All-together function that runs autodock and generate a pandas
        dataframe with necessary information for further analysis

        @param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        @param output_dir: str, path to the output directory
        @param single_job_timeout: int, timeout for each job in seconds
        @param spacing: grid space for autodock4, default = 0.375
        @param nrun: number of runs for each ligand, default = 10
        
        @return: dataframe that has necessary information of ad4result.
        """
        # convert ligands into their pdbqts
        lig_outs = []
        for file in ligands:
            out = self.convert_ligand(file)
            if out:
                lig_outs.append(out)
    
        self.run_autodock(spacing = spacing, nrun = nrun, liglist=lig_outs)
        #print("Docking finished. Time elapsed %s hrs"%((time.time()-st)/3600.))
        os.makedirs(output_dir, exist_ok=True)
            
        analysis_df = pd.DataFrame(columns=['original_name', 'smiles', 'score', 'path'])
        for file in lig_outs:
            f = str(file.stem) + ".dlg"
            result = self.dlg_analysis(self.result_path / f, output_dir)
            if len(result) > 0:
                analysis_df.loc[len(analysis_df.index)] = result
        return analysis_df
        
    def rescore(self, ligands):
        """
        All-together function that runs autodock for scoring in place and generate a pandas
        dataframe with necessary information for further analysis

        @param ligands: list of ligands, each ligand is a path to the corresponding .sdf/.pdbqt file
                        If given .pdbqt files, assumes in the same folder
        
        @return: dataframe that has necessary information of ad4result.
        """
        # convert ligands into their pdbqts
        lig_outs = []
        for file in ligands:
            if file.endswith('.sdf'):
                ligand_output = self.convert_ligand(file)
                if ligand_output:
                    lig_outs.append(ligand_output)
            elif file.endswith('.pdbqt'):
                lig_outs.append(os.path.abspath(file))
        
        self.run_autodock(spacing = 0.375, nrun = 1, liglist=lig_outs)

        analysis_df = pd.DataFrame(columns=['ligand_name', 'smiles', 'score'])
        for file in lig_outs:
            f = str(file.stem) + ".dlg"
            analysis_df.loc[len(analysis_df.index)] = self.dlg_analysis(self.result_path / f,
                    None, True)
        
        return analysis_df
