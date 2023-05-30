'''
Author: Jie Li
Date Created: May 29, 2023

Using PLIP to analyze interactions for different ligands with different residues on a protein
'''

from iMiner.core.project import BaseProject
from iMiner.analysis import plip_analyze_single_frame
from iMiner.utils import box_from_center_and_size
from iMiner.log import init_logger
import pandas as pd
import shutil
import os
import sys
sys.path.append("/global/home/groups/co_armada2/avidd/plip")
from rdkit import Chem



class PLIPAnalyzer(BaseProject):
    def __init__(self, project_name, project_path=None, verbose=True) -> None:
        '''
        Initialize a project with a project name and a project path

        When project_path is None, the project will be initialized in the current working directory,
        using the project_name as the project folder name
        :param verbose: bool, whether to show and log processing messages
        '''
        super().__init__(project_name, project_path, verbose)


    def run_analysis(self, protein_name=None, ligand_names=None, output_csv=None, clean_tmp_after_finish=False, **kwargs):
        '''
        Using PLIP to run an analysis of ligand-residue interactions for multiple ligands with the same protein

        :param protein_name: str, name of the protein to run plip analysis with, should be an unbounded protein only
        :param ligand_names: list of str, names of the ligands to run analysis with. When not specified, use all ligands
        :param output_csv: str, path to the output csv file
        :param clean_tmp_after_finish: bool, whether to clean the temporary directory after the docking is finished
        '''
        # get protein name
        if protein_name is None:
            assert len(self.proteins) == 1, "Please specify the protein name if there are multiple proteins in the project!"
            protein_name = list(self.proteins.keys())[0]
        
        if ligand_names is None:
            ligand_names = self.ligands.keys()
            ligand_paths = self.ligands.values()
        else:
            ligand_paths = [self.ligands[ligand_name] for ligand_name in ligand_names]

        results = {}
        for ligand_name, ligand_path in zip(ligand_names, ligand_paths):
            complex_path = self.create_complex_pdb(ligand_name, self.proteins[protein_name], ligand_path)
            plip_results = plip_analyze_single_frame(complex_path)
            for interaction in plip_results:
                assert plip_results[interaction] == 1, f"Unexpected interaction count for {interaction}!"
                split_interaction = interaction.split("/")
                interaction_type = split_interaction[0]
                residue_id = "/".join(split_interaction[1:])

                if (ligand_name, residue_id) not in results:
                    results[(ligand_name, residue_id)] = interaction_type
                else:
                    results[(ligand_name, residue_id)] += f";{interaction_type}"

        
        results_df = pd.DataFrame()
        for (ligand_name, residue_id) in results:
            results_df.loc[ligand_name, residue_id] = results[(ligand_name, residue_id)]

        # when output_csv is not specified, auto-generate one using the protein_name
        if output_csv is None:
            output_csv = self.project_path / "PLIP"  / f"{protein_name}_results.csv"
        results_df.to_csv(output_csv)

        if self.verbose:
            self.logger.info(f"PLIP analysis completed! Results saved to {output_csv}")
        
        if clean_tmp_after_finish:
            self.clean_tmp()


    def create_complex_pdb(self, name, protein_pdb, ligand_sdf, working_dir=None, **kwargs):
        '''
        Create a complex pdb file for a given protein and ligand

        :param name: str, a name to identify the complex
        :param protein_name: str, name of the protein to create complex with
        :param ligand_name: str, name of the ligand to create complex with
        :param working_dir: str, where the complex structure will be stored. Default is temp path
        '''
        if working_dir is None:
            working_dir = self.temp_path

        mol = Chem.MolFromMolFile(ligand_sdf)
        mol_pdb = Chem.MolToPDBFile(mol, os.path.join(working_dir, f"{name}_ligand.pdb"))

        with open(protein_pdb, "r") as f:
            protein_pdb_lines = f.readlines()
            protein_pdb_lines = [line for line in protein_pdb_lines if not (line.startswith("TER") \
                                 or line.startswith("END"))]
        with open(os.path.join(working_dir, f"{name}_ligand.pdb"), "r") as f:
            ligand_pdb_lines = f.readlines()
            ligand_pdb_lines = [line for line in ligand_pdb_lines if not line.startswith("CONECT")]

        complex_path = os.path.join(working_dir, f"{name}.pdb")
        with open(complex_path, "w") as f:
            f.writelines(protein_pdb_lines)
            f.writelines(ligand_pdb_lines)
        return complex_path

    def clean_tmp(self):
        '''
        Clean the temporary directory
        '''
        shutil.rmtree(self.temp_path)
        self.logger.info(f"Temporary directory {self.temp_path} cleaned!")
