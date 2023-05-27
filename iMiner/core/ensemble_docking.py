'''
Author: Jie Li
Date Created: Mar 17, 2023

An ensemble docking module for iMiner
'''

from iMiner.core.project import BaseProject
from iMiner.docking import *
from iMiner.utils import box_from_center_and_size
from iMiner.log import init_logger
import pandas as pd
import shutil
import os


docking_protocol_map = {
    "ad4": AD4Docking,
    "vina": VinaDocking,
    "vina-gpu": VinaGPUDocking,
    "icm": ICMDocking,
    "tankbind": TankBindDocking
}

class EnsembleDocking(BaseProject):
    def __init__(self, project_name, project_path=None, docking_protocol=None, verbose=True) -> None:
        '''
        Initialize a project with a project name and a project path

        When project_path is None, the project will be initialized in the current working directory,
        using the project_name as the project folder name

        :param docking_protocol: the docking protocols to be used for ensemble docking, one of ["vina", "vina-gpu", "ad4", "icm", "tankbind"]
        :param verbose: bool, whether to show and log processing messages
        '''
        super().__init__(project_name, project_path, verbose)
        self.docking_protocol = docking_protocol_map[docking_protocol]
        self.ensemble_info = {}
        if self._new_project:
            self._update_ensemble_info_in_meta()
        else:
            self.ensemble_info = self.meta_data["ensemble_info"]


    def _update_ensemble_info_in_meta(self):
        self.meta_data["ensemble_info"] = self.ensemble_info
        self.update_meta_data()

    def add_protein_confs(self, protein_name, protein_confs, protein_conf_names=None, binding_sites=None, binding_site_dataframe=None, binding_box_size=25):
        '''
        Add multiple conformations of the same protein to the project, and specify the binding sites in each of the conformations

        :param protein_name: str, name of the protein
        :param protein_confs: list of str, paths to the protein conformations, or str, path to the directory containing the protein conformations
        :param protein_conf_names: optional, list of str, names of the protein conformations
        :param binding_sites: needed when binding_site_dataframe is None, list of (x,y,z) tuples, binding site coordinates in the corresponding protein conformations
        :param binding_site_dataframe: needed when binding_stes is None, csv file containing binding site coordinates in the corresponding protein conformations
                the dataframe should contain the following columns: [name, x, y, z]
        :param binding_box_size: int, size of the binding box. assuming the binding box is a cube
        '''
        if isinstance(protein_confs, str):
            assert os.path.exists(protein_confs), f"protein_confs has to be a list of protein paths or a path to the directory containing the protein conformations!"
            protein_confs = [os.path.join(protein_confs, protein_conf) for protein_conf in os.listdir(protein_confs)]
        if protein_conf_names is None:
            protein_conf_names = [str(i) for i in range(len(protein_confs))]
        full_protein_names = [f"{protein_name}_{protein_conf_name}" for protein_conf_name in protein_conf_names]
        if binding_sites is not None:
            assert binding_site_dataframe is None, "Please specify either binding_sites or binding_site_dataframe, not both!"
            assert len(protein_confs) == len(binding_sites), "The number of protein conformations and binding sites should be the same!"
        else:
            assert binding_site_dataframe is not None, "Please specify either binding_sites or binding_site_dataframe!"
            binding_site_df = pd.read_csv(binding_site_dataframe)
            binding_site_df.index = binding_site_df["name"]
            binding_sites = []
            for protein_conf_name in protein_conf_names:
                binding_sites.append(binding_site_df.loc[protein_conf_name, ["x", "y", "z"]].values)

        # add all proteins to the project
        for protein_conf, full_protein_name, binding_site in zip(protein_confs, full_protein_names, binding_sites):
            binding_box = box_from_center_and_size(center=binding_site, size=(binding_box_size, binding_box_size, binding_box_size))
            self.add_protein(protein_conf, full_protein_name, binding_site=binding_box)

        # add ensemble information to the project
        self.ensemble_info.update({
            protein_name: full_protein_names
        })
        self._update_ensemble_info_in_meta()

        self.create_docking_objects(protein_name)


    def create_docking_objects(self, protein_name=None, **kwargs):
        '''
        prepare docking objects for each protein conformation

        :param protein_name: str, name of the protein to dock into, it should be in ensemble_info and contains multiple conformations
        '''
        # get protein name
        if protein_name is None:
            assert len(self.ensemble_info) == 1, "Please specify the protein name when there are multiple proteins in the project!"
            protein_name = list(self.ensemble_info.keys())[0]
        self.docking_obj = {}
        for full_protein_name in self.ensemble_info[protein_name]:
            self.docking_obj[full_protein_name] = self.docking_protocol(self.proteins[full_protein_name], 
                                                                        self.binding_sites[full_protein_name],
                                                                        self.temp_path, self.logger, **kwargs)
        


    def run_ensemble_docking(self, protein_name=None, ligand_names=None, output_prefix=None, clean_tmp_after_finish=False, **kwargs):
        '''
        Run ensemble docking for all (or given) ligands in the project into all conformations of the protein, and return the best result from all conformations

        :param protein_name: str, name of the protein to dock into
        :param ligand_names: list of str, names of the ligands to dock. When not specified, use all ligands
        :param output_prefix: str, optional path prefix to the output csv files ({prefix}_full_results.csv and {prefix}_aggr_results.csv)
        :param clean_tmp_after_finish: bool, whether to clean the temporary directory after the docking is finished
        '''
        # get protein name
        if protein_name is None:
            assert len(self.ensemble_info) == 1, "Please specify the protein name if there are multiple proteins in the project!"
            protein_name = list(self.ensemble_info.keys())[0]
        ensemble_docking_path = self.project_path / "ensemble_docking" / protein_name

        results_df = []
        for conf in self.ensemble_info[protein_name]:
            docking_obj = self.docking_obj[conf]
            docking_path = ensemble_docking_path / conf
            docking_path.mkdir(exist_ok=True, parents=True)
            if ligand_names is None:
                ligand_names = self.ligands.keys()
                ligand_paths = self.ligands.values()
            else:
                ligand_paths = [self.ligands[ligand_name] for ligand_name in ligand_names]
            
            assert len(ligand_names) > 0, "No ligand for docking!"
            if self.verbose:
                n_cores = kwargs.get("n_jobs", 1)
                n_ligands = len(ligand_names)
                self.logger.info(f"Start docking {n_ligands} ligands into conformation {conf} using {n_cores} cores...")
            results = docking_obj.dock_parallel(ligand_paths, docking_path, **kwargs)
            results["ligand_names"] = ligand_names
            results["conformation"] = conf
            results_df.append(results)

        full_results = pd.concat(results_df)[["ligand_names", "score", "smiles", "conformation", "path", "original_names"]]

        # when output_prefix is not specified, auto-generate names using the protein_name
        if output_prefix is None:
            full_result_csv = ensemble_docking_path / f"ensemble_docking_full_results.csv"
            aggr_result_csv = ensemble_docking_path / f"ensemble_docking_aggr_results.csv"
        else:
            full_result_csv = f"{output_prefix}_full_results.csv"
            aggr_result_csv = f"{output_prefix}_aggr_results.csv"
        full_results.to_csv(full_result_csv, index=False)

        # aggregrate results based on conformations
        full_results["score_conf"] = full_results.apply(lambda row:(row['score'], row['conformation']),axis=1)
        aggr_results = full_results.groupby("ligand_names").agg({"score_conf": "min", "smiles": "first"})
        aggr_results["score"] = aggr_results["score_conf"].apply(lambda x: x[0])
        aggr_results["conformation"] = aggr_results["score_conf"].apply(lambda x: x[1])
        aggr_results.drop("score_conf", axis=1).to_csv(aggr_result_csv, index=True)

        if self.verbose:
            self.logger.info(f"Consensus docking completed! Results saved to {aggr_result_csv}")
        
        if clean_tmp_after_finish:
            self.clean_tmp()

    def clean_tmp(self):
        '''
        Clean the temporary directory
        '''
        shutil.rmtree(self.temp_path)
        self.logger.info(f"Temporary directory {self.temp_path} cleaned!")
