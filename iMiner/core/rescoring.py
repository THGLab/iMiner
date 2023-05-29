'''
Author: Jie Li
Date Created: May 29, 2023

Rescoring poses with one or more scoring functions
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
    "rfscore": RFScoring,
}

class Rescoring(BaseProject):
    def __init__(self, project_name, project_path=None, protocols=None, extra_params=None, verbose=True) -> None:
        '''
        Initialize a project with a project name and a project path

        When project_path is None, the project will be initialized in the current working directory,
        using the project_name as the project folder name

        :param protocols: a list of scoring functions to be used for scoring, one of ["vina", "vina-gpu", "ad4", "rfscore"]
        :param extra_params: None, or a dictionary describing additional parameters for each scoring protocol
        :param verbose: bool, whether to show and log processing messages
        '''
        super().__init__(project_name, project_path, verbose)
        self.protocols = {protocol_name: docking_protocol_map[protocol_name] for protocol_name in protocols}
        self.extra_params = {}
        if extra_params is not None:
            self.extra_params.update(extra_params)


    def run_rescoring(self, protein_name=None, ligand_names=None, output_csv=None, clean_tmp_after_finish=False, **kwargs):
        '''
        Run consensus docking for all (or given) ligands in the project into the given protein, using different docking protocols

        :param protein_name: str, name of the protein to dock into
        :param ligand_names: list of str, names of the ligands to dock. When not specified, use all ligands
        :param output_csv: str, path to the output csv file
        :param clean_tmp_after_finish: bool, whether to clean the temporary directory after the docking is finished
        '''
        # get protein name
        if protein_name is None:
            assert len(self.proteins) == 1, "Please specify the protein name if there are multiple proteins in the project!"
            protein_name = list(self.proteins.keys())[0]
        output_path = self.project_path / "rescoring" / protein_name
        
        if ligand_names is None:
            ligand_names = self.ligands.keys()
            ligand_paths = self.ligands.values()
        else:
            ligand_paths = [self.ligands[ligand_name] for ligand_name in ligand_names]

        results_df = []
        for protocol in self.protocols:
            scoring_obj = self.protocols[protocol](self.proteins[protein_name],
                                                    self.binding_sites[protein_name],
                                                    self.temp_path, self.logger,
                                                    **self.extra_params.get(protocol, {}), **kwargs)
            results = scoring_obj.rescore_parallel(ligand_paths, output_path, **kwargs)
            results.rename(columns={"ligand_name": "ligand_paths"}, inplace=True)
            results["ligand_names"] = ligand_names
            results["protocol"] = protocol
            results_df.append(results)
            if self.verbose:
                self.logger.info(f"{protocol} finished")
        final_results = pd.concat(results_df)[["ligand_names", "score", "smiles", "protocol", "ligand_paths"]]

        # when output_csv is not specified, auto-generate one using the protein_name
        if output_csv is None:
            output_csv = output_path / f"{protein_name}_rescoring_results.csv"
        final_results.to_csv(output_csv, index=False)

        if self.verbose:
            self.logger.info(f"All rescoring completed! Results saved to {output_csv}")
        
        if clean_tmp_after_finish:
            self.clean_tmp()



    def clean_tmp(self):
        '''
        Clean the temporary directory
        '''
        shutil.rmtree(self.temp_path)
        self.logger.info(f"Temporary directory {self.temp_path} cleaned!")
