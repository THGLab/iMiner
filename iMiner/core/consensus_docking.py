'''
Author: Jie Li
Date Created: Oct 21, 2022

A consensus docking module for iMiner
'''

from iMiner.core.project import BaseProject
from iMiner.docking import *
from iMiner.log import init_logger
import pandas as pd


docking_protocol_map = {
    "ad4": AD4Docking,
    "vina": VinaDocking,
    "vina-gpu": VinaGPUDocking,
    "icm": ICMDocking
}

class ConsensusDocking(BaseProject):
    def __init__(self, project_name, project_path=None, docking_protocols=None, verbose=True) -> None:
        '''
        Initialize a project with a project name and a project path

        When project_path is None, the project will be initialized in the current working directory,
        using the project_name as the project folder name

        :param docking_protocols: list of docking protocols to be used for consensus docking, list of ["vina", "vina-gpu", "ad4", "icm"]
        :param verbose: bool, whether to show and log processing messages
        '''
        super().__init__(project_name, project_path, verbose)
        self.docking_protocols = {protocol_name: docking_protocol_map[protocol_name] for protocol_name in docking_protocols}

    def run_consensus_docking(self, protein_name=None, ligand_names=None, output_csv=None, **kwargs):
        '''
        Run consensus docking for all (or given) ligands in the project into the given protein, using different docking protocols

        :param protein_name: str, name of the protein to dock into
        :param ligand_names: list of str, names of the ligands to dock. When not specified, use all ligands
        :param output_csv: str, path to the output csv file
        '''
        # get protein name
        if protein_name is None:
            assert len(self.proteins) == 1, "Please specify the protein name if there are multiple proteins in the project!"
            protein_name = list(self.proteins.keys())[0]
        consensus_docking_path = self.project_path / "consensus_docking" / protein_name

        results_df = []
        for protocol in self.docking_protocols:
            docking_obj = self.docking_protocols[protocol](self.proteins[protein_name],
                                                           self.binding_sites[protein_name],
                                                           self.temp_path, self.logger, **kwargs)
            docking_path = consensus_docking_path / protocol
            docking_path.mkdir(exist_ok=True, parents=True)
            if ligand_names is None:
                ligand_names = self.ligands.keys()
                ligand_paths = self.ligands.values()
            else:
                ligand_paths = [self.ligands[ligand_name] for ligand_name in ligand_names]
            if self.verbose:
                n_cores = kwargs.get("n_jobs", 1)
                n_ligands = len(ligand_names)
                self.logger.info(f"Start docking {n_ligands} ligands with {protocol} using {n_cores} cores...")
            results = docking_obj.dock_parallel(ligand_paths, docking_path, **kwargs)
            results["ligand_names"] = ligand_names
            results["protocol"] = protocol
            results_df.append(results)

        final_results = pd.concat(results_df)[["ligand_names", "score", "smiles", "protocol", "path", "original_names"]]

        # when output_csv is not specified, auto-generate one using the protein_name
        if output_csv is None:
            output_csv = consensus_docking_path / f"{protein_name}_consensus_docking_results.csv"
        final_results.to_csv(output_csv, index=False)

        if self.verbose:
            self.logger.info(f"Consensus docking completed! Results saved to {output_csv}")
