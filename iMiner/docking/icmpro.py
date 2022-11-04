'''
Author: Jie Li
Date Created: Nov 4, 2022

Defines the docking class for ICM-pro
'''

from pathlib import Path
from typing import Optional
import os
from iMiner.docking.base import BaseDocking
from iMiner.cmd import run_command, set_directory
from rdkit import Chem

ICM_HOME = Path("/global/home/groups/co_armada2/bins/icm-pro/")

class ICMDocking(BaseDocking):
    def __init__(self, protein_pdb, docking_box, temp_path: Optional[os.PathLike] = None, **kwargs) -> None:
        super().__init__(protein_pdb, docking_box)
        self.working_path = temp_path / "{}-icm".format(self.protein_name)
        os.makedirs(self.working_path, exist_ok = True)
        self.docking_box = docking_box
        self.prepare_icm_docking_project(protein_pdb, docking_box)

        # extra parameters to be used during ICM docking
        params = {"confs": 1, "effort": 3.}
        params.update(kwargs)

        self.params = params
        

    def prepare_icm_docking_project(self, protein_pdb, docking_box):
        with set_directory(self.working_path):
            cmd = f"{ICM_HOME}/icm64 {ICM_HOME}/_dockBatch {protein_pdb} " + \
                "box={},{},{},{},{},{} ".format(*docking_box)
            run_command(cmd)
        self.docking_project_name = "D_" + Path(protein_pdb).stem.upper()

    def dock(self, ligands, output_dir):
        '''
        Run actual docking with ICM-pro

        :param ligands: list of ligands, each ligand is a path (str or os.PathLike) to the corresponding .sdf file
        :param output_dir: str, path to the output directory where sdf files for the docked conformations will be saved
        '''
        # prepare lists to record results
        ligand_smiles = []
        ligand_scores = []
        ligand_conformation_paths = []

        # First make sure output_dir exists
        os.makedirs(output_dir, exist_ok = True)

        # prepare all ligands into a single .sdf file, using cat command
        cmd = "cat " + " ".join(ligands) + " > " + str(self.working_path / "ligands.sdf")
        run_command(cmd)

        # do actual docking
        with set_directory(self.working_path):
            cmd = f"{ICM_HOME}/icm64 {ICM_HOME}/_dockScan {self.docking_project_name} -a -S " + \
                "confs={} effort={} ligands.sdf".format(self.params['confs'], self.params['effort'])
            run_command(cmd)

            # after docking finishes, the results should be stored in the file {docking_project_name}_ligands1.ob
            # convert the icm object file to sdf file
            answer_ob_file = self.docking_project_name + "_ligands1.ob"
            answer_sdf_file = self.docking_project_name + "_ligands1.sdf"
            raise NotImplementedError()

        # finally break down the hitlist sdf file into individual sdf files for each ligand, and record their scores
        mol_supplier = Chem.SDMolSupplier(answer_sdf_file)
        for idx, lig in enumerate(ligands):
            ligand_name = Path(lig).stem
            mol = mol_supplier[idx]

            # record the smiles and score
            ligand_smiles.append(Chem.MolToSmiles(mol))
            ligand_scores.append(float(mol.GetProp("Score")))

            # save the conformation to a separate sdf file
            # find the first available name
            if os.path.exists(output_dir / f"{ligand_name}.sdf"):
                i = 1
                while os.path.exists(output_dir / f"{ligand_name}_{i}.sdf"):
                    i += 1
                ligand_name = f"{ligand_name}_{i}"
            final_name = output_dir / f"{ligand_name}.sdf"

            with Chem.SDWriter(final_name) as writer:
                writer.write(mol)

            ligand_conformation_paths.append(final_name)
        
        # generate the final pandas dataframe and return
        df = pd.DataFrame({"smiles": ligand_smiles, "score": ligand_scores, "path": ligand_conformation_paths})
        df["index"] = df.index
        return df
