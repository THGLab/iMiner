import os, sys, glob, shutil
from pathlib import Path
import json
from typing import Union, Optional, Dict, Any, Callable

from iMiner.cmd import set_directory, run_command
from iMiner.log import init_logger
from .prep import run_tleap


leap_test = '''
source leaprc.water.tip3p
loadAmberParams frcmod.ionsjc_tip3p
source leaprc.protein.ff14SB

protein = loadpdb protein.pdb

savepdb protein protein_after_test.pdb
saveamberparm protein protein.prmtop protein.inpcrd

quit
'''


leap_ions_str = '''
source leaprc.water.tip3p
loadAmberParams frcmod.ionsjc_tip3p
source leaprc.protein.ff14SB

protein = loadpdb protein.pdb

solvatebox protein TIP3PBOX {buffer:.1f} 0.5

quit
'''


class AmberRbfeProject:
    """
    Project for a Amber Relative Binding Free Energy Calculation
    """
    def __init__(self, wdir: os.PathLike = '.'):
        """
        Initialize a AmberRbfeProject instance

        Parameters
        ----------
        wdir: os.PathLike
            Working directory of the project
        """
        self.wdir = Path(wdir).resolve()
        self.ligands_dir = self.wdir / 'ligands'
        self.proteins_dir = self.wdir / 'proteins'
        self.rbfe_dir = self.wdir / 'rbfe'
        self.logger = init_logger()

        self.wdir.mkdir(exist_ok=True)
        self.ligands_dir.mkdir(exist_ok=True)
        self.proteins_dir.mkdir(exist_ok=True)
        self.rbfe_dir.mkdir(exist_ok=True)

    def add_ligand(
        self, 
        fpath: os.PathLike, 
        name: Optional[str] = None, 
        parametrize: bool = True, 
        forcefield: str = 'gaff2', 
        charge_method: str = 'bcc', 
        net_charge: Union[str, int] = 'auto',
        overwrite: bool = False,
    ):
        """
        Add ligand to the project and use ACPYPE parametrize it

        Parameters
        ----------
        fpath: os.PathLike
            Path to the ligand file. Only sdf format is supported now
        name: str
            Name of the ligand. If None, will be inferred from the input path
        parametrize: bool
            Whether to parameterize the ligand.
        forcefield: str
            Force field used to parametrize the ligand, i.e. "-a" option in acpype.
            'gaff' or 'gaff2' is accecpted. Default is 'gaff2'. 
        charge_method: str
            Method to assign atomic charges, i.e. "-c" option in acpype.
            'gas' or 'bcc' is accepted. Default is 'bcc'
        net_charge: str or int
            Net charge of the ligand, i.e. '-n' option in acpype.
            If 'auto', rdkit will be used to calculate the total net charge of the molecule.
            Default is 'auto'
        overwrite: bool
            Whether to overwrite existing ligand directory. Default False.
        """
        suffix = Path(fpath).suffix
        name = Path(fpath).stem if name is None else name
        assert suffix == '.sdf', 'Only sdf format is supported'
        lig_dir = self.ligands_dir / name
        if overwrite and lig_dir.is_dir():
            shutil.rmtree(lig_dir)
            self.logger.warn(f'Will overwrite all files in {lig_dir}')
        lig_dir.mkdir()
        flig = lig_dir / f'{name}{suffix}'
        shutil.copyfile(fpath, flig)
        self.logger.info(f'Ligand {name} is added: {flig}')
        
        if parametrize:
            self.parametrize_ligand(name, forcefield, charge_method, net_charge)

    def parametrize_ligand(self, name, forcefield, charge_method, net_charge):

        from ..prep.ligand import run_acpype

        lig_dir = self.ligands_dir / name
        self.logger.info(f"Parametrizing ligand: {name}")
        with set_directory(lig_dir):
            run_acpype(f'{name}.sdf', 'MOL', charge_method, forcefield, net_charge)
        self.logger.info(f"Ligand {name} is parametrized with {forcefield} and charge method {charge_method}")

    def add_protein(self, fpath: os.PathLike, name: Optional[str] = None, check_ff: bool = True, overwrite: bool = False):
        """
        Add protein to the project

        Parameters
        ----------
        fpath: os.PathLike
            Path to the protein file to be added. Only pdb format is supported
        name: str
            Name of the protein. If None, will be inferred from the input path
        check_ff: bool
            Whether to check the protein can be parametrized by Amber14SB force field.
        overwrite: bool
            Whether overwrite existing protein directory. Default False.
        """
        suffix = Path(fpath).suffix
        name = Path(fpath).stem if name is None else name
        assert suffix == '.pdb', 'Only PDB format is supported'
        prot_dir = self.proteins_dir / name
        if overwrite and prot_dir.is_dir():
            shutil.rmtree(prot_dir)
            self.logger.warn(f'Will overwrite all files in {prot_dir}')
        prot_dir.mkdir()
        fprot = prot_dir / f'{name}{suffix}'
        shutil.copyfile(fpath, fprot)
        self.logger.info(f'Protein {name} is added: {fprot}')

        if check_ff:
            self.logger.info("Checking if protein can be parametrized with Amber force field")
            test_dir = prot_dir / 'test'
            if test_dir.is_dir():
                shutil.rmtree(test_dir)
            test_dir.mkdir()

            with set_directory(test_dir):
                os.symlink(f'../{name}{suffix}', 'protein.pdb')
                with open('leap.in', 'w') as f:
                    f.write(leap_test)
                try:
                    run_tleap('leap.in')
                    self.logger.info('Force field check Passed.')
                except Exception as e:
                    self.logger.error(f'The protein can not be parametrized with Amber force field. Please see {test_dir / "leap.log"} for details')
                    sys.exit(1)
    
    def add_perturbation(
        self, 
        ligandA_name: str, ligandB_name: str, protein_name: str, 
        pert_name: Optional[str] = None, 
        mcs: Optional[Union[str, os.PathLike]] = None, 
        config: Optional[Union[Dict[str, Any], os.PathLike]] = None,
        submit: bool = False,
        skip_gas: bool = True,
        overwrite: bool = False,
        patch_func: Optional[Callable] = None
    ):
        """
        Create a perturbation pair (relative binding free energy between ligand A and ligand B) 
        and set up FEP simulation

        Parameters
        ----------
        ligandA_name: str
            Name of ligand A
        ligandB__name: str
            Name of ligand B
        protein_name: str
            Name of the protein
        pert_name: str
            Name of this perturbation. If None, will use `ligandA_name`~`ligandB_name`
        mcs: str or os.PathLike
            Maximum common structure (MCS) between ligand A and B. If a sdf file or a SMARTS string is provided, rdkit will be used 
            to parse it. If None, will use rdkit to determine the MCS. Manually provided MCS is highly recommended.
        config: dict or os.PathLike
            JSON-formatted configuration file or dictionary. Must be provided.
        submit: bool
            Whether to use SLURM (sbatch command) to submit the simulation job. Default False.
        skip_gas: bool
            Whether to skip the gas-phase FEP simulation, only useful when `submit=True`. Default True.
        overwrite: bool
            Whether to overwrite existing perturbation directory. Default False.
        patch_func: Callable
            Function to modify the regular behavior of this function, used for modifying force field parameters
        """
        from rdkit import Chem
        from .mcs import find_mcs, get_common_core, generate_mask, check_common_core
        from .op import fep_workflow
        from .prep import determine_num_ions_from_leap_log

        if config is None:
            raise RuntimeError("Configuration file must be provided")
        
        if pert_name is None:
            pert_name = f"{ligandA_name}~{ligandB_name}"
        
        pert_dir = self.rbfe_dir / pert_name
        if overwrite and pert_dir.is_dir():
            shutil.rmtree(pert_dir)
            self.logger.warn(f'Will overwrite all files in {pert_dir}')
        pert_dir.mkdir()
        self.logger.info(f'Creating dirctory: {pert_dir}')

        # Config
        if not isinstance(config, dict):
            self.logger.info(f'Read config from {config}')
            with open(config) as f:
                config = json.load(f)
            
        # Read ligands
        molA = Chem.SDMolSupplier(str(self.ligands_dir / f'{ligandA_name}/{ligandA_name}.sdf'), removeHs=False)[0]
        molB = Chem.SDMolSupplier(str(self.ligands_dir / f'{ligandB_name}/{ligandB_name}.sdf'), removeHs=False)[0]
        posA = molA.GetConformer().GetPositions()
        posB = molB.GetConformer().GetPositions()
        
        # Prepare MCS structure
        if mcs is None:
            self.logger.info("No MCS file is provided. Try to use RDKit to find MCS")
            mcs_struct = find_mcs(molA, molB)
        elif os.path.isfile(mcs):
            self.logger.info(f"Reading mcs from {mcs}")
            assert Path(mcs).suffix == '.sdf', 'Only sdf format is supported'
            mcs_struct = Chem.SDMolSupplier(str(mcs))[0]
        else:
            self.logger.info(f"Find mcs SMARTS: {mcs}. Try to use RDKit to parse it.")
            mcs_struct = Chem.MolFromSmarts(mcs)
        
        with Chem.SDWriter(str(pert_dir / 'mcs.sdf')) as w:
            w.write(mcs_struct)
        self.logger.info(f'MCS written to {pert_dir / "mcs.sdf"}')
        
        cc = get_common_core(molA, molB, mcs_struct)
        with open(pert_dir / 'common_core.txt', 'w') as f:
            for c in cc:
                f.write(f"{c[0]} {c[1]}\n")
        self.logger.info(f'Write common core indices to: {pert_dir / "common_core.txt"}')
        try:
            check_common_core(posA, posB, cc)
        except Exception as e:
            self.logger.error(f"Common core checking failed: {e}")
            sys.exit(1)

        mask = generate_mask(molA.GetNumAtoms(), molB.GetNumAtoms(), cc[:, 0], cc[:, 1])

        with open(pert_dir / 'mask.json', 'w') as f:
            json.dump(mask, f, indent=4)
        self.logger.info(f"Found MCS. Write scmask, timask to {pert_dir / 'mask.json'}")


        with set_directory(pert_dir):
            # link protein and ligands
            os.symlink(f"../../ligands/{ligandA_name}", "ligandA")
            os.symlink(f"../../ligands/{ligandB_name}", "ligandB")

            prep_dir = pert_dir / 'prep'
            prep_dir.mkdir()
            with set_directory(prep_dir):
                os.symlink("../ligandA/MOL.acpype/MOL_AC.frcmod", "ligandA.frcmod")
                os.symlink("../ligandA/MOL.acpype/MOL_AC.lib", "ligandA.lib")
                os.symlink(list(glob.glob('../ligandA/MOL.acpype/*mol2'))[0], "ligandA.mol2")
                os.symlink("../ligandB/MOL.acpype/MOL_AC.frcmod", "ligandB.frcmod")
                os.symlink("../ligandB/MOL.acpype/MOL_AC.lib", "ligandB.lib")
                os.symlink(list(glob.glob('../ligandB/MOL.acpype/*mol2'))[0], "ligandB.mol2")
                os.symlink(f"../../../proteins/{protein_name}/{protein_name}.pdb", "protein.pdb")
                
                # Calculate ions
                with open('leap_ions.in', 'w') as f:
                    f.write(leap_ions_str.format(buffer=config.get('complex', {}).get('buffer', 12.0)))
                
                try:
                    ionic_strength = config.get('complex', {}).get('ionic_strength', 0.15)
                    self.logger.info(f"Try to determine number of ions needs to be added for {ionic_strength} mol/L ionic strength")
                    run_tleap('leap_ions.in')
                    num_ions = determine_num_ions_from_leap_log('leap.log', ionic_strength)    
                    os.remove('leap.log')
                    self.logger.info(f'Number of ions to be added: {num_ions}')
                except:
                    self.logger.info(f'Error occurs. See {prep_dir / "leap.log"}')
                    sys.exit(1)
                
                # Ligands/Complex legs prmtop/inpcrd
                with open(Path(__file__).parent / 'leap.in') as f:
                    leap_str = f.read()
                    leap_str = leap_str.replace('@COMPLEX_BUFFER', f"{config.get('complex', {}).get('buffer', 12.0):.1f}")
                    leap_str = leap_str.replace('@LIGANDS_BUFFER', f"{config.get('ligands', {}).get('buffer', 15.0):.1f}")
                    leap_str = leap_str.replace('@NUM_IONS', str(num_ions))
                with open('leap.in', 'w') as f:
                    f.write(leap_str)

                try:
                    self.logger.info("Preparing ligands/complex FEP simulation prmtop/inpcrd using tleap")
                    run_tleap('leap.in')
                except:
                    self.logger.error(f'Error occurs. See {prep_dir / "leap.log"}')
                    sys.exit(1)
                
                # Gas phase prmtop/inpcrd
                with open(Path(__file__).parent / 'leap_gas.in') as f:
                    leap_gas_str = f.read().replace("@GAS_BUFFER", f"{config.get('gas', {}).get('buffer', 20.0):.1f}")
                with open('leap_gas.in', 'w') as f:
                    f.write(leap_gas_str)
                
                try:
                    self.logger.info("Preparing gas phase FEP simulation prmtop/inpcrd using tleap")
                    run_tleap('leap_gas.in')
                except:
                    self.logger.error(f'Error occurs. See {prep_dir / "leap.log"}')
                    sys.exit(1)
                
                # Prep workflow
                for leg in ['gas', 'ligands', 'complex']:
                    leg_dir = pert_dir / leg
                    leg_dir.mkdir()
                    if leg != 'gas':
                        prmtop = pert_dir / f'prep/{leg}_solvated.prmtop'
                        inpcrd = pert_dir / f'prep/{leg}_solvated.inpcrd'
                    else:
                        prmtop = pert_dir / f'prep/ligands_gas.prmtop'
                        inpcrd = pert_dir / f'prep/ligands_gas.inpcrd'

                    with set_directory(leg_dir):
                        leg_config = {
                            "inpcrd": os.path.relpath(inpcrd, leg_dir),
                            "prmtop": os.path.relpath(prmtop, leg_dir),
                        }
                        leg_config.update(config[leg])
                        leg_config.update(mask)

                        with open('config.json', 'w') as f:
                            json.dump(leg_config, f, indent=4)
                    
                        fep_workflow(leg_config, leg_dir, gas_phase=(leg == 'gas'))
                        self.logger.info(f"FEP simulation workflow is set for leg: {leg}. Config file written to: {leg_dir / 'config.json'}")

                        with open(Path(__file__).parent / 'submit.slurm') as f:
                            slurm = f.read()
                            slurm = slurm.replace('@SLURM_CONFIG', '\n'.join(config['slurm_config']))
                            slurm = slurm.replace('@SET_UP_ENV', f'source {config["env"]}' if 'env' in config else '')
                            slurm = slurm.replace('@NUM_LAMBDA', str(len(config[leg]['lambdas'])))
                            slurm = slurm.replace(
                                '@STAGES', 
                                '("em" "heat" "pres_0" "pres_1" "pres_2" "pre_prod")' if leg != 'gas' else '("em" "heat")'
                            )
                        
                        with open('run.slurm', 'w') as f:
                            f.write(slurm)
                
                # force field patch
                if patch_func is not None:
                    patch_func(self, pert_name)
                
                if submit:
                    legs = ['ligands', 'complex'] if skip_gas else ['ligands', 'complex', 'gas']
                    for leg in legs:
                        with set_directory(pert_dir / leg):
                            _, out, _ = run_command(['sbatch', 'run.slurm'])
                        self.logger.info(f"Job submitted for {leg}: {out.split()[-1]}")
        
    def analyze(self, pert_name: str, skip_gas: bool = True):
        """
        Analyze FEP simulation results using `alchemlyb` package.
        Free energy will be estimated using MBAR, overlap matrix and convergence analysis are also performed.

        For details, one can refer to: J Comput Aided Mol Des (2015) 29:397-411

        Parameters
        ----------
        pert_name: str
            Name of the perturbation to analyze
        skip_gas: bool
            Whether to skip gas-phase simulation analysis. Default is True.
        """
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        import alchemlyb
        from alchemlyb.estimators import MBAR
        from alchemlyb.parsing.amber import extract_u_nk
        from alchemlyb.convergence import forward_backward_convergence
        from alchemlyb.visualisation.convergence import plot_convergence
        from alchemlyb.visualisation.mbar_matrix import plot_mbar_overlap_matrix

        dG = {}
        dG_std = {}

        ddG = {}
        ddG_std = {}

        pert_dir = self.rbfe_dir / pert_name

        legs = ['ligands', 'complex'] if skip_gas else ['ligands', 'complex', 'gas']
        for leg in legs:
            leg_dir = pert_dir / leg
            with open(leg_dir / 'config.json') as f:
                T = json.load(f).get('temperature', 298.15)
                kBT = 8.314 * T / 1000 / 4.184

            self.logger.info(f"Performing MBAR for {leg}")
            self.logger.info("Extracting data from output...")
            u_nks = []
            num_lambda = len(list(leg_dir.glob('lambda*')))
            for i in tqdm(range(num_lambda), leave=True):
                out = str(leg_dir / f"lambda{i}/prod/prod.out")
                u_nks.append(extract_u_nk(out, T=T))
            
            # evaluate free energy with MBAR
            self.logger.info("Running MBAR estimator...")
            mbarEstimator = MBAR()
            mbarEstimator.fit(alchemlyb.concat(u_nks))
            dG[leg] = mbarEstimator.delta_f_.iloc[0, -1] * kBT
            dG_std[leg] = mbarEstimator.d_delta_f_.iloc[0, -1] * kBT
            
            # convergence analysis
            self.logger.info("Running convergence analysis...")
            conv_df = forward_backward_convergence(u_nks, "mbar")
            for key in ['Forward', 'Forward_Error', 'Backward', 'Backward_Error']:
                conv_df[key] *= kBT
                conv_df.to_csv(leg_dir /"convergence.csv", index=None)
                conv_ax = plot_convergence(conv_df)
                conv_ax.set_ylabel("$\Delta G$ (kcal/mol)")
                conv_ax.set_title(f"Convergence Analysis - {leg.capitalize()}")
                conv_ax.figure.savefig(str(leg_dir /"convergence.png"), dpi=300)

            # overlap matrix
            self.logger.info("Plotting overlap matrix...")
            overlap_ax = plot_mbar_overlap_matrix(mbarEstimator.overlap_matrix)
            overlap_ax.figure.savefig(str(leg_dir /"overlap.png"), dpi=300)

        convergence = {leg: pd.read_csv(str(pert_dir / leg / 'convergence.csv')) for leg in legs}

        pairs = [('complex', 'ligands'), ('ligands', 'gas'), ('complex', 'gas')]
        names = ['total', 'solvation', 'complex']

        for (leg1, leg2), name in zip(pairs, names):
            ddG[name] = dG[leg1] - dG[leg2]
            ddG_std[name] = np.linalg.norm([dG_std[leg1], dG_std[leg2]])

            ddG_conv_df = convergence['complex'].copy()
            for tag in ['Forward', 'Backward']:
                ddG_conv_df[tag] = convergence[leg1][tag] - convergence[leg2][tag]
                ddG_conv_df[f'{tag}_Error'] = np.sqrt(
                    convergence[leg1][f'{tag}_Error'] ** 2 + convergence[leg2][f'{tag}_Error'] ** 2
                )
            ddG_conv_df.to_csv(pert_dir / f"{name}_convergence.csv", index=None)
            conv_ax = plot_convergence(ddG_conv_df)
            conv_ax.set_ylabel("$\Delta\Delta G$ (kcal/mol)")
            conv_ax.set_title(f"Convergence Analysis")
            conv_ax.figure.savefig(str(pert_dir / f"{name}_convergence.png"), dpi=300)

            if skip_gas:
                break

        with open(pert_dir / 'result.json', 'w') as f: 
            json.dump({"dG": dG, "dG_std": dG_std, 'ddG': ddG, 'ddG_std': ddG_std}, f, indent=4)
        self.logger.info("Finished!")