from pathlib import Path
from iMiner.md.amber.rbfe import AmberRbfeProject

'''
> For THG group members using UCBerkeley Savio platform, one can use this conda environment: 
`conda acitvate /global/home/groups/fc_armada2/conda_envs/iminer`

Dependencies:

+ Python >= 3.8
+ numpy
+ scipy
+ tqdm
+ rdkit
+ alchemlyb
+ acpype
+ ambertools

'''

def test_amber_rbfe():
    data_dir = Path(__file__).parent / 'data/rbfe'
    wdir = Path(__file__).parent / 'test_rbfe_project'
    '''
    1. Initialize a RBFE Project directory
       The program will create three directories under the project directory (`wdir`):
       `ligands/`, `proteins/` and `rbfe/` 
    '''
    proj = AmberRbfeProject(wdir=wdir)
    
    '''
    2. Add ligands to the project and parametrize it
       The program needs ligands in .sdf format with correct 3d structures (binding pose)
       and explicit hydrogens. Tools like rdkit, openbabel or pymol can be used to prepare the ligand.
       The program will copy the input sdf file to `ligands/{name}/{name}.sdf` and parametrize it with
       `forcefield` and assign atomic charge using `charge_method`.
       Note that the ligands must have unique name, otherwise the program will throw an error.
    '''
    proj.add_ligand(
        data_dir / 'CDD_1845.sdf', 
        name='CDD_1845', 
        forcefield='gaff2', # use GAFF2 force field
        charge_method='gas' # use Gasteiger charges only for testing, please change it to 'bcc' when do a real calculation
    )
    proj.add_ligand(
        data_dir / 'CDD_1819.sdf', 
        name='CDD_1819', 
        forcefield='gaff2', 
        charge_method='gas'
    )

    '''
    3. Add a protein
       The program requires a PDB-formatted protein file without any missing heavy atoms and residues.
       One can use tools like pdbfixer to prepare the protein.
       A directory named `proteins/{name}` will be created and the input pdb file will be copied to 
       `proteins/{name}/{name}.pdb`
       Note that the proteins must have unique name, otherwise the program will throw an error.
       The program will check if the input pdb is compatitable with Amber force field. 
    '''
    proj.add_protein(
        data_dir / 'CDD_1845.pdb', 
        name='CDD_1845',
        check_ff=True # Disable this process by setting `check_ff` to False. Not recommended to do so
    )

    '''
    4. Add a perturbation and prepare simulation inputs
       The program will create a directory `rbfe/{pert_name}` as the working directory for this FEP simulation
       If `pert_name` is None, will use `{ligandA_name}~{ligandB_name}`
    '''
    proj.add_perturbation(
        'CDD_1845', # name of ligandA
        'CDD_1819', # name of ligandB
        'CDD_1845', # name of the protein
        pert_name=None,
        mcs=data_dir/'mcs.sdf', # path or SMARTS string of the MCS between A and B, optional (set to None)
        config=data_dir/'config.json', # config json file
        submit=False, # set to True if you want the jobs submitted automatically using Slurm (sbatch)
                      # Otherwise you can manually submit the `run.slurm` file under `rbfe/{pert_name}/complex,ligands`
                      # after calling this function
        skip_gas=True # set to False if you also want to perform a gas-phase simulation, 
                      # which is used for analyzing solvation effects
    )

    '''
    5. Analyze the results
       Call the alchemlyb to analyze the FEP results, as well as to perform overlap and convergence analysis.
       Commenting this out only for testing.

       Several files will be created uner `rbfe/{pert_name}`:
       + result.json
       + total_convergence.csv
       + total_convergence.png
       + complex/convergence.csv
       + complex/overlap.png
       + ligands/convergence.csv
       + ligands/convergence.png

       If performing the solvation analysis, the additional files are also created:
       + solvation_convergence.csv
       + solvation_convergence.png
       + complex_convergence.csv
       + complex_convergence.csv
       + gas/convergence.csv
       + gas/convergence.png

       The RBFE value is in ddG/total in result.json
    '''
    # proj.analyze(
    #     pert_name='CDD_1845~CDD_1819',
    #     skip_gas=True # set to False if you want to analyze solvation effects. 
    #                   # To do so, the gas-phase simulation must also be performed (setting skip_gas to False in `add_perturbation`)
    # )


