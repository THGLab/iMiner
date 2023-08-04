from pathlib import Path
import os

# CHANGE THESE UNDER DIFFERENT SYSTEMS

# (LAWRENCIUM)
ign_python_path = '/global/scratch/users/ozhang/.conda/envs/ign/bin/python3.7'
ign_src_path = '/global/scratch/users/ozhang/InteractionGraphNet/codes'
ign_dir_path = os.path.dirname(ign_python_path)

ADFR_install_path = '/global/scratch/users/jerry-li1996/src/ADFRsuite_x86_64Linux_1.0/bin'
meeko_install_path = '/global/scratch/users/ozhang/env/iminer-rl/bin'
iMiner_path = '/global/scratch/users/ozhang/covid/iMiner/iMiner'
ad4gpu_path = '/global/scratch/users/jerry-li1996/bins/AutoDock-GPU/bin/autodock_gpu_128wi'

# (SAVIO)
# ADFR_install_path = '/global/home/groups/co_armada2/local/ADFRsuite/bin'
# meeko_install_path = '/global/home/groups/co_armada2/conda_envs/vina/bin'
# iMiner_path = '/global/home/groups/co_armada2/avidd/iMiner/iMiner'
# ad4gpu_path = '/global/home/groups/co_armada2/local/AutoDock-GPU/bin/autodock_gpu_128wi'
######################################

protein_prep_path = f'{ADFR_install_path}/prepare_receptor'
meeko_ligprep_path = f"{meeko_install_path}/mk_prepare_ligand.py"
meeko_ligconv_path = f"{meeko_install_path}/mk_export.py"

autogrid_path = f'{ADFR_install_path}/autogrid4'

VINA_BINARY = Path(iMiner_path) / 'docking/bins/vina'
VINA_GPU_SCRIPT = Path(iMiner_path) / 'docking/run_vina_gpu.sh'
VINA_GPU_BINARY_PATH = Path(iMiner_path) / "docking/bins"
IGN_SCRIPT = Path(iMiner_path) / 'docking/run_ign.sh'

RF_model_path = Path(iMiner_path) / 'docking/bins/RF_newsplit.pkl'
RF_col_mask = [True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True,  True,  True,  True,
               False, False, False, False, False, False, False, False, False,
               False, False, False, False, False, False, False, False, False,
               True,  True,  True,  True,  True,  True,  True,  True,  True,
               False, False, False, False, False, False, False, False, False,
               False, False, False, False, False, False, False, False, False,
               False, False, False, False, False, False, False, False, False]