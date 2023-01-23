from pathlib import Path

# CHANGE THESE UNDER DIFFERENT SYSTEMS
# (LAWRENCIUM)
ADFR_install_path = '/global/home/users/jerry-li1996/src/ADFRsuite_x86_64Linux_1.0/bin'
meeko_install_path = '/global/home/users/jerry-li1996/.conda/envs/iMiner/bin'
iMiner_path = '/global/scratch/users/jerry-li1996/iMiner/iMiner'
ad4gpu_path = '/global/home/groups/co_armada2/local/AutoDock-GPU/bin/autodock_gpu_128wi'

# (SAVIO)
# ADFR_install_path = '/global/home/groups/co_armada2/local/ADFRsuite/bin'
# meeko_install_path = '/global/home/groups/co_armada2/conda_envs/vina/bin'
# iMiner_path = '/global/home/groups/co_armada2/avidd/iMiner/iMiner'
# ad4gpu_path = '/global/home/groups/co_armada2/local/AutoDock-GPU/bin/autodock_gpu_128wi'
######################################

protein_prep_path = f'{ADFR_install_path}/prepare_receptor'
meeko_ligprep_path = f"{meeko_install_path}/mk_prepare_ligand.py"
meeko_ligconv_path = f"{meeko_install_path}/mk_copy_coords.py"

autogrid_path = f'{ADFR_install_path}/autogrid4'

VINA_BINARY = Path(iMiner_path) / 'docking/bins/vina'
VINA_GPU_SCRIPT = Path(iMiner_path) / 'docking/run_vina_gpu.sh'
VINA_GPU_BINARY_PATH = Path(iMiner_path) / "docking/bins"
