from pathlib import Path
import os

# CHANGE THESE UNDER DIFFERENT SYSTEMS

# (LAWRENCIUM)
# ADFR_install_path = '/global/scratch/users/jerry-li1996/src/ADFRsuite_x86_64Linux_1.0/bin'
# meeko_install_path = '/global/scratch/users/ozhang/env/iminer-rl/bin'
# iMiner_path = '/global/scratch/users/ozhang/covid/iMiner'
# ad4gpu_path = '/global/home/groups/co_armada2/local/AutoDock-GPU/bin/autodock_gpu_128wi'


# savio
ADFR_install_path = '/global/home/groups/co_armada2/local/ADFRsuite/bin'
meeko_install_path = '/global/home/groups/co_armada2/conda_envs/vina/bin'
iMiner_path = '/global/home/users/kysun/iMiner-github/'
ad4gpu_path = '/global/home/groups/co_armada2/local/AutoDock-GPU/bin/autodock_gpu_128wi'
ad4gpu_analysis_path = '/global/home/groups/co_armada2/local/AutoDock-GPU/bin/adgpu_analysis'
flexrec_prep_path = f'{ADFR_install_path}/prepare_flexreceptor.py'

protein_prep_path = f'{ADFR_install_path}/prepare_receptor'
pythonsh_path = f'{ADFR_install_path}/pythonsh'
meeko_ligprep_path = f"{meeko_install_path}/mk_prepare_ligand.py"
meeko_ligconv_path = f"{meeko_install_path}/mk_export.py"

autogrid_path = f'{ADFR_install_path}/autogrid4'

VINA_BINARY = Path(iMiner_path) / 'iMiner/docking/bins/vina'
VINA_GPU_SCRIPT = Path(iMiner_path) / 'iMiner/docking/run_vina_gpu.sh'
VINA_GPU_BINARY_PATH = Path(iMiner_path) / "iMiner/docking/bins"

# tankbind_dir_path = os.path.dirname(tankbind_python_path)
# tankbind_python_path = '/home/jerry/anaconda3/envs/tankbind/bin/python'
# tankbind_src_path = '/home/jerry/data/TankBind'
# p2rank_path = '/home/jerry/src/p2rank_2.4/prank'

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
