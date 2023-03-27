from pathlib import Path
import os

ADFR_install_path = '/global/home/groups/co_armada2/local/ADFRsuite/bin'
meeko_install_path = '/global/home/groups/co_armada2/conda_envs/vina/bin'
iMiner_path = '/global/home/users/kysun/iMiner-github/iMiner'
ad4gpu_path = '/global/home/groups/co_armada2/local/AutoDock-GPU/bin/autodock_gpu_128wi'
ad4gpu_analysis_path = '/global/home/groups/co_armada2/local/AutoDock-GPU/bin/adgpu_analysis'

protein_prep_path = f'{ADFR_install_path}/prepare_receptor'
pythonsh_path = f'{ADFR_install_path}/pythonsh'
flexrec_prep_path = Path(iMiner_path) / "prepare_flexreceptor.py"
meeko_ligprep_path = f"{meeko_install_path}/mk_prepare_ligand.py"
meeko_ligconv_path = f"{meeko_install_path}/mk_copy_coords.py"

autogrid_path = f'{ADFR_install_path}/autogrid4'

VINA_BINARY = Path(iMiner_path) / 'docking/bins/vina'
VINA_GPU_SCRIPT = Path(iMiner_path) / 'docking/run_vina_gpu.sh'
VINA_GPU_BINARY_PATH = Path(iMiner_path) / "docking/bins"

tankbind_python_path = '/home/jerry/anaconda3/envs/tankbind/bin/python'
tankbind_src_path = '/home/jerry/data/TankBind'
tankbind_dir_path = os.path.dirname(tankbind_python_path)
p2rank_path = '/home/jerry/src/p2rank_2.4/prank'
