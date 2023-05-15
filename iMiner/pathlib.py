from pathlib import Path
import os

# CHANGE THESE UNDER DIFFERENT SYSTEMS
# (LOCAL MACHINE)
ADFR_install_path = '/home/jerry/src/ADFRsuite_x86_64Linux_1.0/bin'
meeko_install_path = '/home/jerry/anaconda3/envs/iMiner/bin'
iMiner_path = '/home/jerry/data/iMiner/iMiner'
ad4gpu_path = '/Users/jerryli/Downloads/AutoDock-GPU/bin/autodock_gpu_128wi'
tankbind_python_path = '/home/jerry/anaconda3/envs/tankbind/bin/python'
tankbind_src_path = '/home/jerry/data/TankBind'
p2rank_path = '/home/jerry/src/p2rank_2.4/prank'

# (LAWRENCIUM)
# ADFR_install_path = '/global/home/users/jerry-li1996/src/ADFRsuite_x86_64Linux_1.0/bin'
# meeko_install_path = '/global/home/users/jerry-li1996/.conda/envs/iMiner/bin'
# iMiner_path = '/global/scratch/users/jerry-li1996/iMiner/iMiner'
# ad4gpu_path = '/global/home/groups/co_armada2/local/AutoDock-GPU/bin/autodock_gpu_128wi'
pythonsh_path = "/global/scratch/users/jerry-li1996/covid/rdkit_vina/bin/pythonsh"
ligprep_path = "/global/scratch/users/jerry-li1996/covid/rdkit_vina/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
        

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

tankbind_dir_path = os.path.dirname(tankbind_python_path)

rcPath = {
    "tankbind_python_path": '/home/jerry/anaconda3/envs/tankbind/bin/python',
    "tankbind_src_path": '/home/jerry/data/TankBind',
    "p2rank_path": '/home/jerry/src/p2rank_2.4/prank',
}

def use_savio_path_config():
    paths = [
        "/global/home/groups/co_armada2/local/ADFRsuite/bin",
        Path(__file__).parent / "docking/bins"
    ]
    os.environ['PATH'] = os.environ.get("PATH") + ":" + ":".join(str(x) for x in paths)

def use_gcc_740_savio():
    env = dict(
        CPATH="/global/software/sl-7.x86_64/modules/langs/gcc/7.4.0/include",
        FPATH="/global/software/sl-7.x86_64/modules/langs/gcc/7.4.0/include",
        GCC_DIR="/global/software/sl-7.x86_64/modules/langs/gcc/7.4.0",
        INCLUDE="/global/software/sl-7.x86_64/modules/langs/gcc/7.4.0/include",
        PATH=f"/global/software/sl-7.x86_64/modules/langs/gcc/7.4.0/bin:{os.environ['PATH']}",
        LD_LIBRARY_PATH=f"/global/software/sl-7.x86_64/modules/langs/gcc/7.4.0/lib64:{os.environ['LD_LIBRARY_PATH']}",
        LIBRARY_PATH=f"/global/software/sl-7.x86_64/modules/langs/gcc/7.4.0/lib64:{os.environ['LIBRARY_PATH']}"
    )
    os.environ.update(env)
        