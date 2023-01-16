# this script is used to set up all required modules on Lawrencium ES1 partition and then run Vina-GPU binary

module load gcc/7.4.0
module load boost
ulimit -s 8192
VINA_GPU=/global/scratch/users/jerry-li1996/iMiner/iMiner/docking/bins/Vina-GPU
$VINA_GPU --config ${1} --ligand ${2}.pdbqt --out ${2}_out.pdbqt
