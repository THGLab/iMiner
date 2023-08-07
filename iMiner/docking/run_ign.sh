export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/scratch/users/ozhang/.conda/envs/ign/lib

conda run -n ign python /global/scratch/users/ozhang/InteractionGraphNet/codes/score.py --protein ${1} --ligands ${2} --graph_dic_path ${3} --output ${4} --num_process ${5}