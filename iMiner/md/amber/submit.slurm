#!/bin/bash
@SLURM_CONFIG

@SET_UP_ENV
echo "Allocated GPU IDs: $CUDA_VISIBLE_DEVICES on $(hostname)"

num_lambda=@NUM_LAMBDA
prep_stages=@STAGES

last_stage=${prep_stages[-1]}
for n in `seq 0 $((num_lambda - 1))`
do
    if [ -f lambda${n}/${last_stage}/${last_stage}.rst7 ]; then
        echo "Found finished equilibrium process in lambda${n}"
    else
        for stage in ${prep_stages[@]}
        do
            cd lambda${n}/${stage}
            echo "Running ${stage} for lambda ${n}"
            source ${stage}.sh > ${stage}.stdout 2>&1
            if [ $? -ne 0 ]; then
                echo "Error occurs!"
                exit 1
            fi
            cd ../..
        done
    fi
done

echo "Running Production..."
mpirun -np $num_lambda pmemd.cuda.MPI -rem 3 -ng $num_lambda -groupfile prod.groupfile -remlog prod.log

if [ $? -eq 0 ]; then
  touch done.tag
else
  echo "Error occurs!"
  exit 1
fi

for n in `seq 0 $((num_lambda - 1))`
do
    cd lambda${n}/prod
    if [ -f prod.rst7 ]; then
        source prod.sh
    fi
    cd ../../
done