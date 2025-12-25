#!/bin/sh
#SBATCH --job-name="Sample Job REL-TT"
#SBATCH --partition=ewi-st,general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

export APPTAINER_IMAGE="daic.sif"
export DATASET="rel-f1"
export TASK="driver-position"
export HOME="/tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/tt-rdl"

# If you use GPUs
module use /opt/insy/modulefiles
module load cuda/11.8

mkdir $HOME/scratch
ln -s $HOME/.cache/relbench $HOME/scratch/relbench

srun apptainer exec -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/tt-rdl:/workspace --pwd workspace/rustler $APPTAINER_IMAGE maturin develop --uv --release

srun apptainer exec -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/tt-rdl/ $APPTAINER_IMAGE python download_task.py --dataset $DATASET --task $TASK

srun apptainer exec -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/tt-rdl:/workspace --pwd workspace/rustler $APPTAINER_IMAGE cargo run --release -- pre $DATASET

srun apptainer exec -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/tt-rdl $APPTAINER_IMAGE python src/embed.py $DATASET
srun apptainer exec --nv --env-file .env -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/tt-rdl $APPTAINER_IMAGE python main.py  --save_artifacts --dataset $DATASET --task $TASK --eval_freq 1000 --seed 1