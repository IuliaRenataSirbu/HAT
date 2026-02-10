#!/bin/bash

DATA_PATH="/export/projects/nlp/datasets/BMAD_Benchmark/BraTS2021_slice"
SIF_PATH="/export/projects/nlp/containers/duti_v19.sif"

# export TOKENIZERS_PARALLELISM=false


mkdir -p logs

sbatch -p dgxa100 \
    --time=0-12 \
    --mem=64G \
    --gres gpu:1 \
    --job-name=BrainSR \
    --output=logs/train_%j.log \
    ~/commons/apptainer-exec-nlp.sh /export/projects/nlp/containers/hat3.sif \
"hat/train.py -opt options/train/BrainTS_train_HAT_SRx4_from_scratch.yml"
