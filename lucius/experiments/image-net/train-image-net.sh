#!/bin/bash

SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CUDA_DEVICES="0"
INPUT_TRAINING_DATASET="/data/1tb-ssd/image-net-50-2000/training/database.txt"
INPUT_VALIDATION_DATASET="/data/1tb-ssd/image-net-50-2000/validation/database.txt"
LOGS="NesterovAcceleratedGradientSolver,BenchmarkImageNet,LabelMatchResultProcessor,InputVisualDataProducer"
LAYER_SIZE="1024"
INPUT_WIDTH="112"
INPUT_HEIGHT="112"
INPUT_COLORS="3"
LAYERS="13"
MINI_BATCH_SIZE="128"
LAYER_OUTPUT_REDUCTION_FACTOR="1"
EPOCHS="20"
LEARNING_RATE="0.01"
MOMENTUM="0.9"
RESUME_FROM="/data/1tb-ssd/checkout/video-classifier/lucius/experiments/image-net/benchmark-image-net-112-width-112-height-3-colors-1024-layer-size-13-layers-128-mini-batch-0.01-learning-rate-0.9-momentum-1-reduction-factor/model.tar"

EXPERIMENT_NAME="benchmark-image-net-$INPUT_WIDTH-width-$INPUT_HEIGHT-height-$INPUT_COLORS-colors-$LAYER_SIZE-layer-size-$LAYERS-layers-$MINI_BATCH_SIZE-mini-batch-$LEARNING_RATE-learning-rate-$MOMENTUM-momentum-$LAYER_OUTPUT_REDUCTION_FACTOR-reduction-factor-resume"

EXPERIMENT_DIRECTORY="$SCRIPT_DIRECTORY/$EXPERIMENT_NAME"
LOG_FILE="$EXPERIMENT_DIRECTORY/log"
MODEL_FILE="$EXPERIMENT_DIRECTORY/model.tar"
VALIDATION_ERROR_FILE="$EXPERIMENT_DIRECTORY/validation-error.csv"

mkdir -p $EXPERIMENT_DIRECTORY

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

COMMAND="benchmark-imagenet -l $LAYER_SIZE -e $EPOCHS -b $MINI_BATCH_SIZE -f $LAYER_OUTPUT_REDUCTION_FACTOR \
         --momentum $MOMENTUM --learning-rate $LEARNING_RATE -c $INPUT_COLORS -x $INPUT_WIDTH -y $INPUT_HEIGHT \
         -i $INPUT_TRAINING_DATASET -t $INPUT_VALIDATION_DATASET -L $LOGS --log-file $LOG_FILE \
         -o $MODEL_FILE -r $VALIDATION_ERROR_FILE"

if [[ -n $RESUME_FROM ]]
then
    COMMAND="$COMMAND -m $RESUME_FROM"
fi

echo $COMMAND

$COMMAND

