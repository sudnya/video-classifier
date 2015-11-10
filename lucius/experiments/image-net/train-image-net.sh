#!/bin/bash

SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CUDA_DEVICES="1"
INPUT_TRAINING_DATASET="/data/1tb-ssd/image-net-50-2000/training/database.txt"
INPUT_VALIDATION_DATASET="/data/1tb-ssd/image-net-50-2000/validation/database.txt"
LOGS="NesterovAcceleratedGradientSolver,BenchmarkImageNet,LabelMatchResultProcessor,InputVisualDataProducer"
LAYER_SIZE="1024"
INPUT_WIDTH="224"
INPUT_HEIGHT="224"
INPUT_COLORS="3"
LAYERS="13"
MINI_BATCH_SIZE="8"
LAYER_OUTPUT_REDUCTION_FACTOR="1"
BATCH_NORMALIZATION=1
EPOCHS="80"
LEARNING_RATE="0.001"
ANNEALING_RATE="1.00001"
MOMENTUM="0.9"
RESUME_FROM=""

EXPERIMENT_NAME="$BATCH_NORMALIZATION-bn-$INPUT_WIDTH-w-$INPUT_HEIGHT-h-$INPUT_COLORS-c-$LAYER_SIZE-layer-size-$LAYERS-layers-$MINI_BATCH_SIZE-mb-$LEARNING_RATE-lr-$MOMENTUM-m-$ANNEALING_RATE-anneal"

if [[ -n $RESUME_FROM ]]
then
    EXPERIMENT_NAME="$EXPERIMENT_NAME-resume"
fi

EXPERIMENT_DIRECTORY="$SCRIPT_DIRECTORY/$EXPERIMENT_NAME"
LOG_FILE="$EXPERIMENT_DIRECTORY/log"
MODEL_FILE="$EXPERIMENT_DIRECTORY/model.tar"
VALIDATION_ERROR_FILE="$EXPERIMENT_DIRECTORY/validation-error.csv"

mkdir -p $EXPERIMENT_DIRECTORY

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

COMMAND="benchmark-imagenet -l $LAYER_SIZE -e $EPOCHS -b $MINI_BATCH_SIZE -f $LAYER_OUTPUT_REDUCTION_FACTOR \
         --momentum $MOMENTUM --learning-rate $LEARNING_RATE -c $INPUT_COLORS -x $INPUT_WIDTH -y $INPUT_HEIGHT \
         -i $INPUT_TRAINING_DATASET -t $INPUT_VALIDATION_DATASET -L $LOGS --log-file $LOG_FILE \
         -o $MODEL_FILE -r $VALIDATION_ERROR_FILE --batch-normalization $BATCH_NORMALIZATION --annealing-rate $ANNEALING_RATE"

if [[ -n $RESUME_FROM ]]
then
    COMMAND="$COMMAND -m $RESUME_FROM"
fi

echo $COMMAND

$COMMAND

