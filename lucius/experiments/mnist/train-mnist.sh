#!/bin/bash

SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CUDA_DEVICES="0"
INPUT_TRAINING_DATASET="/Users/gregdiamos/temp/mnist/train/database.txt"
INPUT_VALIDATION_DATASET="/Users/gregdiamos/temp/mnist/test/database.txt"
LOGS="NesterovAcceleratedGradientSolver,BenchmarkImageNet,LabelMatchResultProcessor,InputVisualDataProducer"
LAYER_SIZE="256"
INPUT_WIDTH="28"
INPUT_HEIGHT="28"
INPUT_COLORS="1"
LAYERS="3"
MINI_BATCH_SIZE="64"
LAYER_OUTPUT_REDUCTION_FACTOR="1"
BATCH_NORMALIZATION=0
FORWARD_BATCH_NORMALIZATION=1
EPOCHS="80"
LEARNING_RATE="1.0e-2"
ANNEALING_RATE="1.000001"
MOMENTUM="0.9"
RESUME_FROM=""


EXPERIMENT_NAME="$FORWARD_BATCH_NORMALIZATION-fbn-$BATCH_NORMALIZATION-bn-$INPUT_WIDTH-w-$INPUT_HEIGHT-h-$INPUT_COLORS-c-$LAYER_SIZE-layer-size-$LAYERS-layers-$MINI_BATCH_SIZE-mb-$LEARNING_RATE-lr-$MOMENTUM-m-$ANNEALING_RATE-anneal"

if [[ -n $RESUME_FROM ]]
then
    EXPERIMENT_NAME="$EXPERIMENT_NAME-resume"
fi

EXPERIMENT_DIRECTORY="$SCRIPT_DIRECTORY/$EXPERIMENT_NAME"
LOG_FILE="$EXPERIMENT_DIRECTORY/log"
MODEL_FILE="$EXPERIMENT_DIRECTORY/model.tar"
VALIDATION_ERROR_FILE="$EXPERIMENT_DIRECTORY/validation-error.csv"
TRAINING_ERROR_FILE="$EXPERIMENT_DIRECTORY/training-error.csv"

mkdir -p $EXPERIMENT_DIRECTORY

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

COMMAND="benchmark-imagenet -l $LAYER_SIZE -e $EPOCHS -b $MINI_BATCH_SIZE -f $LAYER_OUTPUT_REDUCTION_FACTOR \
         --momentum $MOMENTUM --learning-rate $LEARNING_RATE -c $INPUT_COLORS -x $INPUT_WIDTH -y $INPUT_HEIGHT \
         -i $INPUT_TRAINING_DATASET -t $INPUT_VALIDATION_DATASET -L $LOGS --log-file $LOG_FILE \
         -o $MODEL_FILE -r $VALIDATION_ERROR_FILE --training-report-path $TRAINING_ERROR_FILE \
         --layers $LAYERS \
         --annealing-rate $ANNEALING_RATE"

if [ $BATCH_NORMALIZATION -eq 1 ]
then
    COMMAND="$COMMAND --batch-normalization"
fi

if [ $FORWARD_BATCH_NORMALIZATION -eq 1 ]
then
    COMMAND="$COMMAND --forward-batch-normalization"
fi

if [[ -n $RESUME_FROM ]]
then
    COMMAND="$COMMAND -m $RESUME_FROM"
fi

echo $COMMAND

$COMMAND

