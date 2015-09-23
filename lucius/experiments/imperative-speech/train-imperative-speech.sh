#!/bin/bash

SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CUDA_DEVICES="0"

INPUT_TRAINING_DATASET="/mnt/Data/summer2015/imperative-speech/training/database.txt"
INPUT_VALIDATION_DATASET="/mnt/Data/summer2015/imperative-speech/validation/database.txt"
LOGS="NesterovAcceleratedGradientSolver,BenchmarkImperativeSpeech,InputAudioDataProducer,LabelMatchResultProcessor"

LAYER_SIZE="128"
SAMPLING_RATE=8000
FRAME_DURATION=160
RECURRENT_LAYERS="2"
FORWARD_LAYERS="3"
MINI_BATCH_SIZE="64"
EPOCHS="20"
LEARNING_RATE="0.0001"
MOMENTUM="0.9"
RESUME_FROM=""

EXPERIMENT_NAME="benchmark-imperative-speech-$SAMPLING_RATE-frequency-$FRAME_DURATION-frame-$RECURRENT_LAYERS-recurrent-$FORWARD_LAYERS-forward-$LAYER_SIZE-size-$MINI_BATCH_SIZE-mini-batch-$LEARNING_RATE-learning-rate-$MOMENTUM-momentum"

EXPERIMENT_DIRECTORY="$SCRIPT_DIRECTORY/$EXPERIMENT_NAME"
LOG_FILE="$EXPERIMENT_DIRECTORY/log"
MODEL_FILE="$EXPERIMENT_DIRECTORY/model.tar"
VALIDATION_ERROR_FILE="$EXPERIMENT_DIRECTORY/validation-error.csv"

mkdir -p $EXPERIMENT_DIRECTORY

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

COMMAND="benchmark-imperative-speech -l $LAYER_SIZE -e $EPOCHS -b $MINI_BATCH_SIZE -f $FORWARD_LAYERS \
         --momentum $MOMENTUM --learning-rate $LEARNING_RATE -r $RECURRENT_LAYERS --sampling-rate $SAMPLING_RATE \
         -i $INPUT_TRAINING_DATASET -t $INPUT_VALIDATION_DATASET -L $LOGS --log-file $LOG_FILE \
         -o $MODEL_FILE -r $VALIDATION_ERROR_FILE --frame-duration $FRAME_DURATION"

if [[ -n $RESUME_FROM ]]
then
    COMMAND="$COMMAND -m $RESUME_FROM"
fi

echo $COMMAND

$COMMAND

