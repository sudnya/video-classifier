# Get the current directory
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LUCIUS_INSTALL_PATH=$SCRIPTPATH/build_local
export CUDA_PATH=/usr/local/cuda

export PATH=$LUCIUS_INSTALL_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/ffmpeg_build/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib:$CUDA_PATH/lib64:$LUCIUS_INSTALL_PATH/lib

export LUCIUS_KNOB_FILE=$SCRIPTPATH/scripts/lucius.config
export CUDA_ARCH=sm_61




