# Get the current directory
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LUCIOUS_INSTALL_PATH=$SCRIPTPATH/build_local
export CUDA_PATH=/usr/local/cuda

export PATH=$LUCIOUS_INSTALL_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/lib:$CUDA_PATH/lib64:$LUCIOUS_INSTALL_PATH/lib

export LUCIOUS_KNOB_FILE=$SCRIPTPATH/scripts/minerva.config




