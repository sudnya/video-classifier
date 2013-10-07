# Get the current directory
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MINERVA_INSTALL_PATH=$SCRIPTPATH/build_local
export CUDA_PATH=/usr/local/new-cuda

export PATH=$MINERVA_INSTALL_PATH/bin:/usr/lib/lightdm/lightdm:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games
export LD_LIBRARY_PATH=/usr/local/lib:$CUDA_PATH/lib64:$MINERVA_INSTALL_PATH/lib




