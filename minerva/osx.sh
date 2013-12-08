# Get the current directory
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MINERVA_INSTALL_PATH=$SCRIPTPATH/build_local

export PATH=$MINERVA_INSTALL_PATH/bin:/usr/bin:/bin:/opt/local/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/opt/local/lib:$MINERVA_INSTALL_PATH/lib
export CXX=clang++
export CC=clang
export CPP=clang


