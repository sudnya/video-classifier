# Get the current directory
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LUCIUS_INSTALL_PATH=$SCRIPTPATH/build_local

export PATH=$LUCIUS_INSTALL_PATH/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/X11/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/opt/local/lib:$LUCIUS_INSTALL_PATH/lib:/usr/local/lib
export DYLD_LIBRARY_PATH=$LUCIUS_INSTALL_PATH/lib:/usr/local/cuda/lib

export LUCIUS_KNOB_FILE=$SCRIPTPATH/scripts/lucius.config

export CXX=clang++
export CC=clang
export CPP=clang


