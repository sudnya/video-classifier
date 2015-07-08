# Get the current directory
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MINERVA_INSTALL_PATH=$SCRIPTPATH/build_local

export PATH=$MINERVA_INSTALL_PATH/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/X11/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/opt/local/lib:$MINERVA_INSTALL_PATH/lib:/usr/local/lib

export MINERVA_KNOB_FILE=$SCRIPTPATH/scripts/minerva.config

export CXX=clang++
export CC=clang
export CPP=clang


