# Get the current directory
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MINERVA_INSTALL_PATH=$SCRIPTPATH/build_local

export PATH=$MINERVA_INSTALL_PATH/bin:/opt/local/bin:/opt/local/sbin:/usr/local/cuda/bin:/opt/local/bin:/opt/local/sbin:/usr/local/rvm/gems/ruby-1.9.3-p392/bin:/usr/local/rvm/gems/ruby-1.9.3-p392@global/bin:/usr/local/rvm/rubies/ruby-1.9.3-p392/bin:/usr/local/rvm/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/X11/bin:/usr/texbin:/opt/sm/bin:/opt/sm/pkg/active/bin:/opt/sm/pkg/active/sbin
export LD_LIBRARY_PATH=/usr/local/cuda/lib:$MINERVA_INSTALL_PATH/lib:/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.8.sdk/usr/lib:/opt/local/lib
export CXX=clang++
export CC=clang
export CPP=clang


