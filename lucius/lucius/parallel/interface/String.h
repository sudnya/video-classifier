
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>

namespace lucius
{
namespace parallel
{

class string
{
public:
    CUDA_DECORATOR string(const char* s) {}
    CUDA_DECORATOR ~string() {}

};

}
}




