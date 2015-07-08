/*  \file   test-memory.h
    \date   June 27, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the memory unit tests.
*/

// Lucious Includes
#include <lucious/parallel/interface/Memory.h>

#include <lucious/util/interface/debug.h>

// Standard Library Includes
#include <iostream>
#include <random>
#include <cstring>

/*
    A simple test for allocating and freeing memory.
*/
bool testMallocFree()
{
    auto sizes = {(4), (8), 1 << 10, 1 << 20, 1 << 24};

    for(auto size : sizes)
    {
        auto* address = lucious::parallel::malloc(size);

        if(address == nullptr)
        {
            std::cout << " Test Malloc Free Failed:\n";
            std::cout << "  failed to allocate memory of size " << size << "\n";
            return false;
        }

        lucious::parallel::free(address);
    }

    std::cout << " Test Malloc Free Passed\n";

    return true;
}

/*
    Test copy to and from allocated memory.
*/
bool testAllocatedMemoryCopy()
{
    size_t sizes[] = {(4), (8), 1 << 10, 1 << 20, 1 << 24};

    std::default_random_engine engine;

    engine.seed(377);

    for(auto size : sizes)
    {
        auto* address = lucious::parallel::malloc(size);

        std::vector<uint8_t> inputData(size);

        for(auto& element : inputData)
        {
            element = engine();
        }

        std::memcpy(address, inputData.data(), size);

        std::vector<uint8_t> outputData(size);

        std::memcpy(outputData.data(), address, size);

        for(size_t index = 0; index < size; ++index)
        {
            if(outputData[index] != inputData[index])
            {
                std::cout << " Test Allocated Memory Copy Failed:\n";
                std::cout << "  at index " << index << " out of buffer sized "
                    << size << ", input value " << outputData[index]
                    << " does not match output value " << inputData[index] << "\n";
                return false;
            }
        }

        lucious::parallel::free(address);
    }

    std::cout << " Test Allocated Memory Copy Passed\n";

    return true;
}

int main(int argc, char** argv)
{
    //lucious::util::enableAllLogs();

    std::cout << "Running memory unit tests\n";

    bool passed = true;

    passed &= testMallocFree();
    passed &= testAllocatedMemoryCopy();
    //passed &= testModifiedMemoryCopy();

    if(not passed)
    {
        std::cout << "Test Failed\n";
    }
    else
    {
        std::cout << "Test Passed\n";
    }

    return 0;
}



