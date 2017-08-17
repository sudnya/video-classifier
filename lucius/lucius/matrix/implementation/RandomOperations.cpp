
// Lucius Includes
#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/CurandLibrary.h>
#include <lucius/matrix/interface/Matrix.h>

#include <lucius/parallel/interface/Synchronization.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <random>

namespace lucius
{
namespace matrix
{
namespace detail
{

class RandomStateImplementation
{
public:
    RandomStateImplementation()
    : _cudaEngine(nullptr)
    {
        _load();
    }

    ~RandomStateImplementation()
    {
        if(_cudaEngine != nullptr)
        {
            CurandLibrary::curandDestroyGenerator(_cudaEngine);
        }
    }

public:
    RandomStateImplementation(const RandomStateImplementation& state)
    {

    }


public:
    std::default_random_engine& getCpuEngine()
    {
        return _cpuEngine;
    }

    CurandLibrary::curandGenerator_t getCudaEngine()
    {
        return _cudaEngine;
    }

public:
    void swap(RandomStateImplementation& i)
    {
        std::swap(i._cpuEngine,  _cpuEngine);
        std::swap(i._cudaEngine, _cudaEngine);
    }

private:
    void _load()
    {
        CurandLibrary::load();

        if(!_cudaEngine && CurandLibrary::loaded())
        {
            CurandLibrary::curandCreateGenerator(&_cudaEngine,
                CurandLibrary::CURAND_RNG_PSEUDO_DEFAULT);
        }
    }

private:
    std::default_random_engine       _cpuEngine;
    CurandLibrary::curandGenerator_t _cudaEngine;

};

std::unique_ptr<RandomStateImplementation> randomGeneratorState;

RandomStateImplementation& getRandomGeneratorState()
{
    if(!randomGeneratorState)
    {
        randomGeneratorState = std::make_unique<RandomStateImplementation>();
    }

    return *randomGeneratorState;
}

}

RandomState::RandomState()
: _implementation(std::make_unique<detail::RandomStateImplementation>())
{

}

RandomState::~RandomState()
{

}

detail::RandomStateImplementation& RandomState::getImplementation()
{
    return *_implementation;
}

void swapDefaultRandomState(RandomState& state)
{
    detail::getRandomGeneratorState().swap(state.getImplementation());
}

void srand(size_t seed)
{
    if(CurandLibrary::loaded())
    {
        CurandLibrary::curandSetPseudoRandomGeneratorSeed(
            detail::getRandomGeneratorState().getCudaEngine(), seed);
    }
    else
    {
        detail::getRandomGeneratorState().getCpuEngine().seed(seed);
    }
}

void rand(Matrix& result)
{
    if(result.isContiguous() && CurandLibrary::loaded())
    {
        parallel::setNotSynchronized();

        if(result.precision() == SinglePrecision())
        {
            CurandLibrary::curandGenerateUniform(detail::getRandomGeneratorState().getCudaEngine(),
                static_cast<float*>(result.data()), result.elements());
        }
        else if(result.precision() == DoublePrecision())
        {
            CurandLibrary::curandGenerateUniformDouble(
                detail::getRandomGeneratorState().getCudaEngine(),
                static_cast<double*>(result.data()), result.elements());
        }
        else
        {
            assertM(false, "Rand not implemented for this precision.");
        }
    }
    else
    {
        if(result.precision() == SinglePrecision())
        {
            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

            for(auto i = result.begin(); i != result.end(); ++i)
            {
                *i = distribution(detail::getRandomGeneratorState().getCpuEngine());
            }
        }
        else if(result.precision() == DoublePrecision())
        {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);

            for(auto i = result.begin(); i != result.end(); ++i)
            {
                *i = distribution(detail::getRandomGeneratorState().getCpuEngine());
            }
        }
        else
        {
            assertM(false, "Rand not implemented for this precision.");
        }

    }
}

void randn(Matrix& result)
{
    if(result.isContiguous() && CurandLibrary::loaded())
    {
        parallel::setNotSynchronized();

        if(result.precision() == SinglePrecision())
        {
            CurandLibrary::curandGenerateNormal(detail::getRandomGeneratorState().getCudaEngine(),
                static_cast<float*>(result.data()), result.elements(), 0.0f, 1.0f);
        }
        else if(result.precision() == DoublePrecision())
        {
            CurandLibrary::curandGenerateNormalDouble(
                detail::getRandomGeneratorState().getCudaEngine(),
                static_cast<double*>(result.data()), result.elements(), 0.0, 1.0);
        }
        else
        {
            assertM(false, "Rand not implemented for this precision.");
        }
    }
    else
    {
        if(result.precision() == SinglePrecision())
        {
            std::normal_distribution<float> distribution;

            for(auto i = result.begin(); i != result.end(); ++i)
            {
                *i = distribution(detail::getRandomGeneratorState().getCpuEngine());
            }
        }
        else if(result.precision() == DoublePrecision())
        {
            std::normal_distribution<double> distribution;

            for(auto i = result.begin(); i != result.end(); ++i)
            {
                *i = distribution(detail::getRandomGeneratorState().getCpuEngine());
            }
        }
        else
        {
            assertM(false, "Rand not implemented for this precision.");
        }

    }

}

Matrix rand(const Dimension& size, const Precision& precision)
{
    Matrix result(size, precision);

    rand(result);

    return result;
}

Matrix randn(const Dimension& size, const Precision& precision)
{
    Matrix result(size, precision);

    randn(result);

    return result;
}

}
}


