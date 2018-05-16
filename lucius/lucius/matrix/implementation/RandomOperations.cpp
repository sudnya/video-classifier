
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
        _cpuEngine  = state._cpuEngine;
        _cudaEngine = state._cudaEngine;

        _seed   = state._seed;
        _offset = state._offset;
    }

    RandomStateImplementation& operator=(const RandomStateImplementation& state)
    {
        if(this == &state)
        {
            return *this;
        }

        seed(state._seed);
        seek(state._offset);

        return *this;
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

    void seed(size_t seed)
    {
        if(CurandLibrary::loaded())
        {
            CurandLibrary::curandSetPseudoRandomGeneratorSeed(getCudaEngine(), seed);
        }
        else
        {
            getCpuEngine().seed(seed);
        }

        _offset = 0;
        _seed   = seed;
    }

    void seek(size_t offset)
    {
        if(CurandLibrary::loaded())
        {
            CurandLibrary::curandSetGeneratorOffset(getCudaEngine(), offset);
        }
        else
        {
            getCpuEngine().discard(offset);
        }

        _offset = offset;
    }

    void updateOffset(size_t offset)
    {
        _offset += offset;
    }

public:
    void swap(RandomStateImplementation& i)
    {
        std::swap(i._cpuEngine,  _cpuEngine);
        std::swap(i._cudaEngine, _cudaEngine);
        std::swap(i._seed,       _seed);
        std::swap(i._offset,     _offset);
    }

private:
    void _load()
    {
        CurandLibrary::load();

        if(!_cudaEngine && CurandLibrary::loaded())
        {
            CurandLibrary::curandCreateGenerator(&_cudaEngine,
                CurandLibrary::CURAND_RNG_PSEUDO_DEFAULT);

            _seed = 0;
            _offset = 0;
        }
    }

private:
    std::default_random_engine _cpuEngine;

private:
    CurandLibrary::curandGenerator_t _cudaEngine;
    size_t _seed;
    size_t _offset;

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

RandomState::RandomState(const RandomState& state)
: RandomState()
{
    *this = state;
}

RandomState& RandomState::operator=(const RandomState& state)
{
    *_implementation = *state._implementation;

    return *this;
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
    detail::getRandomGeneratorState().seed(seed);
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

        detail::getRandomGeneratorState().updateOffset(result.elements());
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

        detail::getRandomGeneratorState().updateOffset(result.elements());
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


