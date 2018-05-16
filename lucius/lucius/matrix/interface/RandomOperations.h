#pragma once

// Standard Library Includes
#include <cstddef>
#include <memory>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;         } }
namespace lucius { namespace matrix { class StaticOperator; } }
namespace lucius { namespace matrix { class Precision;      } }
namespace lucius { namespace matrix { class Dimension;      } }

namespace lucius
{
namespace matrix
{

namespace detail
{
class RandomStateImplementation;
}

class RandomState
{
public:
    RandomState();
    RandomState(const RandomState& );
    ~RandomState();

public:
    RandomState& operator=(const RandomState&);

public:
    detail::RandomStateImplementation& getImplementation();

private:
    std::unique_ptr<detail::RandomStateImplementation> _implementation;
};

void swapDefaultRandomState(RandomState& state);

void srand(size_t seed);

void rand(Matrix& result);
void randn(Matrix& result);

Matrix rand(const Dimension&, const Precision& );
Matrix randn(const Dimension&, const Precision& );

}
}

