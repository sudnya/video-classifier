#pragma once

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }
namespace lucius { namespace matrix { class Precision; } }

namespace lucius
{
namespace matrix
{

typedef std::vector<std::vector<size_t>> LabelVector;
typedef std::vector<size_t> IndexVector;

void computeCtc(Matrix& costs, Matrix& gradients, const Matrix& inputActivations,
    const LabelVector& labels, const IndexVector& timestepsPerSample);

Dimension getBeamSearchOutputSize(const Dimension&, size_t beamSize);

void ctcBeamSearch(Matrix& outputActivationWeights, Matrix& inputPaths, Matrix& outputActivations,
    const Matrix& inputActivations, size_t beamSize);

void ctcBeamSearchInputGradients(Matrix& inputDeltas, const Matrix& outputActivationWeights,
    const Matrix& inputPaths, const Matrix& outputDeltas, size_t beamSize);

}
}


