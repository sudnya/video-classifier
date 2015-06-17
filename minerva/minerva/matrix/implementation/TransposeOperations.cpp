
// Minerva Includes
#include <minerva/matrix/interface/TransposeOperations.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/Dimension.h>

namespace minerva
{
namespace matrix
{

Matrix transpose(const Matrix& input)
{
    auto newSize = input.size();

    if(newSize.size() > 1)
    {
        std::swap(newSize[0], newSize[1]);
    }

    Matrix result(newSize, input.precision());

    transpose(result, input);

    return result;
}

void transpose(Matrix& output, const Matrix& input)
{
    assert(false && "Not implemented.");
}

}
}



