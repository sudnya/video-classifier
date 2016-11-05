
// Lucius Includes
#include <lucius/matrix/interface/ReduceByKeyOperations.h>
#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace matrix
{

Matrix reduceByKey(const Matrix& keys, const Matrix& values, const Operation& op)
{
    Matrix result(values.size(), values.precision());

    reduceByKey(result, keys, values, op);

    return result;
}

void reduceByKey(Matrix& result, const Matrix& keys, const Matrix& values, const Operation& op)
{
    assertM(false, "Not implemented.");
}

}
}

