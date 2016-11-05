
// Lucius Includes
#include <lucius/matrix/interface/SortOperations.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Dimension.h>
#include <lucius/matrix/interface/DimensionTransformations.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace matrix
{

void sort(Matrix& values, const Dimension& dimensionsToSort)
{
    assertM(false, "Not implemented.");
}

void sort(Matrix& values)
{
    sort(values, range(values.size()));
}

void sortByKey(Matrix& keys, Matrix& values, const Dimension& dimensionsToSort)
{
    assertM(false, "Not implemented.");
}

void sortByKey(Matrix& keys, Matrix& values)
{
    sortByKey(keys, values, range(values.size()));
}

}
}


