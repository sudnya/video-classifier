#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }
namespace lucius { namespace matrix { class Operation; } }

namespace lucius
{
namespace matrix
{

void sort(Matrix& values, const Dimension& dimensionsToSort, const Operation& comparisonOperation);
void sort(Matrix& values, const Operation& comparisonOperation);

void sortByKey(Matrix& keys, Matrix& values, const Operation& comparisonOperation);
void sortByKey(Matrix& keys, Matrix& values, const Dimension& dimensionsToSort,
    const Operation& comparisonOperation);

}
}


