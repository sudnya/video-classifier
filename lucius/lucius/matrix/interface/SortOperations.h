#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;         } }
namespace lucius { namespace matrix { class Dimension;      } }
namespace lucius { namespace matrix { class StaticOperator; } }

namespace lucius
{
namespace matrix
{

void sort(Matrix& values, const Dimension& dimensionsToSort, const StaticOperator& comparisonStaticOperator);
void sort(Matrix& values, const StaticOperator& comparisonStaticOperator);

void sortByKey(Matrix& keys, Matrix& values, const StaticOperator& comparisonStaticOperator);
void sortByKey(Matrix& keys, Matrix& values, const Dimension& dimensionsToSort,
    const StaticOperator& comparisonStaticOperator);

}
}


