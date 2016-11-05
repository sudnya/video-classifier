#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{
namespace matrix
{

void sort(Matrix& values, const Dimension& dimensionsToSort);
void sort(Matrix& values);

void sortByKey(Matrix& keys, Matrix& values);
void sortByKey(Matrix& keys, Matrix& values, const Dimension& dimensionsToSort);

}
}


