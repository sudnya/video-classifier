#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{
namespace matrix
{

void sort(Matrix& values, const Dimension& dimensionsToSort = Dimension());
void sortByKey(Matrix& keys, Matrix& values, const Dimension& dimensionsToSort = Dimension());

Matrix reduceByKey(const Matrix& keys, const Matrix& values, const Operation& op);
void reduceByKey(Matrix& result, const Matrix& keys, const Matrix& values, const Operation& op);

Matrix applyToAdjacentElements(const Matrix& input, const Dimension& dimensionsToApplyTo,
    const Operation& op, double initialValue = 0.0);
void applyToAdjacentElements(Matrix& output, const Matrix& input,
    const Dimension& dimensionsToApplyTo, const Operation& op, double initialValue = 0.0);

void inclusiveScan(Matrix& output, const Matrix& input, const Dimension& dimensionsToReduce,
    const Operation& op);
Matrix inclusiveScan(const Matrix& input, const Dimension& dimensionsToReduce,
    const Operation& op);

void exclusiveScan(Matrix& output, const Matrix& input, const Dimension& dimensionsToReduce,
    const Operation& op);
Matrix exclusiveScan(const Matrix& input, const Dimension& dimensionsToReduce,
    const Operation& op);


}
}

