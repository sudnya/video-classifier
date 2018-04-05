
// Lucius Includes
#include <lucius/matrix/interface/MatrixVectorOperations.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/GenericOperators.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/DimensionTransformations.h>

#include <lucius/parallel/interface/ParallelFor.h>
#include <lucius/parallel/interface/ConcurrentCollectives.h>

namespace lucius
{
namespace matrix
{

void apply(MatrixVector& result, const MatrixVector& left, const MatrixVector& right,
    const StaticOperator& op)
{
    assert(result.size() == left.size());
    assert(result.size() == right.size());

    size_t elements = result.size();

    parallel::parallelFor([&](parallel::ThreadGroup g)
    {
        for(size_t i = g.id(); i < elements; i += g.size())
        {
            apply(result[i], left[i], right[i], op);
        }
    });
}

MatrixVector apply(const MatrixVector& left, const MatrixVector& right, const StaticOperator& op)
{
    MatrixVector result;

    for(auto& matrix : left)
    {
        result.push_back(Matrix(matrix.size(), matrix.precision()));
    }

    apply(result, left, right, op);

    return result;
}

void apply(MatrixVector& result, const MatrixVector& input, const StaticOperator& op)
{
    assert(result.size() == input.size());

    size_t elements = result.size();

    parallel::parallelFor([&](parallel::ThreadGroup g)
    {
        for(size_t i = g.id(); i < elements; i += g.size())
        {
            apply(result[i], input[i], op);
        }
    });
}

MatrixVector apply(const MatrixVector& input, const StaticOperator& op)
{
    MatrixVector result;

    for(auto& i : input)
    {
        result.push_back(Matrix(i.size(), i.precision()));
    }

    apply(result, input, op);

    return result;
}

void reduce(MatrixVector& result, const MatrixVector& input, const Dimension& d,
    const StaticOperator& op)
{
    assert(result.size() == input.size());

    size_t elements = result.size();

    parallel::parallelFor([&](parallel::ThreadGroup g)
    {
        for(size_t i = g.id(); i < elements; i += g.size())
        {
            reduce(result[i], input[i], d, op);
        }
    });

    if(d.empty())
    {
        for(size_t i = 1; i < result.size(); ++i)
        {
            apply(result[0], result[0], result[i], op);
        }

        result.resize(1);
    }
}

MatrixVector reduce(const MatrixVector& input, const Dimension& d, const StaticOperator& op)
{
    MatrixVector result;

    for(auto& i : input)
    {
        result.push_back(Matrix(removeDimensions(i.size(), d), i.precision()));
    }

    reduce(result, input, d, op);

    return result;
}

double dotProduct(const MatrixVector& left, const MatrixVector& right)
{
    auto results = reduce(apply(left, right, Multiply()), {}, Add());

    double result = 0.0;

    for(auto element : results)
    {
        result += element[0];
    }

    return result;
}

void zeros(MatrixVector& result)
{
    apply(result, result, Fill(0.0));
}

void copy(MatrixVector& result, const MatrixVector& input)
{
    for(auto& matrix : input)
    {
        result.push_back(copy(matrix));
    }
}

MatrixVector copy(const MatrixVector& input)
{
    MatrixVector result;

    copy(result, input);

    return result;
}

}
}



