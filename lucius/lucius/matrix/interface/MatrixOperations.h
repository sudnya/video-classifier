
#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Operation; } }
namespace lucius { namespace matrix { class Precision; } }
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{
namespace matrix
{

void apply(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix apply(const Matrix& left, const Matrix& right, const Operation& op);

void apply(Matrix& result, const Matrix& input, const Operation& op);
Matrix apply(const Matrix& input, const Operation& op);

void reduce(Matrix& result, const Matrix& input, const Dimension& d, const Operation& op);
Matrix reduce(const Matrix& input, const Dimension& d, const Operation& op);

void broadcast(Matrix& result, const Matrix& left, const Matrix& right,
    const Dimension& d, const Operation& op);
Matrix broadcast(const Matrix& left, const Matrix& right, const Dimension& d,
    const Operation& op);

void zeros(Matrix& result);
Matrix zeros(const Dimension& size, const Precision& precision);

void ones(Matrix& result);
Matrix ones(const Dimension& size, const Precision& precision);

}
}

