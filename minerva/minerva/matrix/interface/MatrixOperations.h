
#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix;    } }
namespace minerva { namespace matrix { class Operation; } }
namespace minerva { namespace matrix { class Precision; } }
namespace minerva { namespace matrix { class Dimension; } }

namespace minerva
{
namespace matrix
{

void apply(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix apply(const Matrix& left, const Matrix& right, const Operation& op);

void apply(Matrix& result, const Matrix& input, const Operation& op);
Matrix apply(const Matrix& input, const Operation& op);

void reduce(Matrix& result, const Matrix& input, const Dimension& d, const Operation& op);
Matrix reduce(const Matrix& input, const Dimension& d, const Operation& op);

void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix broadcast(const Matrix& left, const Matrix& right, const Operation& op);

void zeros(Matrix& result);
Matrix zeros(const Dimension& size, const Precision& precision);
 
Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end);
Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end, const Dimension& stride);
Matrix resize(const Matrix& input, const Dimension& size);
Matrix reshape(const Matrix& input, const Dimension& size);

}
}

