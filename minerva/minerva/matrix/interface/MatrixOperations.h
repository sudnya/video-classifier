
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

void reduce(Matrix& result, const Matrix& left, const Matrix& right, const Dimension& d, const Operation& op);
Matrix reduce(const Matrix& left, const Matrix& right, const Dimension& d, const Operation& op);

void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix broadcast(const Matrix& left, const Matrix& right, const Operation& op);

void zeros(Matrix& result);
Matrix zeros(const Dimension& size, const Precision& precision);

void copy(Matrix& result, const Matrix& input);
Matrix copy(const Matrix& input);

void copy(Matrix& result, const Matrix& input, const Precision&);
Matrix copy(const Matrix& input, const Precision&);
 
Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end);
Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end, const Dimension& stride);
Matrix resize(const Matrix& input, const Dimension& size);
Matrix reshape(const Matrix& input, const Dimension& size);

void gemm(Matrix& result, const Matrix& left, const Matrix& right);
Matrix gemm(const Matrix& left, const Matrix& right);

void gemm(Matrix& result, const Matrix& left, bool transposeLeft, const Matrix& right, bool transposeRight);
Matrix gemm(const Matrix& left, bool transposeLeft, const Matrix& right, bool transposeRight);

void gemm(Matrix& result, double beta,
	const Matrix& left, bool transposeLeft, double alpha,
	const Matrix& right, bool transposeRight);
Matrix gemm(double beta,
	const Matrix& left, bool transposeLeft, double alpha,
	const Matrix& right, bool transposeRight);

}
}

