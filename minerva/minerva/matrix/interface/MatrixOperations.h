
#pragma once

namespace minerva
{
namespace matrix
{

void apply(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix apply(const Matrix& left, const Matrix& right, const Operation& op);

void apply(Matrix& result, const Matrix& input, const Operation& op);
Matrix apply(Matrix& result, const Matrix& input, const Operation& op);

void reduce(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix reduce(const Matrix& left, const Matrix& right, const Operation& op, const Dimension& d);

void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix broadcast(const Matrix& left, const Matrix& right, const Operation& op);

void copy(Matrix& result, const Matrix& input);
Matrix copy(const Matrix& input);

void copy(Matrix& result, const Matrix& input, const Precision&);
Matrix copy(const Matrix& input, const Precision&);
 
Matrix slice(const Dimension& begin, const Dimension& end);
Matrix slice(const Dimension& begin, const Dimension& end, const Dimension& stride);
Matrix resize(const Dimension& size);
Matrix reshape(const Dimension& size);

void gemm(Matrix& result, const Matrix& left, const Matrix& right);
Matrix gemm(const Matrix& left, const Matrix& right);

void gemm(Matrix& result, double beta,
	const Matrix& left, bool transposeLeft, double alpha,
	const Matrix& right, bool transposeRight);
Matrix gemm(double beta,
	const Matrix& left, bool transposeLeft, double alpha,
	const Matrix& right, bool transposeRight);

}
}

