#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix;    } }
namespace minerva { namespace matrix { class Dimension; } }

namespace minerva
{
namespace matrix
{

void gemm(Matrix& result, const Matrix& left, const Matrix& right);
Matrix gemm(const Matrix& left, const Matrix& right);

void gemm(Matrix& result, const Matrix& left, bool transposeLeft, const Matrix& right, bool transposeRight);
Matrix gemm(const Matrix& left, bool transposeLeft, const Matrix& right, bool transposeRight);

void gemm(Matrix& result, double beta,
	const Matrix& left, bool transposeLeft, double alpha,
	const Matrix& right, bool transposeRight);
Matrix gemm(const Matrix& left, bool transposeLeft, double alpha,
	const Matrix& right, bool transposeRight);

}
}

