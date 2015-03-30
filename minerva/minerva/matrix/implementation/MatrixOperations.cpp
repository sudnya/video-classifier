
// Minerva Includes
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/MatrixTransformations.h>
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/parallel/interface/MultiBulkSynchronousParallel.h>

#include <minerva/util/interface/Metaprogramming.h>

// Standard Library Includes
#include <tuple>

namespace minerva
{
namespace matrix
{

namespace detail
{

template<typename OperationType, typename T>
void applyOverPrecisions(Matrix& result, const Matrix& left, const Matrix& right,
	const Operation& op, const Precision& precision, std::tuple<T> precisions)
{
	typedef T PrecisionPrimitive;
	typedef typename PrecisionPrimitive::type NativeType;

	assert(precision == PrecisionPrimitive());
	
	auto nativeOperation = static_cast<const OperationType&>(op);
	
	assert(result.isContiguous() && left.isContiguous() && right.isContiguous()); // TODO: handle complex strides
	
	auto rawResult = static_cast<NativeType*>(result.data());
	auto rawLeft   = static_cast<const NativeType*>(left.data());
	auto rawRight  = static_cast<const NativeType*>(right.data());
	
	auto flattenedResult = flatten(result);
	
	size_t elements = flattenedResult.elements();

	parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
	{
		for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
		{
			rawResult[i] = nativeOperation(rawLeft[i], rawRight[i]);
		}
	});
}

template<typename OperationType, typename PossiblePrecisions>
void applyOverPrecisions(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op,
	const Precision& precision, PossiblePrecisions precisions)
{
	typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

	if(precision == PossiblePrecisionType())
	{
		applyOverPrecisions<OperationType>(result, left, right, op, precision,
			std::tuple<PossiblePrecisionType>());
	}
	else
	{
		typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;
		
		applyOverPrecisions<OperationType>(result, left, right, op, precision, RemainingPrecisions());
	}
}


template<typename T>
void applyOverOperations(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op,
	const Precision& precision, const std::tuple<T>& operations)
{
	typedef T PossibleOperationType;

	assert(op == PossibleOperationType());
	
	applyOverPrecisions<PossibleOperationType, AllPrecisions>(result, left, right, op, precision, AllPrecisions());
}

template<typename PossibleOperations>
void applyOverOperations(Matrix& result, const Matrix& left, const Matrix& right,
	const Operation& op, const Precision& precision, const PossibleOperations& operations)
{
	typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

	if(op == PossibleOperationType())
	{
		applyOverOperations(result, left, right, op, precision, std::tuple<PossibleOperationType>());
	}
	else
	{
		typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;
		
		applyOverOperations(result, left, right, op, precision, RemainingOperations());
	}
}

void applyOverOperations(Matrix& result, const Matrix& left, const Matrix& right,
	const Operation& op, const Precision& precision)
{
	applyOverOperations<AllBinaryOperations>(result, left, right, op, precision, AllBinaryOperations());
}

}

void apply(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op)
{
	auto precision = left.precision();
	
	detail::applyOverOperations(result, left, right, op, precision);
}

Matrix apply(const Matrix& left, const Matrix& right, const Operation& op)
{
	assert(left.size() == right.size());
	assert(left.precision() == right.precision());
	
	Matrix temp(left.size());
	
	apply(temp, left, right, op);
	
	return temp;
}

void apply(Matrix& result, const Matrix& input, const Operation& op)
{
	//auto precision = input.precision();
	
	//detail::applyOverOperations(result, input, op, precision);
}

Matrix apply(const Matrix& input, const Operation& op)
{
	Matrix result(input.size());
	
	apply(result, input, op);
	
	return result;
}

void reduce(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix reduce(const Matrix& left, const Matrix& right, const Operation& op, const Dimension& d);

void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix broadcast(const Matrix& left, const Matrix& right, const Operation& op);

void copy(Matrix& result, const Matrix& input);
Matrix copy(const Matrix& input);

void copy(Matrix& result, const Matrix& input, const Precision&);

Matrix copy(const Matrix& input, const Precision& precision)
{
	Matrix result(input.size(), precision);
	
	copy(result, input, precision);
	
	return result;
}
 
Matrix slice(Matrix input, const Dimension& begin, const Dimension& end);
Matrix slice(Matrix input, const Dimension& begin, const Dimension& end, const Dimension& stride);
Matrix resize(Matrix input, const Dimension& size);

Matrix reshape(Matrix input, const Dimension& size);

void gemm(Matrix& result, const Matrix& left, const Matrix& right)
{
	gemm(result, 0.0, left, false, 1.0, right, false);
}

Matrix gemm(const Matrix& left, const Matrix& right)
{
	Matrix result({left.size()[0], right.size()[1]});

	gemm(result, left, right);
	
	return result;
}

void gemm(Matrix& result, double beta,
	const Matrix& left, bool transposeLeft, double alpha,
	const Matrix& right, bool transposeRight)
{

}

Matrix gemm(double beta,
	const Matrix& left, bool transposeLeft, double alpha,
	const Matrix& right, bool transposeRight)
{
	size_t rows    = transposeLeft  ? left.size()[1]  : left.size()[0];
	size_t columns = transposeRight ? right.size()[0] : right.size()[1];
	
	Matrix result({rows, columns}, left.precision());
	
	gemm(result, beta, left, transposeLeft, alpha, right, transposeRight);
}

}
}


