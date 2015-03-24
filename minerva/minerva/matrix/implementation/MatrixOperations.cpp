
// Minerva Includes
#include <minerva/matrix/interface/MatrixOperations.h>

#include <minerva/parallel/interface/MultiBulkSynchronousParallel.h>

namespace minerva
{
namespace matrix
{

namespace detail
{

template<typename T>
template<typename std::tuple<T>, typename OperationType>
void applyOverPrecisions(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op, const Precision& precision)
{
	typedef T PrecisionType;
	typedef typename PrecisionType::type PrecisionPrimitive;

	assert(precision == PrecisionType());
	
	auto nativeOperation = static_cast<const OperationType&>(op);
	
	auto rawResult = static_cast<PrecisionPrimitive>(result.data());
	auto rawLeft   = static_cast<PrecisionPrimitive>(left.data());
	auto rawRight  = static_cast<PrecisionPrimitive>(right.data());
	
	auto flattenedResult = flatten(result);
	
	size_t elements = flattenedResult.elements();

	multiBulkSynchronousParallel([=](ThreadGroup threadGroup)
	{
		for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
		{
			rawResult[i] = nativeOperation(rawLeft[i], rawRight[i]);
		}
	});
}

template<typename PossiblePrecisions, template OperationType>
void applyOverPrecisions(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op, const Precision& precision)
{
	typedef std::tuple_element<0, decltype(PossiblePrecisions)>::type PossiblePrecisionType;

	if(precision == PossiblePrecisionType())
	{
		applyOverPrecisions<std::tuple<PossiblePrecisionType>, OperationType>(result, left, right, op, precision);
	}
	else
	{
		typedef RemoveFirstType<PossiblePrecisions>::type RemainingOperations;
		
		applyOverPrecisions<RemainingPrecisions, OperationType>(result, left, right, op, precision);
	}
}


template<typename T>
template<std::tuple<T>>
void applyOverOperations(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op, const Precision& precision)
{
	typedef T PossibleOperationType;

	assert(op == PossibleOperationType());
	
	applyOverPrecisions<AllPrecisions, PossibleOperationType>(result, left, right, op, precision);
}

template<typename PossibleOperations>
void applyOverOperations(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op, const Precision& precision)
{
	typedef std::tuple_element<0, decltype(PossibleOperations)>::type PossibleOperationType;

	if(op == PossibleOperationType())
	{
		applyOverPrecisions<AllPrecisions, PossibleOperationType>(result, left, right, op, precision);
	}
	else
	{
		typedef RemoveFirstType<PossibleOperations>::type RemainingOperations;
		
		applyOverOperations<RemainingOperations>(result, left, right, op, precision);
	}
}

void applyOverOperations(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op, const Precision& precision)
{
	applyOverOperations<AllBinaryOperations>(result, left, right, op, precision);
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
	auto precision = input.precision();
	
	detail::applyOverOperations(result, input, op, precision);
}

Matrix apply(Matrix& result, const Matrix& input, const Operation& op)
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

void copy(Matrix& result, const Matrix& input, const Precision&)
{

}

Matrix copy(const Matrix& input, const Precision& )
{
	Matrix result(input.size());
	
	if(hasCudaSupport())
	{
		
	}
}
 
Matrix slice(Matrix input, const Dimension& begin, const Dimension& end);
Matrix slice(Matrix input, const Dimension& begin, const Dimension& end, const Dimension& stride);
Matrix resize(Matrix input, const Dimension& size);

Matrix reshape(Matrix input, const Dimension& size)
{
	
}

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


