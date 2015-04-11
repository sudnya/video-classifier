
// Minerva Includes
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/MatrixTransformations.h>
#include <minerva/matrix/interface/CopyOperations.h>
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixView.h>
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
	
	size_t elements = result.elements();

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

namespace detail
{

template<typename OperationType, typename T>
void applyOverPrecisions(Matrix& result, const Matrix& input,
	const Operation& op, const Precision& precision, std::tuple<T> precisions)
{
	typedef T PrecisionPrimitive;
	typedef typename PrecisionPrimitive::type NativeType;

	assert(precision == PrecisionPrimitive());
	
	auto nativeOperation = static_cast<const OperationType&>(op);
	
	assert(result.isContiguous() && input.isContiguous()); // TODO: handle complex strides
	
	auto rawResult = static_cast<NativeType*>(result.data());
	auto rawInput  = static_cast<const NativeType*>(input.data());
	
	size_t elements = result.elements();

	parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
	{
		for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
		{
			rawResult[i] = nativeOperation(rawInput[i]);
		}
	});
}

template<typename OperationType, typename PossiblePrecisions>
void applyOverPrecisions(Matrix& result, const Matrix& input, const Operation& op,
	const Precision& precision, PossiblePrecisions precisions)
{
	typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

	if(precision == PossiblePrecisionType())
	{
		applyOverPrecisions<OperationType>(result, input, op, precision,
			std::tuple<PossiblePrecisionType>());
	}
	else
	{
		typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;
		
		applyOverPrecisions<OperationType>(result, input, op, precision, RemainingPrecisions());
	}
}


template<typename T>
void applyOverOperations(Matrix& result, const Matrix& input, const Operation& op,
	const Precision& precision, const std::tuple<T>& operations)
{
	typedef T PossibleOperationType;

	assert(op == PossibleOperationType());
	
	applyOverPrecisions<PossibleOperationType, AllPrecisions>(result, input, op, precision, AllPrecisions());
}

template<typename PossibleOperations>
void applyOverOperations(Matrix& result, const Matrix& input,
	const Operation& op, const Precision& precision, const PossibleOperations& operations)
{
	typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

	if(op == PossibleOperationType())
	{
		applyOverOperations(result, input, op, precision, std::tuple<PossibleOperationType>());
	}
	else
	{
		typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;
		
		applyOverOperations(result, input, op, precision, RemainingOperations());
	}
}

void applyOverOperations(Matrix& result, const Matrix& input,
	const Operation& op, const Precision& precision)
{
	applyOverOperations<AllUnaryOperations>(result, input, op, precision, AllUnaryOperations());
}

}

void apply(Matrix& result, const Matrix& input, const Operation& op)
{
	detail::applyOverOperations(result, input, op, input.precision());
}

Matrix apply(const Matrix& input, const Operation& op)
{
	Matrix result(input.size());
	
	apply(result, input, op);
	
	return result;
}

void reduce(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix reduce(const Matrix& left, const Matrix& right, const Operation& op, const Dimension& d);

namespace detail
{

template <typename ActualOperation, typename ActualPrecision>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op, const std::tuple<ActualPrecision>& p)
{
	typedef typename ActualPrecision::type NativeType;
	
	assert(ActualPrecision()  == result.precision());
	assert(result.precision() == left.precision());
	assert(result.precision() == right.precision());
	assert(result.size()      == left.size());
	
	size_t elements = result.elements();

	auto nativeOperation = static_cast<const ActualOperation&>(op);
	
	MatrixView<NativeType>      resultView(result);
	ConstMatrixView<NativeType> leftView(left);
	ConstMatrixView<NativeType> rightView(right);
	
	parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
	{
		for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
		{
			auto fullDimension = linearToDimension(i, resultView.size());
			auto reducedDimension = fullDimension;

			reducedDimension.pop_back(fullDimension.size() - rightView.size().size());
			
			resultView(fullDimension) = nativeOperation(leftView(fullDimension), rightView(reducedDimension));
		}
	});
}

template <typename ActualOperation, typename PossiblePrecisions>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op, const PossiblePrecisions& possiblePrecisions)
{
	typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;
	if(result.precision() == PossiblePrecisionType())
	{
		broadcast<ActualOperation>(result, left, right, op, std::tuple<PossiblePrecisionType>());
	}
	else
	{
		typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;
		broadcast<ActualOperation>(result, left, right, op, RemainingPrecisions());
	}
}

template <typename PossibleOperation>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op, const std::tuple<PossibleOperation>& p)
{
	assert(PossibleOperation() == op);
	broadcast<PossibleOperation>(result, left, right, op, AllPrecisions());
}

template <typename PossibleOperations>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op, const PossibleOperations& possibleOperations)
{
	typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;
	if(op == PossibleOperationType())
	{
		broadcast(result, left, right, op, std::tuple<PossibleOperationType>());
	}
	else
	{
		typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

		broadcast(result, left, right, op, RemainingOperations());
	}
	
}

}

void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op)
{
	detail::broadcast(result, left, right, op, AllBinaryOperations());
}

Matrix broadcast(const Matrix& left, const Matrix& right, const Operation& op) 
{
	Matrix retVal (left.size(), left.precision());
	broadcast(retVal, left, right, op);
	return retVal;	
}

Matrix slice(Matrix input, const Dimension& begin, const Dimension& end)
{
	auto size = end - begin;

	return Matrix(size, linearStride(size), input.precision(), input.allocation(), input[begin].address());
}

Matrix slice(Matrix input, const Dimension& begin, const Dimension& end, const Dimension& stride)
{
	auto size = (end - begin) / stride;

	return Matrix(size, stride, input.precision(), input.allocation(), input[begin].address());
}

Matrix resize(Matrix input, const Dimension& size)
{
	Matrix result(size, input.precision());
	
	copy(result, size);
	
	return result;
}

static Dimension fillInDimension(const Dimension& newSize, const Dimension& inputSize)
{
	Dimension size(newSize);
	
	// fill in remaining dimensions
	size_t remaining = inputSize.product() / size.product();

	size_t dimension = size.size();
	
	assert(inputSize.product() % size.product() == 0);
	
	// TODO: be smarter about the remainder
	for(size_t d = dimension; d < inputSize.size(); ++d)
	{
		size.push_back(remaining);
		remaining /= remaining;
	}

	assert(size.product() == inputSize.product());
	
	return size;
}

Matrix reshape(Matrix input, const Dimension& size)
{
	return Matrix(fillInDimension(size, input.size()), input.stride(), input.precision(), input.allocation(), input.data());
}

}
}


