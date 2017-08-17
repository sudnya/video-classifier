/*  \file   MatrixOperations.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the lazy matrix operation interface functions.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Dimension; } }
namespace lucius { namespace matrix { class Precision; } }

namespace lucius { namespace lazy { class LazyValue; } }

namespace lucius { namespace lazy { class UnaryOperator; } }
namespace lucius { namespace lazy { class BinaryOperator; } }

namespace lucius
{

namespace lazy
{

// Namespace imports
using Dimension = matrix::Dimension;
using Precision = matrix::Precision;

/*! \brief Apply an operation to an input, resulting in a single output. */
LazyValue apply(const LazyValue& input, const UnaryOperator& op);
void apply(LazyValue& output, const LazyValue& input, const UnaryOperator& op);

/*! \brief Apply an operation to two inputs, resulting in a single output. */
LazyValue applyBinary(const LazyValue& left, const LazyValue& right, const BinaryOperator& op);
void applyBinary(LazyValue& output, const LazyValue& left, const LazyValue& right,
    const BinaryOperator& op);

/*! \brief Apply an operation along a set of dimensions to be reduced. */
void reduce(LazyValue& result, const LazyValue& input, const Dimension& d,
    const BinaryOperator& op);
LazyValue reduce(const LazyValue& input, const Dimension& d,
    const BinaryOperator& op);

/*! \brief Apply an operation to two inputs, replicating the left dimension
           along the specified dimensions.
*/
void broadcast(LazyValue& result, const LazyValue& left, const LazyValue& right,
    const Dimension& d, const BinaryOperator& op);
LazyValue broadcast(const LazyValue& left, const LazyValue& right, const Dimension& d,
    const BinaryOperator& op);

void zeros(LazyValue& result);
LazyValue zeros(const Dimension& size, const Precision& precision);

void ones(LazyValue& result);
LazyValue ones(const Dimension& size, const Precision& precision);

void range(LazyValue& result);
LazyValue range(const Dimension& size, const Precision& precision);

}

}




