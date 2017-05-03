/*  \file   MatrixOperations.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the lazy matrix operation interface functions.
*/

#pragma once

// Forward Declarations
class lucius { class lazy { class LazyValue; } }

namespace lucius
{

namespace lazy
{

/*! \brief Apply an operation to an input, resulting in a single output. */
LazyValue apply(const LazyValue& input, const Operation& op);
void apply(LazyValue& output, const LazyValue& input, const Operation& op);

/*! \brief Apply an operation to two inputs, resulting in a single output. */
LazyValue applyBinary(const LazyValue& left, const LazyValue& right, const Operation& op);
void applyBinary(LazyValue& output, const LazyValue& left, const LazyValue& right,
    const Operation& op);

/*! \brief Apply an operation along a set of dimensions to be reduced. */
void reduce(LazyValue& result, const LazyValue& input, const Dimension& d, const Operation& op);
LazyValue reduce(const LazyValue& input, const Dimension& d, const Operation& op);

/*! \brief Apply an operation to two inputs, replicating the left dimension
           along the specified dimensions.
*/
void broadcast(LazyValue& result, const LazyValue& left, const LazyValue& right,
    const Dimension& d, const Operation& op);
LazyValue broadcast(const LazyValue& left, const LazyValue& right, const Dimension& d,
    const Operation& op);

void zeros(LazyValue& result);
LazyValue zeros(const Dimension& size, const Precision& precision);

void ones(LazyValue& result);
LazyValue ones(const Dimension& size, const Precision& precision);

void range(LazyValue& result);
LazyValue range(const Dimension& size, const Precision& precision);

}

}




