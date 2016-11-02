/*    \file   CostFunction.h
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the CostFunction class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucius { namespace matrix  { class Matrix; } }
namespace lucius { namespace network { class Bundle; } }

namespace lucius
{

namespace network
{

class CostFunction
{
public:
    typedef matrix::Matrix Matrix;

public:
    virtual ~CostFunction();

public:
    /*! \brief Run the cost function on the specified output and reference. */
    virtual void computeCost(Bundle& bundle) const = 0;

    /*! \brief Determine the change in the output needed to minimize the cost function. */
    virtual void computeDelta(Bundle& bundle) const = 0;

public:
    virtual std::string typeName() const = 0;

public:
    /*! \brief Clone */
    virtual std::unique_ptr<CostFunction> clone() const = 0;

};

}

}

