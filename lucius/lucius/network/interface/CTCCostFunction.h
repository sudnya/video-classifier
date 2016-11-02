/*  \file   CTCCostFunction.h
    \date   Feb 24th, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The header file for the CTCCostFunction class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/CostFunction.h>

namespace lucius
{

namespace network
{

/*! \brief A CTC cost function. */
class CTCCostFunction : public CostFunction
{
public:
    virtual ~CTCCostFunction();

public:
    /*! \brief Run the cost function on the specified output and reference. */
    virtual void computeCost(Bundle& bundle) const;

    /*! \brief Determine the derivative of the cost function for the specified
               output and reference. */
    virtual void computeDelta(Bundle& bundle) const;

public:
    virtual std::string typeName() const;

public:
    virtual std::unique_ptr<CostFunction> clone() const;

};

}

}



