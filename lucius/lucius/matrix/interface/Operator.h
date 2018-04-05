/*  \file   Operator.h
    \date   February 19, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the Operator classes.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace matrix { class StaticOperator; } }

namespace lucius
{

namespace matrix
{

/*! \brief A dynamic container for a static operator description. */
class Operator
{
public:
    Operator();
    ~Operator();

public:
    Operator(const Operator&);
    Operator(const StaticOperator&);

public:
    Operator& operator=(const Operator& );
    Operator& operator=(const StaticOperator& );

public:
    const StaticOperator& getStaticOperator() const;
          StaticOperator& getStaticOperator();

private:
    std::unique_ptr<StaticOperator> _operator;

};

bool operator==(const Operator&, const StaticOperator& );
bool operator==(const StaticOperator&, const Operator& );

}

}

