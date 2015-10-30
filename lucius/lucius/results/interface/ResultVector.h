/*    \file   ResultVector.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ResultVector class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <cstddef>

// Forward Declarations
namespace lucius { namespace results { class Result; } }

namespace lucius
{

namespace results
{

/*! \brief A container of results.  */
class ResultVector
{
private:
    typedef std::vector<Result*> Vector;

public:
    typedef Vector::iterator       iterator;
    typedef Vector::const_iterator const_iterator;
    typedef Vector::value_type     value_type;

public:
    ResultVector();
    ~ResultVector();

public:
    ResultVector(const ResultVector&);
    ResultVector& operator=(const ResultVector&);

public:
    void push_back(Result*);

public:
    iterator       begin();
    const_iterator begin() const;

    iterator       end();
    const_iterator end() const;

public:
          value_type& back();
    const value_type& back() const;

          value_type& front();
    const value_type& front() const;

public:
    size_t size()  const;
    bool   empty() const;

public:
    void clear();

private:
    Vector _results;

};

}

}


