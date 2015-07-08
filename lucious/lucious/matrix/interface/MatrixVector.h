/*    \file   MatrixVector.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the MatrixVector class.
*/


#pragma once

// Lucious Includes
#include <lucious/matrix/interface/Matrix.h>

// Standard Library Includes
#include <vector>

namespace lucious
{

namespace matrix
{

class MatrixVector
{
private:
    typedef Matrix value_type;
    typedef value_type& reference_type;
    typedef const value_type& const_reference_type;

public:
    typedef std::vector<value_type> Vector;
    typedef std::vector<Dimension> DimensionVector;

public:
    typedef Vector::iterator       iterator;
    typedef Vector::const_iterator const_iterator;

    typedef Vector::reverse_iterator       reverse_iterator;
    typedef Vector::const_reverse_iterator const_reverse_iterator;

public:
    MatrixVector();

public:
    MatrixVector(const DimensionVector& sizes);
    MatrixVector(const DimensionVector& sizes, const Precision& p);

public:
    MatrixVector(std::initializer_list<Matrix> l);

public:
    MatrixVector(const MatrixVector&);
    MatrixVector(MatrixVector&&);

public:
    MatrixVector& operator=(const MatrixVector&  );
    MatrixVector& operator=(MatrixVector&& );

public:
    reference_type operator[](size_t i);
    const_reference_type operator[](size_t i) const;

public:
    bool empty() const;

public:
    size_t size() const;
    DimensionVector sizes() const;

public:
    void reserve(size_t size);
    void resize(size_t size);

    void clear();

public:
    iterator       begin();
    const_iterator begin() const;

    iterator       end();
    const_iterator end() const;

public:
    reverse_iterator       rbegin();
    const_reverse_iterator rbegin() const;

    reverse_iterator       rend();
    const_reverse_iterator rend() const;

public:
    void push_back(const Matrix&  );
    void push_back(Matrix&&);
    void push_back(MatrixVector&&);
    void push_back(const MatrixVector&);

public:
    void pop_back();

public:
    Matrix& back();
    const Matrix& back() const;

    Matrix& front();
    const Matrix& front() const;

public:
    std::string toString() const;

public:
    bool operator==(const MatrixVector&) const;
    bool operator!=(const MatrixVector&) const;

private:
    Vector _matrix;

};

}

}



