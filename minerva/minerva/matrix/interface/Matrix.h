/*	\file   Matrix.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Matrix class.
*/

#pragma once

// Minerva Includes
#include <minerva/matrix/interface/Vector.h>

// Standard Library Includes
#include <string>
#include <cstddef>

// Forward Declarations
namespace minerva { namespace matrix { class MatrixImplementation; } }

namespace minerva
{

namespace matrix
{

class Matrix
{
public:
	typedef std::vector<float> FloatVector;

public:
	class FloatReference;
	class ConstFloatReference;
	class FloatPointer;
	class ConstFloatPointer;
	class iterator;
	class const_iterator;

public:
	explicit Matrix(size_t rows = 0, size_t colums = 0,
		const FloatVector& data = FloatVector());
	~Matrix();

public:
	Matrix(const Matrix&);
	Matrix& operator=(const Matrix&);

public:
	iterator	   begin();
	const_iterator begin() const;

	iterator	   end();
	const_iterator end() const;

public:
	     FloatReference operator[](size_t index);
	ConstFloatReference operator[](size_t index) const;

	     FloatReference operator()(size_t row, size_t column);
	ConstFloatReference operator()(size_t row, size_t column) const;

	void  setValue(size_t position, float value);
	float getValue(size_t position) const;

public:
	Matrix getColumn(size_t number) const;
	Matrix getRow(size_t number) const;
	
public:
	Matrix append(const Matrix& m) const;
	void resize(size_t rows, size_t columns);
	
public:
	size_t size()  const;
	bool   empty() const;

    size_t columns() const;
	size_t rows()	const;
 
 	size_t getPosition(size_t row, size_t column) const;
 
public: 
	Matrix multiply(const Matrix& m) const;
	Matrix multiply(float f) const;
	Matrix elementMultiply(const Matrix& m) const;

	Matrix add(const Matrix& m) const;
	Matrix add(float f) const;

	Matrix subtract(const Matrix& m) const;
	Matrix subtract(float f) const;

	Matrix log() const;
	Matrix negate() const;
	Matrix sigmoid() const;

public:
	Matrix slice(size_t startRow, size_t startColumn,
		size_t rows, size_t columns) const;
	Matrix transpose() const;

public:
	void negateSelf();
	void logSelf();
    void sigmoidSelf();

	void transposeSelf();

public:
    float reduceSum() const;

public:
	FloatVector data() const;
	void setDataRowMajor(const FloatVector& data);

public:
    std::string toString(size_t maxRows = 10, size_t maxColumns = 10) const;

public:

	class FloatPointer
	{
	public:
		FloatPointer(Matrix* matrix, size_t position);
	
	public:
		     FloatReference operator*();
		ConstFloatReference operator*() const;

	private:
		Matrix* _matrix;
		size_t  _position;

	public:
		friend class Matrix::ConstFloatPointer;

	};

	class ConstFloatPointer
	{
	public:
		ConstFloatPointer(const Matrix* matrix, size_t position);
		ConstFloatPointer(const FloatPointer& p);	

	public:
		ConstFloatReference operator*() const;

	private:
		const Matrix* _matrix;
		size_t        _position;
	
	};

	class FloatReference
	{
	public:
		FloatReference(Matrix* matrix, size_t position);

	public:
		FloatReference& operator=(float f);
		FloatReference& operator+=(float f);
		FloatReference& operator-=(float f);

	public:
		operator float() const;
		
	public:
		FloatPointer operator&();
		FloatPointer operator->();

	private:
		Matrix* _matrix;
		size_t  _position;

	};

	class ConstFloatReference
	{
	public:
		ConstFloatReference(const Matrix* matrix, size_t position);

	public:
		ConstFloatReference& operator=(const ConstFloatReference& ) = delete;
		
	public:
		operator float() const;
		
	public:
		ConstFloatPointer operator&();
		ConstFloatPointer operator->();

	private:
		const Matrix* _matrix;
		size_t        _position;

	};

	class const_iterator;

	class iterator: public std::iterator<std::random_access_iterator_tag, float>
	{	
	public:
		explicit iterator(Matrix*);
		iterator(Matrix* , size_t);

	public:
		FloatReference operator*();
		ConstFloatReference operator*() const;

		FloatPointer operator->();
		ConstFloatPointer operator->() const;

	public:
		iterator& operator++();
		iterator operator++(int);

		iterator& operator--();
		iterator operator--(int);
	
	public:
		difference_type operator-(const const_iterator&) const;
		difference_type operator-(const Matrix::iterator&) const;

	public:
		operator const_iterator() const;

	public:
		bool operator!=(const const_iterator& ) const;
		bool operator==(const const_iterator& ) const;
		bool operator<(const const_iterator& ) const;
		
        bool operator!=(const Matrix::iterator& ) const;
		bool operator==(const Matrix::iterator& ) const;
		bool operator<(const Matrix::iterator& ) const;

	private:
		Matrix* _matrix;
		size_t  _position;
		
	private:
		friend class Matrix::const_iterator;
	};

	class const_iterator:
		public std::iterator<std::random_access_iterator_tag, const float>
	{
	public:
		explicit const_iterator(const Matrix*);
		const_iterator(const Matrix* , size_t);
		const_iterator(const Matrix::iterator&);

	public:
		ConstFloatReference operator*() const;
		ConstFloatPointer operator->() const;

	public:
		const_iterator& operator++();
		const_iterator operator++(int);

		const_iterator& operator--();
		const_iterator operator--(int);
	
	public:
		difference_type operator-(const const_iterator&) const;
		difference_type operator-(const Matrix::iterator&) const;
	
	public:
		bool operator!=(const const_iterator& ) const;
		bool operator==(const const_iterator& ) const;
		bool operator<(const const_iterator& ) const;
		
        bool operator!=(const Matrix::iterator& ) const;
		bool operator==(const Matrix::iterator& ) const;
		bool operator<(const Matrix::iterator& ) const;

	private:
		const Matrix* _matrix;
		size_t        _position;
		
	private:
		friend class Matrix::iterator;
	};

private:
	size_t _getRow(size_t position) const;
	size_t _getColumn(size_t position) const;

private:
	Matrix(MatrixImplementation* implementation);

private:
	MatrixImplementation* _matrix;



};

}

}


