/*	\file   MatrixImplementation.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the MatrixImplementation class.
*/

// Minerva Includes
#include <minerva/matrix/interface/MatrixImplementation.h>

#include <minerva/matrix/interface/NaiveMatrix.h>
#include <minerva/matrix/interface/CublasMatrix.h>
#include <minerva/matrix/interface/AtlasMatrix.h>

namespace minerva
{

namespace matrix
{


MatrixImplementation::MatrixImplementation(size_t rows, size_t columns)
: _rows(rows), _columns(columns)
{

}

MatrixImplementation::~MatrixImplementation()
{

}

MatrixImplementation::Value* MatrixImplementation::createBestImplementation(
	size_t rows, size_t columns, const FloatVector& f)
{
	Value* matrix = nullptr;
	
	if(matrix == nullptr && CublasMatrix::isSupported())
	{
		matrix = new CublasMatrix(rows, columns, f);
	}
	
	if(matrix == nullptr && AtlasMatrix::isSupported())
	{
		matrix = new AtlasMatrix(rows, columns, f);
	}
	
	if(matrix == nullptr)
	{	
		matrix = new NaiveMatrix(rows, columns, f);
	}
	
	return matrix;
}

}

}




