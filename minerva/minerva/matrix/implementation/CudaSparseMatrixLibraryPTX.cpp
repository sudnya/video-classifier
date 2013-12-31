/*! \file   CudaSparseMatrixLibraryPTX.cpp
	\author Gregory Diamos
	\date   Monday December 30, 2013
	\brief  The source file for the PTX loader for the cuda sparse matrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/CudaSparseMatrixLibraryPTX.h>

namespace minerva
{

namespace matrix
{

static const char* ptx = "";

const char* getCudaSparseMatrixLibraryPtx()
{
	return ptx;
}

}

}


