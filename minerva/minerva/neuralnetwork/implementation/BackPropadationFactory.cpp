/*! \file   BackPropagationFactory.cpp
	\author Gregory Diamos
	\date   Sunday December 22, 2013
	\brief  The source file for the BackPropagationFactory class.
*/

// Minerva Includes
#include <minerva/neuralnetwork/interface/BackPropagationFactory.h>

#include <minerva/neuralnetwork/interface/DenseBackPropagation.h>
#include <minerva/neuralnetwork/interface/SparseBackPropagation.h>

namespace minerva
{

namespace neuralnetwork
{

BackPropagation* BackPropagationFactory::create(const std::string& name)
{
	BackPropagation* backPropagation = nullptr;
	
	if(name == "SparseBackPropagation")
	{
		backPropagation = new SparseBackPropagation;
	}
	else if(name == "DenseBackPropagation")
	{
		backPropagation = new DenseBackPropagation;
	}
	
	return backPropagation;
}

}

}




