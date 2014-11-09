/*	\file   FeatureResultProcessor.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the FeatureResultProcessor class.
*/

#pragma once

#include <minerva/results/interface/ResultProcessor.h>

namespace minerva
{

namespace results
{

/*! \brief A class for processing the results of an engine. */
class FeatureResultProcessor : public ResultProcessor
{
public:
	virtual ~FeatureResultProcessor();

public:
	/*! \brief Process a batch of results */
	virtual void process(const ResultVector& );

};

}

}

