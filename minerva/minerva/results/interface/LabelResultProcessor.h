/*	\file   LabelResultProcessor.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LabelResultProcessor class.
*/

#pragma once

#include <lucious/results/interface/ResultProcessor.h>

namespace lucious
{

namespace results
{

/*! \brief A class for processing the results of an engine. */
class LabelResultProcessor : public ResultProcessor
{
public:
	virtual ~LabelResultProcessor();

public:
	/*! \brief Process a batch of results */
	virtual void process(const ResultVector& );

public:
	/*! \brief Return a description of the results. */
	virtual std::string toString() const;

};

}

}

