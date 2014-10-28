/*	\file   ResultProcessor.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ResultProcessor class.
*/

#pragma once

namespace minerva
{

namespace classifiers
{

/*! \brief A class for processing the results of an engine. */
class ResultProcessor
{
public:
	virtual ~ResultProcessor();

public:
	/*! \brief Process a batch of results */
	virtual void process(const ResultVector& ) = 0;

};

}

}

