/*	\file   ResultProcessor.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ResultProcessor class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucious { namespace results { class ResultProcessor; } } 
namespace lucious { namespace results { class ResultVector;    } } 

namespace lucious
{

namespace results
{

/*! \brief A class for processing the results of an engine. */
class ResultProcessor
{
public:
	virtual ~ResultProcessor();

public:
	/*! \brief Process a batch of results */
	virtual void process(const ResultVector& ) = 0;

public:
	/*! \brief Return a description of the results. */
	virtual std::string toString() const = 0;

public:
	virtual void setOutputFilename(const std::string& filename);

};

}

}

