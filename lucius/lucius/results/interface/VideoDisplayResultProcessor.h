/*    \file   VideoDisplayResultProcessor.h
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the VideoDisplayResultProcessor class.
*/

#pragma once

#include <lucius/results/interface/ResultProcessor.h>

namespace lucius
{

namespace results
{

/*! \brief A class for processing the results of an engine. */
class VideoDisplayResultProcessor : public ResultProcessor
{
public:
    virtual ~VideoDisplayResultProcessor();

public:
    /*! \brief Process a batch of results */
    virtual void process(const ResultVector& );

public:
    /*! \brief Return a description of the results. */
    virtual std::string toString() const;

};

}

}


