/*  \file   LabelMatchResultProcessor.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the LabelMatchResultProcessor class.
*/

#pragma once

#include <lucius/results/interface/ResultProcessor.h>

namespace lucius
{

namespace results
{

/*! \brief A class for processing the label match results of an engine. */
class LabelMatchResultProcessor : public ResultProcessor
{
public:
    LabelMatchResultProcessor();
    virtual ~LabelMatchResultProcessor();

public:
    /*! \brief Process a batch of results */
    virtual void process(const ResultVector& );

public:
    /*! \brief Return a description of the results. */
    virtual std::string toString() const;

public:
    virtual double getAccuracy() const;
    virtual double getCost() const;

private:
    size_t _matches;
    size_t _total;
    double _cost;

};

}

}



