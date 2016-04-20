/*  \file   GraphemeMatchResultProcessor.h
    \date   Saturday March 20, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the GraphemeMatchResultProcessor class.
*/

#pragma once

#include <lucius/results/interface/ResultProcessor.h>

namespace lucius
{

namespace results
{

/*! \brief A class for processing the label match results of an engine. */
class GraphemeMatchResultProcessor : public ResultProcessor
{
public:
    GraphemeMatchResultProcessor();
    virtual ~GraphemeMatchResultProcessor();

public:
    /*! \brief Process a batch of results */
    virtual void process(const ResultVector& );

public:
    /*! \brief Return a description of the results. */
    virtual std::string toString() const;

public:
    virtual double getAccuracy() const;
    virtual double getCost() const;

public:
    double getSampleAccuracy() const;

private:
    size_t _totalGraphemes;
    size_t _totalGraphemeMatches;
    size_t _totalSamples;
    size_t _totalSampleMatches;

    double _cost;
    size_t _costCount;

};

}

}




