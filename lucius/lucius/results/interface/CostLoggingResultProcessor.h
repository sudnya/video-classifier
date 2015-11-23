/*  \file   CostLoggingResultProcessor.h
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the LabelMatchResultProcessor class.
*/

#pragma once

// Lucius Includes
#include <lucius/results/interface/ResultProcessor.h>

// Standard Library Includes
#include <vector>

namespace lucius
{

namespace results
{

/*! \brief A class for logging the cost results of an engine. */
class CostLoggingResultProcessor : public ResultProcessor
{
public:
    CostLoggingResultProcessor(const std::string& path);
    virtual ~CostLoggingResultProcessor();

public:
    /*! \brief Process a batch of results */
    virtual void process(const ResultVector& );

public:
    /*! \brief Return a description of the results. */
    virtual std::string toString() const;

private:
    std::vector<std::pair<double, size_t>> _costAndIterations;

private:
    std::string _outputPath;
};

}

}




