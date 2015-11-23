/*  \file   CostLoggingResultProcessor.h
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LabelMatchResultProcessor class.
*/

// Lucius Includes
#include <lucius/results/interface/CostLoggingResultProcessor.h>

#include <lucius/results/interface/ResultVector.h>
#include <lucius/results/interface/CostResult.h>

// Standard Library Includes
#include <fstream>
#include <sstream>

namespace lucius
{

namespace results
{

CostLoggingResultProcessor::CostLoggingResultProcessor(const std::string& path)
: _outputPath(path)
{

}

CostLoggingResultProcessor::~CostLoggingResultProcessor()
{

}

void CostLoggingResultProcessor::process(const ResultVector& results)
{
    for(auto result : results)
    {
        auto* costResult = dynamic_cast<CostResult*>(result);

        if(costResult == nullptr)
        {
            continue;
        }

        _costAndIterations.push_back(std::make_pair(costResult->cost, costResult->iteration));
    }

    std::ofstream file(_outputPath);

    if(file.is_open())
    {
        file << toString();
    }
}

std::string CostLoggingResultProcessor::toString() const
{
    std::stringstream stream;

    bool first = false;

    for(auto costAndIteration : _costAndIterations)
    {
        if(!first)
        {
            first = true;
        }
        else
        {
            stream << "\n";
        }

        stream << costAndIteration.first << ", " << costAndIteration.second;
    }

    return stream.str();
}

}

}





