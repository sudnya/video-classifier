/*  \file   GraphemeMatchResultProcessor.cpp
    \date   Saturday March 20, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the GraphemeMatchResultProcessor class.
*/

// Lucius Includes
#include <lucius/results/interface/GraphemeMatchResultProcessor.h>
#include <lucius/results/interface/LabelMatchResult.h>
#include <lucius/results/interface/CostResult.h>
#include <lucius/results/interface/ResultVector.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/EditDistance.h>
#include <lucius/util/interface/string.h>

// Standard Library Includes
#include <sstream>
#include <cassert>


namespace lucius
{

namespace results
{

GraphemeMatchResultProcessor::GraphemeMatchResultProcessor()
: _totalGraphemes(0), _totalGraphemeMatches(0), _totalSamples(0),
  _totalSampleMatches(0), _cost(0.0), _costCount(0.0)
{

}

GraphemeMatchResultProcessor::~GraphemeMatchResultProcessor()
{
}

void GraphemeMatchResultProcessor::process(const ResultVector& results)
{
    _totalSamples += results.size();

    util::log("GraphemeMatchResultProcessor") << "Processing batch of "
        << results.size() << " results.\n";

    for(auto& result : results)
    {
        auto matchResult = dynamic_cast<LabelMatchResult*>(result);

        // skip results other than label match
        if(matchResult == nullptr)
        {
            auto costResult = dynamic_cast<CostResult*>(result);

            _cost += costResult->cost;
            _costCount += 1;

            util::log("GraphemeMatchResultProcessor::Detail") << " cost '"
                << costResult->cost << "'\n";

            continue;
        }

        auto label = util::strip(matchResult->label, "-SEPARATOR-");

        size_t distance = util::editDistance(matchResult->reference, label);

        util::log("GraphemeMatchResultProcessor::Detail") << " label '" << label
            << "', reference '" << matchResult->reference << "' with distance "
            << distance << "\n";

        if(label == matchResult->reference)
        {
            ++_totalSampleMatches;
        }

        _totalGraphemes += matchResult->reference.size();
        _totalGraphemeMatches +=
            matchResult->reference.size() -
            std::min(matchResult->reference.size(), distance);
    }

}

std::string GraphemeMatchResultProcessor::toString() const
{
    std::stringstream stream;

    stream << "Grapheme accuracy is: " << getAccuracy() << " (" << _totalGraphemeMatches
        << " / " << _totalGraphemes << "), sample accuracy: : " << getSampleAccuracy()
        << " (" << _totalSampleMatches
        << " / " << _totalSamples << ")";

    return stream.str();

}

double GraphemeMatchResultProcessor::getAccuracy() const
{
    return (100.0 * _totalGraphemeMatches) / (_totalGraphemes);
}

double GraphemeMatchResultProcessor::getCost() const
{
    return _cost / _costCount;
}

double GraphemeMatchResultProcessor::getSampleAccuracy() const
{
    return (100.0 * _totalSampleMatches) / (_totalSamples);
}

}

}





