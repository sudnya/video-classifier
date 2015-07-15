/*    \file   LabelMatchResultProcessor.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LabelMatchResultProcessor class.
*/

// Lucius Includes
#include <lucius/results/interface/LabelMatchResultProcessor.h>
#include <lucius/results/interface/LabelMatchResult.h>
#include <lucius/results/interface/ResultVector.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <sstream>
#include <cassert>

namespace lucius
{

namespace results
{

LabelMatchResultProcessor::LabelMatchResultProcessor()
: _matches(0), _total(0)
{

}

LabelMatchResultProcessor::~LabelMatchResultProcessor()
{

}

void LabelMatchResultProcessor::process(const ResultVector& results)
{
    _total += results.size();

    util::log("LabelMatchResultProcessor") << "Processing batch of " << results.size() << " results.\n";

    for(auto result : results)
    {
        auto matchResult = dynamic_cast<LabelMatchResult*>(result);

        // skip results other than label match
        if(matchResult == nullptr)
        {
            continue;
        }

        util::log("LabelMatchResultProcessor") << " label '" << matchResult->label << "', reference '" << matchResult->reference << "'\n";

        if(matchResult->label == matchResult->reference)
        {
            ++_matches;
        }
    }

    util::log("LabelMatchResultProcessor") << toString() << "\n";
}

std::string LabelMatchResultProcessor::toString() const
{
    std::stringstream stream;

    stream << "Accuracy is: " << getAccuracy() << " (" << _matches << " / " << _total << ")";

    return stream.str();
}

float LabelMatchResultProcessor::getAccuracy() const
{
    return (_matches * 100.0f) / _total;
}

}

}




