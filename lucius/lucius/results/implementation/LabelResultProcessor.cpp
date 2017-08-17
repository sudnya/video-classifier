/*  \file   LabelResultProcessor.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LabelResultProcessor class.
*/

// Lucius Includes
#include <lucius/results/interface/LabelResultProcessor.h>
#include <lucius/results/interface/ResultVector.h>
#include <lucius/results/interface/LabelResult.h>

#include <lucius/util/interface/string.h>

// Standard Library Includes
#include <vector>

namespace lucius
{

namespace results
{

class LabelResultProcessorImplementation
{
public:
    void process(const ResultVector& results)
    {
        for(auto result : results)
        {
            auto labelResult = dynamic_cast<LabelResult*>(result);

            if(labelResult != nullptr)
            {
                _labels.push_back(labelResult->label);
            }
        }
    }

public:
    std::string toString() const
    {
        return util::join(_labels, "\n");
    }

private:
    std::vector<std::string> _labels;


};

LabelResultProcessor::LabelResultProcessor()
: _implementation(std::make_unique<LabelResultProcessorImplementation>())
{

}

LabelResultProcessor::~LabelResultProcessor()
{

}

void LabelResultProcessor::process(const ResultVector& v)
{
    _implementation->process(v);
}

std::string LabelResultProcessor::toString() const
{
    return _implementation->toString();
}

}

}


