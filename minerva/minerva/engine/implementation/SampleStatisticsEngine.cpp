/*    \file   SampleStatisticsEngine.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the SampleStatisticsEngine class.
*/

// Lucious Includes
#include <lucious/engine/interface/SampleStatisticsEngine.h>

#include <lucious/model/interface/Model.h>

#include <lucious/results/interface/ResultVector.h>

#include <lucious/matrix/interface/Matrix.h>

#include <lucious/util/interface/debug.h>

// Standard Library Includes
#include <cmath>

namespace lucious
{

namespace engine
{

SampleStatisticsEngine::SampleStatisticsEngine()
{
    setEpochs(1);
    setStandardizeInput(true);
}

SampleStatisticsEngine::~SampleStatisticsEngine()
{

}

void SampleStatisticsEngine::registerModel()
{
    _mean = 0.0;
    _standardDeviation = 0.0;
    _samples = 0.0;
    _sumOfSquaresOfDifferences = 0.0;

    getModel()->setAttribute("InputSampleMean",              0.0);
    getModel()->setAttribute("InputSampleStandardDeviation", 1.0);
}

void SampleStatisticsEngine::closeModel()
{
    _standardDeviation = std::sqrt(_sumOfSquaresOfDifferences / (_samples - 1.0));

    getModel()->setAttribute("InputSampleMean",              _mean);
    getModel()->setAttribute("InputSampleStandardDeviation", _standardDeviation);

    // TODO
    // saveModel();
}

SampleStatisticsEngine::ResultVector SampleStatisticsEngine::runOnBatch(Matrix&& input, Matrix&& reference)
{
    util::log("SimpleStatisticsEngine") << "Computing sample statistics over " << input.size().product() <<  " elements...\n";

    for(auto element : input)
    {
        _samples += 1.0f;

        float delta = element - _mean;

        _mean = _mean + delta / _samples;

        _sumOfSquaresOfDifferences += delta * (element - _mean);
    }

    return ResultVector();
}

}

}


