/*    \file   SampleStatisticsEngine.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the SampleStatisticsEngine class.
*/

// Lucius Includes
#include <lucius/engine/interface/SampleStatisticsEngine.h>

#include <lucius/model/interface/Model.h>

#include <lucius/network/interface/Bundle.h>

#include <lucius/results/interface/ResultVector.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <cmath>

namespace lucius
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
}

SampleStatisticsEngine::ResultVector SampleStatisticsEngine::runOnBatch(const Bundle& bundle)
{
    auto& input = bundle["inputActivations"].get<matrix::MatrixVector>().front();

    util::log("SimpleStatisticsEngine") << "Computing sample statistics over "
        << input.size().product() <<  " elements...\n";

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


