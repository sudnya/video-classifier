/*    \file   SampleStatisticsEngine.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the SampleStatisticsEngine class.
*/

#pragma once

// Lucius Includes
#include <lucius/engine/interface/Engine.h>

namespace lucius
{

namespace engine
{

/*! \brief A class for computing sample statistics and embedding them in the model. */
class SampleStatisticsEngine : public Engine
{
public:
    SampleStatisticsEngine();
    ~SampleStatisticsEngine();

private:
    virtual void registerModel();
    virtual void closeModel();

private:
    virtual ResultVector runOnBatch(const Bundle& bundle);

private:
    double _samples;
    double _mean;
    double _standardDeviation;
    double _sumOfSquaresOfDifferences;

};

}

}


