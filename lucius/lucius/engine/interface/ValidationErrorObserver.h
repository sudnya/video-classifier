/*  \file   ValidationErrorObserver.h
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ValidationErrorObserver class.
*/

#pragma once

// Lucius Includes
#include <lucius/engine/interface/EngineObserver.h>

// Standard Library Includes
#include <vector>
#include <string>

namespace lucius
{

namespace engine
{

/*! \brief An observer that computes and stores validation set error after each epoch. */
class ValidationErrorObserver : public EngineObserver
{
public:
    ValidationErrorObserver(const std::string& validationSetPath, const std::string& outputPath,
        size_t batchSize, size_t maximumSamples);
    virtual ~ValidationErrorObserver();

public:
    /*! \brief Called after each iteration completed by the engine. */
    virtual void epochCompleted(Engine& engine);

private:
    std::vector<double> _accuracy;

private:
    std::string _validationSetPath;
    std::string _outputPath;

private:
    size_t _batchSize;
    size_t _maximumSamples;
};

}

}





