/*  \file   ValidationErrorObserver.cpp
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ValidationErrorObserver class.
*/

// Lucius Includes
#include <lucius/engine/interface/ValidationErrorObserver.h>

#include <lucius/engine/interface/Engine.h>
#include <lucius/engine/interface/EngineFactory.h>

#include <lucius/results/interface/ResultProcessor.h>

// Standard Library Includes
#include <fstream>

namespace lucius
{

namespace engine
{

ValidationErrorObserver::ValidationErrorObserver(
    const std::string& validationSetPath, const std::string& outputPath, size_t batchSize,
    size_t maximumSamples)
: _validationSetPath(validationSetPath), _outputPath(outputPath), _batchSize(batchSize),
  _maximumSamples(maximumSamples)
{

}

ValidationErrorObserver::~ValidationErrorObserver()
{

}

void ValidationErrorObserver::epochCompleted(Engine& runningEngine)
{
    std::unique_ptr<Engine> engine(lucius::engine::EngineFactory::create("ClassifierEngine"));

    engine->setBatchSize(_batchSize);
    engine->setModel(runningEngine.getModel());
    engine->setStandardizeInput(true);
    engine->setUseLabeledData(true);
    engine->setMaximumSamplesToRun(_maximumSamples);

    // read from database and use model to test
    engine->runOnDatabaseFile(_validationSetPath);

    // get the result processor
    auto resultProcessor = engine->getResultProcessor();

    _accuracy.push_back(resultProcessor->getCost());

    // write the output
    std::ofstream output(_outputPath);

    if(!output.is_open())
    {
        throw std::runtime_error("Could not open validation results file '" +
            _outputPath + "' for writing.");
    }

    bool first = true;

    for(auto& accuracy : _accuracy)
    {
        if(first)
        {
            first = false;
        }
        else
        {
            output << ", ";
        }

        output << accuracy;
    }
}

}

}






