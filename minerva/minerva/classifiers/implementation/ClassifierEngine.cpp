/*	\file   ClassifierEngine.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ClassifierEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierEngine.h>

#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/database/interface/SampleDatabase.h>
#include <minerva/database/interface/Sample.h>

#include <minerva/video/interface/Image.h>
#include <minerva/video/interface/Video.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/math.h>
#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <stdexcept>
#include <fstream>
#include <random>
#include <cstdlib>
#include <algorithm>

namespace minerva
{

namespace classifiers
{

ClassifierEngine::ClassifierEngine()
{
	_maximumSamplesToRun = util::KnobDatabase::getKnobValue(
		"ClassifierEngine::MaximumVideoFrames", 20000000);
	_batchSize = util::KnobDatabase::getKnobValue(
		"ClassifierEngine::ImageBatchSize", 512);
}

ClassifierEngine::~ClassifierEngine()
{

}

void ClassifierEngine::setModel(ClassificationModel* model)
{
	_model.reset(model);
}

void ClassifierEngine::loadModel(const std::string& pathToModelFile)
{
	util::log("ClassifierEngine") << "Loading model file '" << pathToModelFile
		<<  "'...\n";
	
	_model.reset(new ClassificationModel(pathToModelFile));

	util::log("ClassifierEngine") << " model loaded.\n";
}

void ClassifierEngine::runOnDatabaseFile(const std::string& path)
{
	_model->load();

	registerModel();
		
	if(paths.empty())
	{
		throw std::runtime_error("No input path provided.");
	}
	
	_dataProducer.reset(InputDataProducerFactory::createForDatabase(path));
	
	while(!_producer->empty())
	{
		auto results = runOnBatch(std::move(_producer->pop()));
		
		_resultProcessor->process(std::move(results));
	}

	// close
	closeModel();
}

std::string ClassifierEngine::reportStatisticsString() const
{
	std::stringstream stream;

	reportStatistics(stream);
	
	return stream.str();
}

void ClassifierEngine::reportStatistics(std::ostream& stream) const
{
	// intentionally blank
}

void ClassifierEngine::registerModel()
{
	// intentionally blank
}

void ClassifierEngine::closeModel()
{
	// intentionally blank
}
	
bool ClassifierEngine::requiresLabeledData() const
{
	return false;
}

void ClassifierEngine::saveModel()
{
	if(_model) _model->save();
}

}

}


