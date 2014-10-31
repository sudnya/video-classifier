/*	\file   ClassifierEngine.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ClassifierEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/video/interface/ImageVector.h>

#include <minerva/util/interface/string.h>

// Standard Library Includes
#include <ostream>
#include <memory>

// Forward Declarations
namespace minerva { namespace model       { class ClassificationModel; } }
namespace minerva { namespace input       { class InputDataProducer;   } }
namespace minerva { namespace classifiers { class ResultProcessor;     } }

namespace minerva
{

namespace classifiers
{

/*! \brief A generic interface to a classifier with support for an arbitrarily
	large input data stream.
 */
class ClassifierEngine
{
public:
	ClassifierEngine();
	virtual ~ClassifierEngine();

public:
	/*! \brief Load a model from a file, the engine takes ownership. */
	void loadModel(const std::string& pathToModelFile);
	
	/*! \brief Add model directly, the engine takes ownership. */
	void setModel(model::ClassificationModel* model);
	
	/*! \brief Set the result handler, the engine takes ownership. */
	void setResultProcessor(ResultProcessor* processor);

	/*! \brief Run on a single image/video database file */
	void runOnDatabaseFile(const std::string& pathToDatabase);

	/*! \brief Set the output file name */
	void setOutputFilename(const std::string& filename);

public:
	/*! \brief Get the model, the caller takes ownership. */
	model::ClassificationModel* extractModel();
	
	/*! \brief Get the result handler, the caller takes ownership */
	ResultProducer* extractResultProcessor();

public:
	/*! \brief Get the model, the engine retains ownership */
	model::ClassificationModel* getModel();
	
	/*! \brief Get the result handler, the enginer retains ownership */
	ResultProducer* getResultProcessor();

public:
	/*! \brief Set the maximum samples to be run by the engine */
	void setMaximumSamplesToRun(size_t samples);
	/*! \brief Set the number of samples to be run in a batch by the engine */
	void setBatchSize(size_t samples);
	/*! \brief Should the engine be allowed to run the same sample multiple times */
	void setAllowSamplingWithReplacement(bool);
	
public:
	std::string reportStatisticsString() const;

public:
	virtual void reportStatistics(std::ostream& stream) const;

protected:
	/*! \brief Called before the engine starts running on a given model */
	virtual void registerModel();
	/*! \brief Called after the engine finishes running on a given model */
	virtual void closeModel();

	/*! \brief Determines whether or not the engine must use labeled data */	
	virtual bool requiresLabeledData() const;

protected:
	void saveModel();

public:
	virtual ResultVector runOnBatch(matrix::Matrix&& batchOfSamples) = 0;
	virtual size_t getInputFeatureCount() const = 0;
	
public:
	ClassifierEngine(const ClassifierEngine&) = delete;
	ClassifierEngine& operator=(const ClassifierEngine&) = delete;

protected:
	typedef model::ClassificationModel ClassificationModel;

protected:
	std::unique_ptr<ClassificationModel> _model;
	std::unique_ptr<InputDataProducer>   _dataProducer;
	std::unique_ptr<ResultProcessor>     _resultProcesssor;

};

}

}


