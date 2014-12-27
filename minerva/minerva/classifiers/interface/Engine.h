/*	\file   Engine.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Engine class.
*/

#pragma once

// Minerva Includes
#include <minerva/util/interface/string.h>

// Standard Library Includes
#include <ostream>
#include <memory>

// Forward Declarations
namespace minerva { namespace model   { class Model;             } }
namespace minerva { namespace input   { class InputDataProducer; } }
namespace minerva { namespace results { class ResultProcessor;   } }
namespace minerva { namespace network { class NeuralNetwork;     } }
namespace minerva { namespace results { class ResultVector;      } }
namespace minerva { namespace matrix  { class Matrix;            } }

namespace minerva
{

namespace classifiers
{

/*! \brief A generic interface to a classifier with support for an arbitrarily
	large input data stream.
 */
class Engine
{
public:
	typedef input::InputDataProducer InputDataProducer;
	typedef model::Model             Model;
	typedef results::ResultProcessor ResultProcessor;
	typedef results::ResultVector    ResultVector;
	typedef network::NeuralNetwork   NeuralNetwork;
	typedef matrix::Matrix           Matrix;

public:
	Engine();
	virtual ~Engine();

public:
	/*! \brief Load a model from a file, the engine takes ownership. */
	void loadModel(const std::string& pathToModelFile);
	
	/*! \brief Add model directly, the engine takes ownership. */
	void setModel(Model* model);
	
	/*! \brief Set the result handler, the engine takes ownership. */
	void setResultProcessor(ResultProcessor* processor);

	/*! \brief Run on a single image/video database file */
	void runOnDatabaseFile(const std::string& pathToDatabase);

	/*! \brief Set the output file name */
	void setOutputFilename(const std::string& filename);

public:
	/*! \brief Get the model, the caller takes ownership. */
	Model* extractModel();
	
	/*! \brief Get the result handler, the caller takes ownership */
	ResultProcessor* extractResultProcessor();

public:
	/*! \brief Get the model, the engine retains ownership */
	Model* getModel();
	
	/*! \brief Get the result handler, the enginer retains ownership */
	ResultProcessor* getResultProcessor();

public:
	/*! \brief Set the maximum samples to be run by the engine */
	void setMaximumSamplesToRun(size_t samples);
	/*! \brief Set the epochs (passes over the entire training set) to be run by the engine */
	void setEpochs(size_t epochs);
	/*! \brief Set the number of samples to be run in a batch by the engine */
	void setBatchSize(size_t samples);

protected:
	/*! \brief Called before the engine starts running on a given model */
	virtual void registerModel();
	/*! \brief Called after the engine finishes running on a given model */
	virtual void closeModel();

	/*! \brief Determines whether or not the engine must use labeled data */	
	virtual bool requiresLabeledData() const;

protected:
	/*! \brief Save the model to persistent storage. */
	void saveModel();

protected:
	/*! \brief Extract and merge all networks from the model. */
	NeuralNetwork getAggregateNetwork();

	/*! \brief Unpack and restore networks back to the model. */
	void restoreAggregateNetwork(NeuralNetwork& network);
	
public:
	/*! \brief Run the engine on the specified batch. */
	virtual ResultVector runOnBatch(Matrix&& batchOfSamples, Matrix&& referenceForEachSample) = 0;
	
private:
	Engine(const Engine&) = delete;
	Engine& operator=(const Engine&) = delete;

private:
	void _setupProducer(const std::string& databasePath);

protected:
	std::unique_ptr<Model>              _model;
	std::unique_ptr<InputDataProducer>  _dataProducer;
	std::unique_ptr<ResultProcessor>    _resultProcessor;

};

}

}


