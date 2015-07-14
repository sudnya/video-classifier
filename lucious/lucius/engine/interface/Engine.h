/*  \file   Engine.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the Engine class.
*/

#pragma once

// Lucious Includes
#include <lucious/util/interface/string.h>

// Standard Library Includes
#include <ostream>
#include <memory>
#include <list>

// Forward Declarations
namespace lucious { namespace model   { class Model;             } }
namespace lucious { namespace input   { class InputDataProducer; } }
namespace lucious { namespace results { class ResultProcessor;   } }
namespace lucious { namespace network { class NeuralNetwork;     } }
namespace lucious { namespace results { class ResultVector;      } }
namespace lucious { namespace matrix  { class Matrix;            } }
namespace lucious { namespace engine  { class EngineObserver;    } }

namespace lucious
{

namespace engine
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
    /*! \brief Add model directly, the caller retains takes ownership. */
    void setModel(Model* model);

    /*! \brief Set the result handler, the engine takes ownership. */
    void setResultProcessor(ResultProcessor* processor);

    /*! \brief Set the output file name */
    void setOutputFilename(const std::string& filename);

public:
    /*! \brief Run on a single image/video database file */
    void runOnDatabaseFile(const std::string& pathToDatabase);

    /*! \brief Run on a single data producer*/
    void runOnDataProducer(InputDataProducer& producer);

public:
    /*! \brief Get the result handler, the caller takes ownership */
    ResultProcessor* extractResultProcessor();

public:
    /*! \brief Get the model, the caller retains ownership */
    Model* getModel();

    /*! \brief Get the model, the caller retains ownership */
    const Model* getModel() const;

    /*! \brief Get the result handler, the enginer retains ownership */
    ResultProcessor* getResultProcessor();

public:
    /*! \brief Set the maximum samples to be run by the engine */
    void setMaximumSamplesToRun(size_t samples);
    /*! \brief Set the epochs (passes over the entire training set) to be run by the engine */
    void setEpochs(size_t epochs);
    /*! \brief Set the number of samples to be run in a batch by the engine */
    void setBatchSize(size_t samples);
    /*! \brief Set should standardize input  */
    void setStandardizeInput(bool standardize);
    /*! \brief Add an observer */
    void addObserver(std::unique_ptr<EngineObserver>&& observer);

public:
    /*! \brief Get the number of epochs to be run by the engine */
    size_t getEpochs() const;

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
    NeuralNetwork* getAggregateNetwork();

    /*! \brief Unpack and restore networks back to the model. */
    void restoreAggregateNetwork();

public:
    /*! \brief Run the engine on the specified batch. */
    virtual ResultVector runOnBatch(Matrix&& batchOfSamples, Matrix&& referenceForEachSample) = 0;

private:
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

private:
    void _setupProducer(const std::string& databasePath);

protected:
    std::unique_ptr<InputDataProducer>  _dataProducer;
    std::unique_ptr<ResultProcessor>    _resultProcessor;

protected:
    std::unique_ptr<NeuralNetwork> _aggregateNetwork;

private:
    typedef std::list<std::unique_ptr<EngineObserver>> ObserverList;

private:
    ObserverList _observers;

};

}

}


