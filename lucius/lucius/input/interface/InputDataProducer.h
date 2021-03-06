/*  \file   InputDataProducer.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the InputDataProducer class.
*/

#pragma once

// Lucius Includes
#include <lucius/util/interface/string.h>

// Forward Declarations
namespace lucius { namespace network { class Bundle;    } }
namespace lucius { namespace matrix  { class Dimension; } }
namespace lucius { namespace matrix  { class Matrix;    } }
namespace lucius { namespace model   { class Model;     } }

// Standard Library Includes
#include <utility>

namespace lucius
{

namespace input
{

/*! \brief A class for accessing a stream of data */
class InputDataProducer
{
public:
    typedef matrix::Dimension Dimension;
    typedef matrix::Matrix    Matrix;
    typedef network::Bundle   Bundle;

public:
    InputDataProducer();
    virtual ~InputDataProducer();

public:
    /*! \brief Initialize the state of the producer after all parameters have been set. */
    virtual void initialize() = 0;

    /*! \brief Deque a set of samples from the producer.

        Note: the caller must ensure that the producer is not empty.
    */
    virtual Bundle pop() = 0;

    /*! \brief Return true if there are no more samples. */
    virtual bool empty() const = 0;

    /*! \brief Reset the producer to its original state, all previously
        popped samples should now be available again. */
    virtual void reset() = 0;

    /*! \brief Get the total number of unique samples that can be produced. */
    virtual size_t getUniqueSampleCount() const = 0;

public:
    /*! \brief Add a new sample to the producer directly. */
    virtual void addRawSample(const void* data, size_t size, const std::string& type,
        const std::string& label);

public:
    void   setEpochs(size_t epochs);
    size_t getEpochs() const;

public:
    void   setPassesPerEpoch(size_t passes);
    size_t getPassesPerEpoch() const;

public:
    void   setBatchSize(size_t batchSize);
    size_t getBatchSize() const;

public:
    void   setMaximumSamplesToRun(size_t batchSize);
    size_t getMaximumSamplesToRun() const;

public:
    void setStandardizeInput(bool standardize);
    bool getStandardizeInput() const;

public:
    void setRequiresLabeledData(bool requiresLabeledData);
    bool getRequiresLabeledData() const;

public:
    virtual void setModel(model::Model* model);
    const model::Model* getModel() const;
    model::Model* getModel();

public:
    Dimension getInputSize() const;

public:
    util::StringVector getOutputLabels() const;

public:
    /*! \brief Standardize an input feature matrix. */
    void standardize(Matrix& input);

private:
    size_t _epochs;
    size_t _passesPerEpoch;
    bool   _requiresLabeledData;
    size_t _batchSize;
    size_t _maximumSamplesToRun;
    bool   _standardizeInput;

protected:
    model::Model* _model;

};

}

}

