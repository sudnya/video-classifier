/*  \file   InputOutputLearnerEngine.h
    \date   January 16, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the InputOutputLearnerEngine class.
*/

#pragma once

// Lucius Includes
#include <lucius/engine/interface/Engine.h>

// Forward Declarations
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{

namespace engine
{

/*! \brief An engine that allows a network to inspect the input and output of a training sample. */
class InputOutputLearnerEngine : public Engine
{
public:
    typedef matrix::Dimension Dimension;

public:
    InputOutputLearnerEngine();
    virtual ~InputOutputLearnerEngine();

public:
    InputOutputLearnerEngine(const InputOutputLearnerEngine&) = delete;
    InputOutputLearnerEngine& operator=(const InputOutputLearnerEngine&) = delete;

public:
    /*! \brief A callback function to get the input dimensions of the current batch. */
    Dimension getInputDimensions() const;
    /*! \brief A callback function to get a slice of the input over the specified range. */
    Matrix getInputSlice(Dimension begin, Dimension end) const;

public:
    /*! \brief A callback function to get the output dimensions of the current batch. */
    Dimension getOutputDimensions() const;
    /*! \brief A callback function to get a slice of the output over the specified range. */
    Matrix getOutputSlice(Dimension begin, Dimension end) const;

public:
    /*! \brief A callback function to get the dimensions of the current batch. */
    Dimension getDimensions() const;
    /*! \brief A callback function to get a slice of the over the specified range. */
    Matrix getSlice(Dimension begin, Dimension end) const;

private:
    virtual ResultVector runOnBatch(const Bundle& bundle);

    virtual bool requiresLabeledData() const;

};

}

}





