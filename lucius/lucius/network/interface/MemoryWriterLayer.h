/*  \file   MemoryWriterLayer.h
    \author Gregory Diamos
    \date   May 15, 2017
    \brief  The interface for the MemoryWriterLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/ControllerLayer.h>

namespace lucius
{

namespace network
{

/*! \brief A layer that iteratively writes into an associative memory on each timestep. */
class MemoryWriterLayer : public ControllerLayer
{
public:
    MemoryWriterLayer();
    MemoryWriterLayer(size_t cellSize, size_t cellCount);
    virtual ~MemoryWriterLayer();

public:
    MemoryWriterLayer(const MemoryWriterLayer& );
    MemoryWriterLayer& operator=(const MemoryWriterLayer&);

public:
    virtual void initialize();

public:
    virtual void runForwardImplementation(Bundle& bundle);
    virtual void runReverseImplementation(Bundle& bundle);

public:
    virtual       MatrixVector& weights();
    virtual const MatrixVector& weights() const;

public:
    virtual const matrix::Precision& precision() const;

public:
    virtual double computeWeightCost() const;

public:
    virtual Dimension getInputSize()  const;
    virtual Dimension getOutputSize() const;

public:
    virtual size_t getInputCount()  const;
    virtual size_t getOutputCount() const;

public:
    virtual size_t totalNeurons()      const;
    virtual size_t totalConnections() const;

public:
    virtual size_t getFloatingPointOperationCount() const;
    virtual size_t getActivationMemory() const;

public:
    virtual void save(util::OutputTarArchive& archive, util::PropertyTree& properties) const;
    virtual void load(util::InputTarArchive& archive, const util::PropertyTree& properties);

public:
    virtual std::unique_ptr<Layer> clone() const;

public:
    virtual std::string getTypeName() const;

public:
    void setController(std::unique_ptr<Layer>&& l);

private:
    size_t _cellSize;
    size_t _cellCount;

};

}

}


