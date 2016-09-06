/*  \file   RecurrentLayer.h
    \author Sudnya Diamos
    \date   May 9, 2016
    \brief  The interface for the RecurrentLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

namespace lucius
{
namespace network
{

/* \brief An implementation of a generic recurrent layer. */
class RecurrentLayer : public Layer
{
public:
    RecurrentLayer();
    RecurrentLayer(size_t layerSize, size_t expectedMiniBatchSize, size_t layers,
        int direction, int layerType, int inputMode);
    RecurrentLayer(size_t layerSize, size_t expectedMiniBatchSize, size_t layers,
        int direction, int layerType, int inputMode, const matrix::Precision&);
    virtual ~RecurrentLayer();

public:
    RecurrentLayer(const RecurrentLayer& );
    RecurrentLayer& operator=(const RecurrentLayer&);

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

private:
    std::unique_ptr<MatrixVector> _parameters;

private:
    matrix::Matrix& _weights;

private:
    size_t _layerSize;
    size_t _expectedMiniBatchSize;
    size_t _layers;

private:
    int _direction;
    int _layerType;
    int _inputMode;

};
}

}




