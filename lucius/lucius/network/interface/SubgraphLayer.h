/*  \file   SubgraphLayer.h
    \author Gregory Diamos
    \date   January 20, 2016
    \brief  The interface file for the SubgraphLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

// Forward Declarations
namespace lucius { namespace network { class SubgraphLayerImplementation; } }

namespace lucius
{
namespace network
{

/* \brief A layer containing an arbitrary subgraph of other layers. */
class SubgraphLayer : public Layer
{
public:
    SubgraphLayer();
    virtual ~SubgraphLayer();

public:
    SubgraphLayer(const SubgraphLayer& );
    SubgraphLayer& operator=(const SubgraphLayer&);

public:
    virtual void initialize();

public:
    virtual void runForwardImplementation(MatrixVector& outputActivations,
        const MatrixVector& inputActivations);
    virtual void runReverseImplementation(MatrixVector& gradients,
        MatrixVector& inputDeltas,
        const MatrixVector& outputDeltas);

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
    virtual size_t totalNeurons()     const;
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
    void addLayer(const std::string& layerName, std::unique_ptr<Layer>&& layer);

    void addForwardConnection(const std::string& source, const std::string& destination);
    void addTimeConnection(const std::string& source, const std::string& destination);

    void prepareSubgraphForEvaluation();

private:
    std::unique_ptr<SubgraphLayerImplementation> _implementation;
};

}
}

