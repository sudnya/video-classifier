/*  \file   FeedForwardLayer.h
	\author Gregory Diamos
 	\date   Dec 24, 2014
 	\brief  The interface for the FeedForwardLayer class.
*/

#pragma once

// Minerva Includes
#include <minerva/network/interface/Layer.h>

namespace minerva
{
namespace network
{

/* \brief A neural network layer interface. */
class FeedForwardLayer : public Layer
{
public:
	FeedForwardLayer(size_t inputs, size_t outputs, const Precision&);
    virtual ~FeedForwardLayer();

public:
	FeedForwardLayer(const FeedForwardLayer& );
	FeedForwardLayer& operator=(const FeedForwardLayer&);

public:
    virtual void initialize();

public:
    virtual Matrix runForward(const Matrix& m) const;
    virtual Matrix runReverse(MatrixVector& gradients,
		const Matrix& inputActivations,
		const Matrix& outputActivations,
		const Matrix& deltas) const;

public:
    virtual       MatrixVector& weights();
    virtual const MatrixVector& weights() const;

public:
	virtual double computeWeightCost() const;

public:
    virtual size_t getInputCount()  const;
    virtual size_t getOutputCount() const;

public:
    virtual size_t totalNeurons()	  const;
    virtual size_t totalConnections() const;

public:
    virtual size_t getFloatingPointOperationCount() const;

public:
	virtual void save(util::TarArchive& archive) const;
	virtual void load(const util::TarArchive& archive, const std::string& name);

public:
	virtual std::unique_ptr<Layer> clone() const;
	virtual std::unique_ptr<Layer> mirror() const;

public:
	virtual std::string getTypeName() const;

private:
	MatrixVector _parameters;

private:
	Matrix& _weights;
	Matrix& _bias;

};

}

}


