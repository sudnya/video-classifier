/*  \file   FeedForwardLayer.h
	\author Gregory Diamos
 	\date   Dec 24, 2014
 	\brief  The implementation of the FeedForwardLayer class.
*/

// Minerva Includes
#include <minerva/network/interface/FeedForwardLayer.h>

#include <minerva/network/interface/ActivationFunction.h>
#include <minerva/network/interface/ActivationCostFunction.h>
#include <minerva/network/interface/WeightCostFunction.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixVector.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/knobs.h>

namespace minerva
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::MatrixVector MatrixVector;

FeedForwardLayer::FeedForwardLayer(size_t inputs, size_t outputs)
: _parameters(std::make_unique<MatrixVector>({Matrix(Dimension(inputs, outputs)), Matrix(Dimension(outputs))})), 
 _weights(*_parameters[0]), _bias(*_parameters[1])
{

}

FeedForwardLayer::~FeedForwardLayer()
{

}

void FeedForwardLayer::initialize()
{
	double e = util::KnobDatabase::getKnobValue("Layer::RandomInitializationEpsilon", 6);

	double epsilon = std::sqrt((e) / (getInputCount() + getOutputCount() + 1));
	
	rand(_weights);
	apply(_weights, _weights, Multiply(epsilon));

	// assign bias to 0.0f
	apply(_bias, Fill(0.0f));
}

Matrix FeedForwardLayer::runForward(const Matrix& m) const
{
	if(util::isLogEnabled("FeedForwardLayer"))
	{
		util::log("FeedForwardLayer") << " Running forward propagation through layer: " << _weights.shapeString() << "\n";
	}

	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  input: " << m.debugString();
		util::log("FeedForwardLayer::Detail") << "  layer: " << _weights.debugString();
		util::log("FeedForwardLayer::Detail") << "  bias:  " << _bias.debugString();
	}

	auto unbiasedOutput = gemm(1.0, _weights, false, 1.0, m, false);

	auto output = broadcast(unbiasedOutput, bias, {1}, Sum());

	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  output: " << output.debugString();
	}
	else
	{
		util::log("FeedForwardLayer") << "  output: " << output.shapeString() << "\n";
	}
	
	auto activation = getActivationFunction()->apply(output);
	
	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  activation: " << activation.debugString();
	}
	else
	{
		util::log("FeedForwardLayer") << "  activation: " << activation.shapeString() << "\n";
	}

	return activation;
}

Matrix FeedForwardLayer::runReverse(MatrixVector& gradients,
	const Matrix& inputActivations,
	const Matrix& outputActivations,
	const Matrix& difference) const
{
	if(util::isLogEnabled("FeedForwardLayer"))
	{
		util::log("FeedForwardLayer") << " Running reverse propagation on matrix (" << difference.rows()
			<< " rows, " << difference.columns() << " columns) through layer with dimensions ("
			<< getBlocks() << " blocks, "
			<< getInputCount() << " inputs, " << getOutputCount()
			<< " outputs, " << _blockStep << " block step).\n";
		util::log("FeedForwardLayer") << "  layer: " << _weights.shapeString() << "\n";
  	}
	
	if(util::isLogEnabled("FeedForwardLayer"))
	{
		util::log("FeedForwardLayer") << "  input: " << difference.shapeString() << "\n";
	}

	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  input: " << difference.debugString();
	}

	// finish computing the deltas
	auto deltas = apply(getActivationFunction()->applyDerivative(outputActivations), difference, Multiply());
	
	// compute gradient for the weights
	auto unnormalizedWeightGradient = gemm(0.0, deltas, inputActivations, true, 1.0 / samples);

	auto samples = outputActivations.size()[1];
	auto weightGradient = apply(unnormalizedWeightGradient, Divide(samples));
	
	// add in the weight cost function term
	if(getWeightCostFunction() != nullptr)
	{
		weightGradient = apply(weightGradient, getWeightCostFunction()->getGradient(_weights), Add());
	}
	
	assert(weightGradient.blocks() == _weights.blocks());

	gradients.push_back(std::move(weightGradient));
	
	// compute gradient for the bias
	auto biasGradient = reduce(apply(deltas, Divide(samples)), {1}, Plus());

	if(util::isLogEnabled("FeedForwardLayer"))
	{
		util::log("FeedForwardLayer") << "  bias grad: " << biasGradient.shapeString() << "\n";
	}

	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  bias grad: " << biasGradient.debugString();
	}
	
	assert(biasGradient.blocks() == _bias.blocks());
	assert(biasGradient.columns() == _bias.columns());
	
	gradients.push_back(std::move(biasGradient));
	
	// compute deltas for previous layer
	auto deltasPropagatedReverse = gemm(_weights, true, deltas, false);
	
	Matrix previousLayerDeltas;
	
	if(getActivationCostFunction() != nullptr)
	{
		auto activationCostFunctionGradient = getActivationCostFunction()->getGradient(outputActivations);
		
		previousLayerDeltas = apply(deltasPropagatedReverse, activationCostFunctionGradient, Multiply());
	}
	else
	{
		previousLayerDeltas = deltasPropagatedReverse;
	}
	
	if(util::isLogEnabled("FeedForwardLayer"))
	{
		util::log("FeedForwardLayer") << "  output: " << previousLayerDeltas.shapeString() << "\n";
	}

	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  output: " << previousLayerDeltas.debugString();
	}

	return previousLayerDeltas;
}

MatrixVector& FeedForwardLayer::weights()
{
	return _parameters;
}

const MatrixVector& FeedForwardLayer::weights() const
{
	return _parameters;
}

float FeedForwardLayer::computeWeightCost() const
{
	return getWeightCostFunction()->getCost(_weights);
}

size_t FeedForwardLayer::getInputCount() const
{
	return _weights.rows();
}

size_t FeedForwardLayer::getOutputCount() const
{
	return _weights.columns();
}

size_t FeedForwardLayer::totalNeurons()	const
{
	return getOutputCount();
}

size_t FeedForwardLayer::totalConnections() const
{
	return _weights.size() + _bias.size();
}

size_t FeedForwardLayer::getFloatingPointOperationCount() const
{
	return totalConnections();
}

void FeedForwardLayer::save(util::TarArchive& archive) const
{
	assertM(false, "Not implemented");
}

void FeedForwardLayer::load(const util::TarArchive& archive, const std::string& name)
{
	assertM(false, "Not implemented");
}

Layer* FeedForwardLayer::clone() const
{
	return new FeedForwardLayer(*this);
}

Layer* FeedForwardLayer::mirror() const
{
	return new FeedForwardLayer(getBlocks(), getOutputBlockingFactor(), getInputBlockingFactor());
}

std::string FeedForwardLayer::getTypeName() const
{
	return "FeedForwardLayer";
}

FeedForwardLayer::FeedForwardLayer(const FeedForwardLayer& l)
: _parameters(l._parameters), _weights(_parameters[0]), _bias(_parameters[1]), _blockStep(l._blockStep)
{

}

FeedForwardLayer& FeedForwardLayer::operator=(const FeedForwardLayer& l)
{
	if(&l == this)
	{
		return *this;
	}
	
	_parameters = l._parameters;
	_blockStep = l._blockStep;
	
	return *this;
}

}

}


