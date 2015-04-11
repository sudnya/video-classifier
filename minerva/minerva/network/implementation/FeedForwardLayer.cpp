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
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/BlasOperations.h>
#include <minerva/matrix/interface/RandomOperations.h>
#include <minerva/matrix/interface/Operation.h>
#include <minerva/matrix/interface/MatrixVector.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/knobs.h>
#include <minerva/util/interface/memory.h>

// Standard Library Includes
#include <memory>

namespace minerva
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::MatrixVector MatrixVector;

FeedForwardLayer::FeedForwardLayer(size_t inputs, size_t outputs, const matrix::Precision& precision)
: _parameters(new MatrixVector({Matrix({inputs, outputs}, precision), Matrix({outputs}, precision)})), 
 _weights((*_parameters)[0]), _bias((*_parameters)[1])
{

}

FeedForwardLayer::~FeedForwardLayer()
{

}

void FeedForwardLayer::initialize()
{
	double e = util::KnobDatabase::getKnobValue("Layer::RandomInitializationEpsilon", 6);

	double epsilon = std::sqrt((e) / (getInputCount() + getOutputCount() + 1));
	
	matrix::rand(_weights);
	apply(_weights, _weights, matrix::Multiply(epsilon));

	// assign bias to 0.0f
	apply(_bias, _bias, matrix::Fill(0.0f));
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

	auto output = broadcast(unbiasedOutput, _bias, matrix::Add());

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
		util::log("FeedForwardLayer") << " Running reverse propagation on matrix (" << difference.size()[0]
			<< " rows, " << difference.size()[1] << " columns) through layer with dimensions ("
			<< getInputCount() << " inputs, " << getOutputCount() << " outputs).\n";
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
	auto deltas = apply(getActivationFunction()->applyDerivative(outputActivations), difference, matrix::Multiply());
	
	// compute gradient for the weights
	auto samples = outputActivations.size()[1];

	auto unnormalizedWeightGradient = gemm(0.0, deltas, false, 1.0 / samples, inputActivations, true);

	auto weightGradient = apply(unnormalizedWeightGradient, matrix::Divide(samples));
	
	// add in the weight cost function term
	if(getWeightCostFunction() != nullptr)
	{
		apply(weightGradient, weightGradient, getWeightCostFunction()->getGradient(_weights), matrix::Add());
	}
	
	gradients.push_back(std::move(weightGradient));
	
	// compute gradient for the bias
	auto biasGradient = reduce(apply(deltas, matrix::Divide(samples)), {1}, matrix::Add());

	if(util::isLogEnabled("FeedForwardLayer"))
	{
		util::log("FeedForwardLayer") << "  bias grad: " << biasGradient.shapeString() << "\n";
	}

	if(util::isLogEnabled("FeedForwardLayer::Detail"))
	{
		util::log("FeedForwardLayer::Detail") << "  bias grad: " << biasGradient.debugString();
	}
	
	assert(biasGradient.size() == _bias.size());
	
	gradients.push_back(std::move(biasGradient));
	
	// compute deltas for previous layer
	auto deltasPropagatedReverse = gemm(_weights, true, deltas, false);
	
	Matrix previousLayerDeltas;
	
	if(getActivationCostFunction() != nullptr)
	{
		auto activationCostFunctionGradient = getActivationCostFunction()->getGradient(outputActivations);
		
		apply(previousLayerDeltas, deltasPropagatedReverse, activationCostFunctionGradient, matrix::Multiply());
	}
	else
	{
		previousLayerDeltas = std::move(deltasPropagatedReverse);
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
	return *_parameters;
}

const MatrixVector& FeedForwardLayer::weights() const
{
	return *_parameters;
}

double FeedForwardLayer::computeWeightCost() const
{
	return getWeightCostFunction()->getCost(_weights);
}

size_t FeedForwardLayer::getInputCount() const
{
	return _weights.size()[1];
}

size_t FeedForwardLayer::getOutputCount() const
{
	return _weights.size()[0];
}

size_t FeedForwardLayer::totalNeurons()	const
{
	return getOutputCount();
}

size_t FeedForwardLayer::totalConnections() const
{
	return _weights.elements() + _bias.elements();
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

std::unique_ptr<Layer> FeedForwardLayer::clone() const
{
	return std::make_unique<FeedForwardLayer>(*this);
}

std::unique_ptr<Layer> FeedForwardLayer::mirror() const
{
	return std::make_unique<FeedForwardLayer>(getInputCount(), getOutputCount(), precision());
}

std::string FeedForwardLayer::getTypeName() const
{
	return "FeedForwardLayer";
}

FeedForwardLayer::FeedForwardLayer(const FeedForwardLayer& l)
: _parameters(std::make_unique<MatrixVector>(*l._parameters)), _weights((*_parameters)[0]), _bias((*_parameters)[1])
{

}

FeedForwardLayer& FeedForwardLayer::operator=(const FeedForwardLayer& l)
{
	if(&l == this)
	{
		return *this;
	}
	
	_parameters = std::move(std::make_unique<MatrixVector>(*l._parameters));
	
	return *this;
}

}

}


