/*  \file   ConvolutionalLayer.cpp
	\author Gregory Diamos
 	\date   Dec 24, 2014
 	\brief  The source for the ConvolutionalLayer class.
*/

// Minerva Includes
#include <minerva/network/interface/ConvolutionalLayer.h>

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

ConvolutionalLayer::ConvolutionalLayer()
: ConvolutionalLayer({}, {}, matrix::SinglePrecision())
{

}

ConvolutionalLayer::ConvolutionalLayer(const matrix::Dimension& size, const matrix::Dimension& stride, const matrix::Precision& precision)
: _parameters(new MatrixVector({Matrix(size, precision)})), 
  _weights((*_parameters)[0]), _stride(std::make_unique<matrix::Dimension>(stride))
{

}

ConvolutionalLayer::~ConvolutionalLayer()
{

}

ConvolutionalLayer::ConvolutionalLayer(const ConvolutionalLayer& l)
: _parameters(std::make_unique<MatrixVector>(*l._parameters)), _weights((*_parameters)[0]), _stride(std::make_unique<matrix::Dimension>(*l._stride)) 
{

}

ConvolutionalLayer& ConvolutionalLayer::operator=(const ConvolutionalLayer& l)
{
	if(&l == this)
	{
		return *this;
	}
	
	_parameters = std::move(std::make_unique<MatrixVector>(*l._parameters));
	_stride     = std::move(std::make_unique<matrix::Dimension>(*l._stride));
	
	return *this;
}

void ConvolutionalLayer::initialize()
{
	double e = util::KnobDatabase::getKnobValue("Layer::RandomInitializationEpsilon", 6);

	double epsilon = std::sqrt((e) / (getInputCount() + getOutputCount() + 1));
	
	matrix::rand(_weights);
	apply(_weights, _weights, matrix::Multiply(epsilon));
}

Matrix ConvolutionalLayer::runForward(const Matrix& m) const
{
	assertM(false, "Not implemented.");
}

Matrix ConvolutionalLayer::runReverse(MatrixVector& gradients,
	const Matrix& inputActivations,
	const Matrix& outputActivations,
	const Matrix& deltas) const
{
	assertM(false, "Not implemented.");
}

MatrixVector& ConvolutionalLayer::weights()
{
	return *_parameters;
}

const MatrixVector& ConvolutionalLayer::weights() const
{
	return *_parameters;
}

const matrix::Precision& ConvolutionalLayer::precision() const
{
	return _weights.precision();
}

double ConvolutionalLayer::computeWeightCost() const
{
	return getWeightCostFunction()->getCost(_weights);
}

size_t ConvolutionalLayer::getInputCount() const
{
	return _weights.size()[0] * _weights.size()[1] * _weights.size()[2];
}

size_t ConvolutionalLayer::getOutputCount() const
{
	return _weights.size()[3];
}

size_t ConvolutionalLayer::totalNeurons() const
{
	return getOutputCount();
}

size_t ConvolutionalLayer::totalConnections() const
{
	return _weights.elements();
}

size_t ConvolutionalLayer::getFloatingPointOperationCount() const
{
	return 2 * totalConnections();
}

void ConvolutionalLayer::save(util::TarArchive& archive) const
{
	assertM(false, "Not implemented.");
}

void ConvolutionalLayer::load(const util::TarArchive& archive, const std::string& name)
{
	assertM(false, "Not implemented.");
}

std::unique_ptr<Layer> ConvolutionalLayer::clone() const
{
	return std::make_unique<ConvolutionalLayer>(*this);
}

std::unique_ptr<Layer> ConvolutionalLayer::mirror() const
{
	return std::make_unique<ConvolutionalLayer>(matrix::Dimension({_weights.size()[0], _weights.size()[1], _weights.size()[3], _weights.size()[2]}),
		*_stride, precision());
}

std::string ConvolutionalLayer::getTypeName() const
{
	return "ConvolutionalLayer";
}

}

}





