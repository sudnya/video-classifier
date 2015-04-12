/*  \file   RecurrentLayer.cpp
	\author Gregory Diamos
 	\date   Dec 24, 2014
 	\brief  The implementation of the RecurrentLayer class.
*/

// Minerva Includes
#include <minerva/network/interface/RecurrentLayer.h>

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

namespace minerva
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::MatrixVector MatrixVector;

RecurrentLayer::RecurrentLayer()
: RecurrentLayer(0, matrix::SinglePrecision())
{

}

RecurrentLayer::RecurrentLayer(size_t size, const matrix::Precision& precision)
: _parameters(new MatrixVector({Matrix({size, size}, precision), Matrix({size}, precision), Matrix({size, size}, precision)})), 
 _forwardWeights((*_parameters)[0]), _bias((*_parameters)[1]), _recurrentWeights((*_parameters)[2])
{

}

RecurrentLayer::~RecurrentLayer()
{

}

RecurrentLayer::RecurrentLayer(const RecurrentLayer& l)
: _parameters(std::make_unique<MatrixVector>(*l._parameters)), 
 _forwardWeights((*_parameters)[0]), _bias((*_parameters)[1]), _recurrentWeights((*_parameters)[2])
{

}

RecurrentLayer& RecurrentLayer::operator=(const RecurrentLayer& l)
{
	if(&l == this)
	{
		return *this;
	}
	
	_parameters = std::move(std::make_unique<MatrixVector>(*l._parameters));
	
	return *this;
}

void RecurrentLayer::initialize()
{
	double e = util::KnobDatabase::getKnobValue("Layer::RandomInitializationEpsilon", 6);

	double epsilon = std::sqrt((e) / (getInputCount() + getOutputCount() + 1));
	
	// initialize the feed forward layer
	matrix::rand(_forwardWeights);
	apply(_forwardWeights, _forwardWeights, matrix::Multiply(epsilon));

	// assign bias to 0.0f
	apply(_bias, _bias, matrix::Fill(0.0f));
	
	// initialize the recurrent weights
	matrix::rand(_recurrentWeights);
	apply(_recurrentWeights, _recurrentWeights, matrix::Multiply(epsilon));
}

Matrix RecurrentLayer::runForward(const Matrix& m) const
{
	assertM(false, "Not impemented.");
}

Matrix RecurrentLayer::runReverse(MatrixVector& gradients,
	const Matrix& inputActivations,
	const Matrix& outputActivations,
	const Matrix& deltas) const
{
	assertM(false, "Not impemented.");
}

MatrixVector& RecurrentLayer::weights()
{
	return *_parameters;
}

const MatrixVector& RecurrentLayer::weights() const
{
	return *_parameters;
}

const matrix::Precision& RecurrentLayer::precision() const
{
	return _forwardWeights.precision();
}

double RecurrentLayer::computeWeightCost() const
{
	return getWeightCostFunction()->getCost(_forwardWeights) + getWeightCostFunction()->getCost(_recurrentWeights);
}

size_t RecurrentLayer::getInputCount() const
{
	return _forwardWeights.size()[1];
}

size_t RecurrentLayer::getOutputCount() const
{
	return getInputCount();
}

size_t RecurrentLayer::totalNeurons() const
{
	return 2 * getInputCount();
}

size_t RecurrentLayer::totalConnections() const
{
	return 2 * _forwardWeights.elements() + _bias.elements();
}

size_t RecurrentLayer::getFloatingPointOperationCount() const
{
	return 2 * totalConnections();
}

void RecurrentLayer::save(util::TarArchive& archive) const
{
	assertM(false, "Not implemented.");
}

void RecurrentLayer::load(const util::TarArchive& archive, const std::string& name)
{
	assertM(false, "Not implemented.");
}

std::unique_ptr<Layer> RecurrentLayer::clone() const
{
	return std::make_unique<RecurrentLayer>(*this);
}

std::unique_ptr<Layer> RecurrentLayer::mirror() const
{
	return clone();
}

std::string RecurrentLayer::getTypeName() const
{
	return "RecurrentLayer";
}

}

}




