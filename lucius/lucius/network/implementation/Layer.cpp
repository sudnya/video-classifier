/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The implementation of the layer class
 */

// Lucius Includes
#include <lucius/network/interface/Layer.h>

#include <lucius/network/interface/ActivationCostFunction.h>
#include <lucius/network/interface/ActivationFunction.h>
#include <lucius/network/interface/WeightCostFunction.h>

#include <lucius/network/interface/ActivationCostFunctionFactory.h>
#include <lucius/network/interface/ActivationFunctionFactory.h>
#include <lucius/network/interface/WeightCostFunctionFactory.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/util/interface/PropertyTree.h>
#include <lucius/util/interface/Units.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{

namespace network
{

Layer::Layer()
: _activationFunction(ActivationFunctionFactory::create()),
  _activationCostFunction(ActivationCostFunctionFactory::create()),
  _weightCostFunction(WeightCostFunctionFactory::create()),
  _isTraining(true),
  _shouldComputeDeltas(true)
{

}

Layer::~Layer()
{

}

Layer::Layer(const Layer& l)
: _activationFunction(l.getActivationFunction()->clone()),
  _activationCostFunction(l.getActivationCostFunction() == nullptr ?
    nullptr : l.getActivationCostFunction()->clone()),
  _weightCostFunction(l.getWeightCostFunction() == nullptr ?
    nullptr : l.getWeightCostFunction()->clone()),
  _isTraining(l.isTraining()),
  _shouldComputeDeltas(l.getShouldComputeDeltas())
{

}

Layer& Layer::operator=(const Layer& l)
{
    if(&l == this)
    {
        return *this;
    }

    setActivationFunction(l.getActivationFunction()->clone());
    setActivationCostFunction(l.getActivationCostFunction() == nullptr ?
        nullptr : l.getActivationCostFunction()->clone());
    setWeightCostFunction(l.getWeightCostFunction() == nullptr ?
        nullptr : l.getWeightCostFunction()->clone());
    setIsTraining(l.isTraining());
    setShouldComputeDeltas(l.getShouldComputeDeltas());

    return *this;
}

static bool compareSize(matrix::Dimension inputSize, matrix::Dimension expectedSize)
{
    size_t minimumSize = expectedSize.size() - 2;

    expectedSize.pop_back(2);

    while(inputSize.size() > minimumSize)
    {
        inputSize.pop_back();
    }

    return inputSize == expectedSize;
}

static bool compareCount(matrix::Dimension inputSize,
    matrix::Dimension expectedSize, size_t inputCount)
{
    size_t product = 1;

    for(size_t dimension = 0; dimension < inputSize.size(); ++dimension)
    {
        product *= inputSize[dimension];

        if(product == inputCount)
        {
            return true;
        }
    }

    return false;
}

static void removeTimeAndBatch(matrix::Dimension& dimension)
{
    // remove time and batch dimensions
    dimension.pop_back(2);

}

static matrix::Matrix reshapeActivations(const matrix::Matrix& m, matrix::Dimension expectedSize)
{
    auto newShape = expectedSize;

    removeTimeAndBatch(newShape);

    // add back batch
    newShape.push_back(m.size()[m.size().size() - 2]);

    // add back time
    newShape.push_back(m.size()[m.size().size() - 1]);

    return reshape(m, newShape);
}

void Layer::runForward(MatrixVector& activations)
{
    if(compareSize(activations.back().size(), getInputSize()))
    {
        return runForwardImplementation(activations);
    }

    if(compareCount(activations.back().size(), getInputSize(), getInputCount()))
    {
        activations.back() = reshapeActivations(activations.back(), getInputSize());

        return runForwardImplementation(activations);
    }

    throw std::runtime_error("Input activation matrix size " + activations.back().size().toString() +
        " is not compatible with layer, expecting " + getInputSize().toString());
}

matrix::Matrix Layer::runReverse(MatrixVector& gradients,
    MatrixVector& activations,
    const Matrix& deltas)
{
    assert(isTraining());

    if(compareCount(activations.back().size(), getOutputSize(), getOutputCount()))
    {
        activations.back() = reshapeActivations(activations.back(), getOutputSize());
    }

    return runReverseImplementation(gradients,
        activations,
        reshapeActivations(deltas, getOutputSize()));
}

size_t Layer::getParameterMemory() const
{
    size_t memory = 0;

    for(auto& weight : weights())
    {
        memory += precision().size() * weight.elements();
    }

    return memory;
}

void Layer::setActivationFunction(ActivationFunction* f)
{
    _activationFunction.reset(f);
}

ActivationFunction* Layer::getActivationFunction()
{
    return _activationFunction.get();
}

const ActivationFunction* Layer::getActivationFunction() const
{
    return _activationFunction.get();
}

void Layer::setActivationCostFunction(ActivationCostFunction* f)
{
    _activationCostFunction.reset(f);
}

ActivationCostFunction* Layer::getActivationCostFunction()
{
    return _activationCostFunction.get();
}

const ActivationCostFunction* Layer::getActivationCostFunction() const
{
    return _activationCostFunction.get();
}

void Layer::setWeightCostFunction(WeightCostFunction* f)
{
    _weightCostFunction.reset(f);
}

WeightCostFunction* Layer::getWeightCostFunction()
{
    return _weightCostFunction.get();
}

const WeightCostFunction* Layer::getWeightCostFunction() const
{
    return _weightCostFunction.get();
}

void Layer::setIsTraining(bool training)
{
    _isTraining = training;
}

bool Layer::isTraining() const
{
    return _isTraining;
}

void Layer::setShouldComputeDeltas(bool shouldComputeDeltas)
{
    _shouldComputeDeltas = shouldComputeDeltas;
}

bool Layer::getShouldComputeDeltas() const
{
    return _shouldComputeDeltas;
}

std::string Layer::shapeString() const
{
    std::stringstream stream;

    stream << "(" << getTypeName() << " type, "
        << getInputSize().toString()
        << " inputs, " << getOutputSize().toString() << " outputs)";

    return stream.str();
}

std::string Layer::resourceString() const
{
    std::stringstream stream;

    stream << "(" << util::flopsString(getFloatingPointOperationCount()) << ", "
        << util::byteString(getActivationMemory())
        << " activations, " << util::byteString(getParameterMemory()) << " parameters)";

    return stream.str();
}

void Layer::saveLayer(util::OutputTarArchive& archive, util::PropertyTree& properties) const
{
    properties["activation-function"] = getActivationFunction()->typeName();

    if(getActivationCostFunction() != nullptr)
    {
        properties["activation-cost-function"] = getActivationCostFunction()->typeName();
    }
    else
    {
        properties["activation-cost-function"] = "None";
    }

    properties["weight-cost-function"] = getWeightCostFunction()->typeName();
}

void Layer::loadLayer(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    setActivationFunction(ActivationFunctionFactory::create(
        properties["activation-function"]));
    setActivationCostFunction(ActivationCostFunctionFactory::create(
        properties["activation-cost-function"]));
    setWeightCostFunction(WeightCostFunctionFactory::create(
        properties["weight-cost-function"]));

    _isTraining = false;
}

}

}


