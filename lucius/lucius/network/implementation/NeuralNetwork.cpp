/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The implementation of the Neural Network class
 */

// Lucius Includes
#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/CostFunction.h>
#include <lucius/network/interface/CostFunctionFactory.h>
#include <lucius/network/interface/Layer.h>
#include <lucius/network/interface/LayerFactory.h>

#include <lucius/optimizer/interface/NeuralNetworkSolver.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixOperations.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/PropertyTree.h>

// Standard Library Includes
#include <cassert>
#include <vector>
#include <ctime>

namespace lucius
{

namespace network
{

typedef optimizer::NeuralNetworkSolver NeuralNetworkSolver;
typedef NeuralNetwork::LayerPointer LayerPointer;

NeuralNetwork::NeuralNetwork()
: _costFunction(CostFunctionFactory::create()), _solver(NeuralNetworkSolver::create(this))
{

}

NeuralNetwork::~NeuralNetwork()
{

}

void NeuralNetwork::initialize()
{
    util::log("NeuralNetwork") << "Initializing neural network randomly.\n";

    for(auto& layer : *this)
    {
        layer->initialize();
    }
}

double NeuralNetwork::getCostAndGradient(MatrixVector& gradient, const Matrix& input,
    const Matrix& reference)
{
    MatrixVector activations;

    activations.push_back(input);

    size_t weightMatrices = 0;

    for(auto layer = begin(); layer != end(); ++layer)
    {
        util::log("NeuralNetwork") << " Running forward propagation through layer "
            << std::distance(begin(), layer) << "\n";

        (*layer)->runForward(activations);

        weightMatrices += (*layer)->weights().size();
    }

    gradient.resize(weightMatrices);

    auto costFunctionResult = getCostFunction()->computeCost(activations.back(), reference);

    auto activation = activations.rbegin();
    auto delta      = getCostFunction()->computeDelta(*activation, reference);

    auto gradientMatrix = gradient.rbegin();

    for(auto layer = rbegin(); layer != rend(); ++layer, ++activation)
    {
        MatrixVector grad;

        util::log("NeuralNetwork") << " Running reverse propagation through layer "
            << std::distance(begin(), std::next(layer).base()) << "\n";

        delta = (*layer)->runReverse(grad, activations, delta);

        for(auto gradMatrix = grad.rbegin(); gradMatrix != grad.rend();
            ++gradMatrix, ++gradientMatrix)
        {
            *gradientMatrix = std::move(*gradMatrix);
        }
    }

    auto weightCost = 0.0;

    for(auto& layer : *this)
    {
        weightCost += layer->computeWeightCost();
    }

    if(util::isLogEnabled("NeuralNetwork::Detail"))
    {
        util::log("NeuralNetwork::Detail") << "  cost function result: "
            << costFunctionResult.debugString();
    }
    else
    {
        util::log("NeuralNetwork") << "  cost function result shape: "
            << costFunctionResult.shapeString() << "\n";
    }


    return weightCost + reduce(costFunctionResult, {}, matrix::Add())[0];
}

double NeuralNetwork::getInputCostAndGradient(Matrix& gradient,
    const Matrix& input, const Matrix& reference)
{
    MatrixVector activations;

    activations.push_back(input);

    for(auto layer = begin(); layer != end(); ++layer)
    {
        util::log("NeuralNetwork") << " Running forward propagation through layer "
            << std::distance(begin(), layer) << "\n";

        (*layer)->runForward(activations);
    }

    auto costFunctionResult = getCostFunction()->computeCost(activations.back(), reference);

    auto activation = activations.rbegin();
    auto delta      = getCostFunction()->computeDelta(*activation, reference);

    for(auto layer = rbegin(); layer != rend(); ++layer, ++activation)
    {
        MatrixVector grad;

        delta = (*layer)->runReverse(grad, activations, delta);
    }

    auto samples = input.size().back();

    gradient = apply(delta, matrix::Multiply(1.0 / samples));

    auto weightCost = 0.0;

    for(auto& layer : *this)
    {
        weightCost += layer->computeWeightCost();
    }

    return weightCost + reduce(costFunctionResult, {}, matrix::Add())[0];
}

double NeuralNetwork::getCost(const Matrix& input, const Matrix& reference)
{
    auto result = runInputs(input);

    float weightCost = 0.0f;

    for(auto& layer : *this)
    {
        weightCost += layer->computeWeightCost();
    }

    return weightCost +
        reduce(getCostFunction()->computeCost(result, reference), {}, matrix::Add())[0];
}

NeuralNetwork::Matrix NeuralNetwork::runInputs(const Matrix& m)
{
    MatrixVector activations;

    activations.push_back(m);

    for (auto i = begin(); i != end(); ++i)
    {
        util::log("NeuralNetwork") << " Running forward propagation through layer "
            << std::distance(_layers.begin(), i) << "\n";

        (*i)->runForward(activations);

        auto output = activations.back();

        activations.clear();

        activations.push_back(output);
    }

    return activations.back();
}

void NeuralNetwork::addLayer(std::unique_ptr<Layer>&& l)
{
    _layers.push_back(std::move(l));
}

void NeuralNetwork::clear()
{
    _layers.clear();
}

NeuralNetwork::LayerPointer& NeuralNetwork::operator[](size_t index)
{
    return _layers[index];
}

const NeuralNetwork::LayerPointer& NeuralNetwork::operator[](size_t index) const
{
    return _layers[index];
}

LayerPointer& NeuralNetwork::back()
{
    return _layers.back();
}

const LayerPointer& NeuralNetwork::back() const
{
    return _layers.back();
}

LayerPointer& NeuralNetwork::front()
{
    return _layers.front();
}

const LayerPointer& NeuralNetwork::front() const
{
    return _layers.front();
}

size_t NeuralNetwork::size() const
{
    return _layers.size();
}

bool NeuralNetwork::empty() const
{
    return _layers.empty();
}

const matrix::Precision& NeuralNetwork::precision() const
{
    assert(!empty());

    return front()->precision();
}

NeuralNetwork::Dimension NeuralNetwork::getInputSize() const
{
    if(empty())
    {
        return Dimension();
    }

    return front()->getInputSize();
}

NeuralNetwork::Dimension NeuralNetwork::getOutputSize() const
{
    if(empty())
    {
        return Dimension();
    }

    return back()->getOutputSize();
}

size_t NeuralNetwork::getInputCount() const
{
    return getInputSize().product();
}

size_t NeuralNetwork::getOutputCount() const
{
    return getOutputSize().product();
}

size_t NeuralNetwork::totalNeurons() const
{
    size_t activations = 0;

    for(auto& layer : *this)
    {
        activations += layer->getOutputCount();
    }

    return activations;
}

size_t NeuralNetwork::totalConnections() const
{
    size_t weights = 0;

    for(auto& layer : *this)
    {
        weights += layer->totalConnections();
    }

    return weights;
}

size_t NeuralNetwork::getFloatingPointOperationCount() const
{
    size_t flops = 0;

    for(auto& layer : *this)
    {
        flops += layer->getFloatingPointOperationCount();
    }

    return flops;
}

void NeuralNetwork::train(const Matrix& input, const Matrix& reference)
{
    getSolver()->setInput(&input);
    getSolver()->setReference(&reference);
    getSolver()->setNetwork(this);

    getSolver()->solve();
}

NeuralNetwork::iterator NeuralNetwork::begin()
{
    return _layers.begin();
}

NeuralNetwork::const_iterator NeuralNetwork::begin() const
{
    return _layers.begin();
}

NeuralNetwork::iterator NeuralNetwork::end()
{
    return _layers.end();
}

NeuralNetwork::const_iterator NeuralNetwork::end() const
{
    return _layers.end();
}

NeuralNetwork::reverse_iterator NeuralNetwork::rbegin()
{
    return _layers.rbegin();
}

NeuralNetwork::const_reverse_iterator NeuralNetwork::rbegin() const
{
    return _layers.rbegin();
}

NeuralNetwork::reverse_iterator NeuralNetwork::rend()
{
    return _layers.rend();
}

NeuralNetwork::const_reverse_iterator NeuralNetwork::rend() const
{
    return _layers.rend();
}

void NeuralNetwork::setCostFunction(CostFunction* f)
{
    _costFunction.reset(f);
}

CostFunction* NeuralNetwork::getCostFunction()
{
    return _costFunction.get();
}

const CostFunction* NeuralNetwork::getCostFunction() const
{
    return _costFunction.get();
}

void NeuralNetwork::setSolver(NeuralNetworkSolver* s)
{
    _solver.reset(s);
}

NeuralNetwork::NeuralNetworkSolver* NeuralNetwork::getSolver()
{
    return _solver.get();
}

const NeuralNetwork::NeuralNetworkSolver* NeuralNetwork::getSolver() const
{
    return _solver.get();
}

void NeuralNetwork::save(util::OutputTarArchive& archive, util::PropertyTree& tree) const
{
    auto& layers = tree["layers"];

    layers["layer-count"] = size();

    size_t index = 0;

    for(auto& layer : *this)
    {
        auto& layerProperties = layers[index++];

        layerProperties["type"] = layer->getTypeName();

        layer->save(archive, layerProperties);
    }

    tree["cost-function"] = getCostFunction()->typeName();
}

void NeuralNetwork::load(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    auto& layers = properties["layers"];

    setCostFunction(CostFunctionFactory::create(properties["cost-function"]));

    size_t layerCount = layers.get<size_t>("layer-count");

    for(size_t i = 0; i < layerCount; ++i)
    {
        auto& layerProperties = layers[i];

        addLayer(LayerFactory::create(layerProperties["type"]));

        back()->load(archive, layerProperties);
    }

    util::log("NeuralNetwork") << "Loaded " << shapeString();
}

void NeuralNetwork::setIsTraining(bool isTraining)
{
    for(auto& layer : *this)
    {
        layer->setIsTraining(isTraining);
    }
}

std::string NeuralNetwork::shapeString() const
{
    std::stringstream stream;

    stream << "Neural Network [" << size() << " layers, " << getInputCount()
        << " inputs, " << getOutputCount() << " outputs ]\n";

    for(auto& layer : *this)
    {
        size_t index = &layer - &*begin();

        stream << " Layer " << index << ": " << layer->shapeString()
            << layer->resourceString() << "\n";
    }

    return stream.str();
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& n)
: _costFunction(n.getCostFunction()->clone()), _solver(n.getSolver()->clone())
{
    for(auto& layer : n)
    {
        addLayer(layer->clone());
    }
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& n)
{
    if(&n == this)
    {
        return *this;
    }

    clear();

    setCostFunction(n.getCostFunction()->clone());
    setSolver(n.getSolver()->clone());

    for(auto& layer : n)
    {
        addLayer(layer->clone());
    }

    return *this;
}

}

}

