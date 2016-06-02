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
#include <lucius/network/interface/Bundle.h>

#include <lucius/optimizer/interface/NeuralNetworkSolver.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixOperations.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Units.h>
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

Bundle NeuralNetwork::getCostAndGradient(const Bundle& input)
{
    auto bundle = input;

    size_t weightMatrices = 0;

    util::log("NeuralNetwork") << " Running forward propagation of input "
        << bundle["inputActivations"].get<MatrixVector>().front().shapeString() << "\n";

    for(auto layer = begin(); layer != end(); ++layer)
    {
        util::log("NeuralNetwork") << " Running forward propagation through layer "
            << std::distance(begin(), layer) << "\n";

        bundle["outputActivations"] = MatrixVector();

        (*layer)->runForward(bundle);

        bundle["inputActivations"] = bundle["outputActivations"];

        weightMatrices += (*layer)->weights().size();
    }

    MatrixVector gradient;

    gradient.resize(weightMatrices);

    if(!empty())
    {
        front()->setShouldComputeDeltas(false);
    }

    getCostFunction()->computeCost(bundle);
    getCostFunction()->computeDelta(bundle);

    auto costFunctionResult = bundle["costs"].get<Matrix>();

    util::log("NeuralNetwork") << " Running forward propagation of delta "
        << bundle["outputDeltas"].get<MatrixVector>().front().shapeString() << "\n";

    auto gradientMatrix = gradient.rbegin();

    for(auto layer = rbegin(); layer != rend(); ++layer)
    {
        bundle["gradients"]   = MatrixVector();
        bundle["inputDeltas"] = MatrixVector();

        util::log("NeuralNetwork") << " Running reverse propagation through layer "
            << std::distance(begin(), std::next(layer).base()) << "\n";

        (*layer)->runReverse(bundle);
        (*layer)->clearReversePropagationData();

        bundle["outputDeltas"] = bundle["inputDeltas"];

        auto gradients = bundle["gradients"].get<MatrixVector>();

        for(auto gradMatrix = gradients.rbegin(); gradMatrix != gradients.rend();
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

    bundle["gradients"] = gradient;
    bundle["cost"] = weightCost + reduce(costFunctionResult, {}, matrix::Add())[0];

    return bundle;
}

Bundle NeuralNetwork::getInputCostAndGradient(const Bundle& input)
{
    auto bundle = input;

    for(auto layer = begin(); layer != end(); ++layer)
    {
        util::log("NeuralNetwork") << " Running forward propagation through layer "
            << std::distance(begin(), layer) << "\n";

        bundle["outputActivations"] = MatrixVector();

        (*layer)->runForward(bundle);

        bundle["inputActivations"] = bundle["outputActivations"];
    }

    if(!empty())
    {
        front()->setShouldComputeDeltas(true);
    }

    getCostFunction()->computeCost(bundle);
    getCostFunction()->computeDelta(bundle);

    auto costFunctionResult = bundle["costs"].get<Matrix>();

    for(auto layer = rbegin(); layer != rend(); ++layer)
    {
        bundle["gradients"]   = MatrixVector();
        bundle["inputDeltas"] = MatrixVector();

        util::log("NeuralNetwork") << " Running reverse propagation through layer "
            << std::distance(begin(), std::next(layer).base()) << "\n";

        (*layer)->runReverse(bundle);
        (*layer)->clearReversePropagationData();

        bundle["outputDeltas"] = bundle["inputDeltas"];
    }

    auto delta = bundle["inputDeltas"].get<MatrixVector>().front();

    auto samples = delta.size()[delta.size().size() - 1] * delta.size()[delta.size().size() - 2];

    bundle["inputDeltas"] = apply(delta, matrix::Multiply(1.0 / samples));

    auto weightCost = 0.0;

    for(auto& layer : *this)
    {
        weightCost += layer->computeWeightCost();
    }

    bundle["cost"] = weightCost + reduce(costFunctionResult, {}, matrix::Add())[0];

    return bundle;
}

Bundle NeuralNetwork::getCost(const Bundle& input)
{
    auto bundle = runInputs(input);

    float weightCost = 0.0f;

    for(auto& layer : *this)
    {
        weightCost += layer->computeWeightCost();
    }

    getCostFunction()->computeCost(bundle);

    auto costFunctionResult = bundle["costs"].get<Matrix>();

    bundle["cost"] = weightCost +
        reduce(costFunctionResult, {}, matrix::Add())[0];

    return bundle;
}

Bundle NeuralNetwork::runInputs(const Bundle& input)
{
    auto bundle = input;

    for (auto layer = begin(); layer != end(); ++layer)
    {
        util::log("NeuralNetwork") << " Running forward propagation through layer "
            << std::distance(_layers.begin(), layer) << "\n";

        bundle["outputActivations"] = MatrixVector();

        (*layer)->runForward(bundle);
        (*layer)->clearReversePropagationData();

        bundle["inputActivations"] = bundle["outputActivations"];
    }

    return bundle;
}

NeuralNetwork::Matrix NeuralNetwork::runInputs(const Matrix& input)
{
    Bundle inputBundle({std::make_pair("inputActivations", MatrixVector({input}))});

    auto bundle = runInputs(inputBundle);

    return bundle["outputActivations"].get<MatrixVector>().front();
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

size_t NeuralNetwork::getParameterMemory() const
{
    size_t bytes = 0;

    for(auto& layer : *this)
    {
        bytes += layer->getParameterMemory();
    }

    return bytes;
}

size_t NeuralNetwork::getActivationMemory() const
{
    size_t bytes = 0;

    for(auto& layer : *this)
    {
        bytes += layer->getActivationMemory();
    }

    return bytes;
}

Bundle NeuralNetwork::train(const Bundle& bundle)
{
    auto result = bundle;

    getSolver()->setBundle(&result);
    getSolver()->setNetwork(this);

    result["cost"] = getSolver()->solve();

    return result;
}

double NeuralNetwork::train(const Matrix& input, const Matrix& output)
{
    Bundle bundle({std::make_pair("inputActivations", MatrixVector({input})),
        std::make_pair("referenceActivations", MatrixVector({output}))});

    auto result = train(bundle);

    return result["cost"].get<double>();
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

        stream << " Layer " << index << ": " << layer->shapeString() << " "
            << layer->resourceString() << "\n";
    }

    stream << "Performance Requirement (" << util::flopsString(getFloatingPointOperationCount())
        << ", " << util::byteString(getActivationMemory()) << " activations, "
        << util::byteString(getParameterMemory()) << " parameters)\n";

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

