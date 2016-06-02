/*  \file   SubgraphLayer.h
    \author Gregory Diamos
    \date   January 20, 2016
    \brief  The interface file for the SubgraphLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/SubgraphLayer.h>

#include <lucius/network/interface/LayerFactory.h>
#include <lucius/network/interface/Bundle.h>

#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/Dimension.h>

#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/PropertyTree.h>

// Standard Library Includes
#include <list>

namespace lucius
{
namespace network
{

typedef matrix::MatrixVector MatrixVector;
typedef matrix::Matrix       Matrix;
typedef matrix::Dimension    Dimension;

class SubgraphLayerImplementation
{
public:

    class Node
    {
    public:
        typedef std::list<Node> NodeList;
        typedef std::list<NodeList::iterator> NodePointerList;

    public:
        Node(const std::string& name, std::unique_ptr<Layer>&& layer,
            bool isFirstLayer = false, bool isLastLayer = false)
        : name(name), layer(std::move(layer)), isFirstLayer(isFirstLayer), isLastLayer(isLastLayer)
        {

        }

    public:
        size_t getInputActivationCount() const
        {
            size_t total = layer->getInputCount();

            size_t split = totalPredecessors();

            size_t splitSize = (total + split - 1) / split;

            return splitSize;
        }

        size_t getOutputActivationCount() const
        {
            size_t total = layer->getOutputCount();

            size_t split = totalSuccessors();

            size_t splitSize = (total + split - 1) / split;

            return splitSize;
        }

        size_t totalPredecessors() const
        {
            size_t predecessors = forwardPredecessors.size() + timePredecessors.size();

            if(isFirstLayer)
            {
                predecessors += 1;
            }

            return predecessors;
        }

        size_t totalSuccessors() const
        {
            size_t successors = forwardSuccessors.size() + timeSuccessors.size();

            if(isLastLayer)
            {
                successors += 1;
            }

            return successors;
        }

    public:
        std::string name;
        std::unique_ptr<Layer> layer;

    public:
        NodePointerList forwardSuccessors;
        NodePointerList forwardPredecessors;

    public:
        NodePointerList timeSuccessors;
        NodePointerList timePredecessors;

        bool isFirstLayer;
        bool isLastLayer;

    };

    typedef Node::NodeList NodeList;
    typedef Node::NodePointerList NodePointerList;
    typedef std::map<std::string, NodeList::iterator> NameToLayerMap;
    typedef std::set<std::pair<std::string, std::string>> ConnectionSet;
    typedef std::set<std::string> NodeSet;
    typedef std::map<std::string, MatrixVector> StringToMatrixMap;

public:
    SubgraphLayerImplementation() = default;

    SubgraphLayerImplementation(const SubgraphLayerImplementation& i)
    {
        _copy(i);
    }

    SubgraphLayerImplementation& operator=(const SubgraphLayerImplementation& i)
    {
        if(this == &i)
        {
            return *this;
        }

        clear();

        _copy(i);

        return *this;
    }

public:
    NodeList       layers;
    NameToLayerMap layerNames;

public:
    MatrixVector weights;

public:
    const Layer& front() const
    {
        return *layers.front().layer;
    }

    const Layer& back() const
    {
        return *layers.back().layer;
    }

public:
    Dimension getInputSize() const
    {
        return front().getInputSize();
    }

    Dimension getOutputSize() const
    {
        auto backSize = back().getOutputSize();

        size_t splits = layers.back().timeSuccessors.size() + 1;

        size_t splitSize = (backSize[0] + splits - 1) / splits;

        backSize[0] = splitSize;

        return backSize;
    }

public:
    void prepareSubgraphForEvaluation()
    {
        // sort layers in topological order according to forward connections
        auto frontier = _findReadyLayers();

        NodeSet visited;

        NodePointerList newOrder;

        while(!frontier.empty())
        {
            auto nextLayer = frontier.front();
            frontier.pop_front();

            visited.insert(nextLayer->name);

            newOrder.push_back(nextLayer);

            // queue up ready successors
            for(auto& successor : nextLayer->forwardSuccessors)
            {
                if(_allPredecessorsVisited(successor, visited))
                {
                    frontier.push_back(successor);
                }
            }
        }

        // if not all nodes were covered, there were cycles
        assert(newOrder.size() == layers.size());

        // Create the new layer list and name map
        typedef std::vector<std::pair<std::string, std::string>> ConnectionList;

        ConnectionList forwardConnections;
        ConnectionList timeConnections;

        for(auto& layer : newOrder)
        {
            for(auto& successor : layer->timeSuccessors)
            {
                timeConnections.push_back({layer->name, successor->name});
            }

            for(auto& successor : layer->forwardSuccessors)
            {
                forwardConnections.push_back({layer->name, successor->name});
            }
        }

        NodeList       newLayerList;
        NameToLayerMap newLayerNames;

        weights.clear();

        for(auto& layer : newOrder)
        {
            layer->forwardSuccessors.clear();
            layer->forwardPredecessors.clear();

            layer->timeSuccessors.clear();
            layer->timePredecessors.clear();

            layer->isFirstLayer = false;
            layer->isLastLayer = false;

            auto position = newLayerList.insert(newLayerList.end(), std::move(*layer));

            newLayerNames[position->name] = position;
            weights.push_back(position->layer->weights());
        }

        layers = std::move(newLayerList);
        layerNames = std::move(newLayerNames);

        for(auto& connection : forwardConnections)
        {
            addForwardConnection(connection.first, connection.second);
        }

        for(auto& connection : timeConnections)
        {
            addTimeConnection(connection.first, connection.second);
        }

        layers.front().isFirstLayer = true;
        layers.back().isLastLayer = true;
    }

public:
    void runForwardAllTimesteps(Bundle& bundle)
    {
        if(layers.empty())
        {
            return;
        }

        auto& inputActivations  = bundle[ "inputActivations"].get<MatrixVector>();
        auto& outputActivations = bundle["outputActivations"].get<MatrixVector>();

        // Set the input activations for the first layer
        std::map<std::string, MatrixVector> layerToInputActivations;

        layerToInputActivations[layers.front().name] = inputActivations;

        // Evaluate the layers in order
        for(auto& layer : layers)
        {
            auto& name = layer.name;

            // get the inputs
            auto& inputs = layerToInputActivations[name];

            _formatInputActivationsForLayer(inputs, layer);

            // run the layer
            Bundle localBundle;

            localBundle["inputActivations"] = inputs;
            MatrixVector& localOutputs = localBundle["outputActivations"].get<MatrixVector>();

            util::log("SubgraphLayer") << "Running forward propagation on layer '"
                << name << "'\n";

            layer.layer->runForward(localBundle);

            inputs.clear();

            _formatOutputActivationsForLayer(localOutputs, layer);

            // initial outputs go to successor outputs
            size_t index = 0;
            for(auto& successor : layer.forwardSuccessors)
            {
                layerToInputActivations[successor->name].push_back(std::move(localOutputs[index]));

                ++index;
            }

            // remaining outputs go to the sublayer outputs
            for(; index < localOutputs.size(); ++index)
            {
                outputActivations.push_back(std::move(localOutputs[index]));
            }
        }
    }

    void runReverseAllTimesteps(Bundle& bundle)
    {
        auto& gradients    = bundle[   "gradients"].get<MatrixVector>();
        auto& inputDeltas  = bundle[ "inputDeltas"].get<MatrixVector>();
        auto& outputDeltas = bundle["outputDeltas"].get<MatrixVector>();

        if(layers.empty())
        {
            return;
        }

        // Set the output deltas for the last layer
        std::map<std::string, MatrixVector> layerToOutputDeltas;

        layerToOutputDeltas[layers.back().name] = outputDeltas;

        MatrixVector reverseGradients;

        // Evaluate the layers in reverse order
        for(auto node = layers.rbegin(); node != layers.rend(); ++node)
        {
            auto& name  = node->name;
            auto& layer = *node->layer;

            // get the inputs
            auto& deltas = layerToOutputDeltas[name];

            _formatOutputDeltasForLayer(deltas, *node);

            // run the layer
            Bundle localBundle;

            localBundle["gradients"]    = MatrixVector();
            localBundle["inputDeltas"]  = MatrixVector();
            localBundle["outputDeltas"] = deltas;

            MatrixVector& localInputDeltas = localBundle["inputDeltas"].get<MatrixVector>();
            MatrixVector& localGradients   = localBundle[  "gradients"].get<MatrixVector>();

            util::log("SubgraphLayer") << "Running reverse propagation on layer '"
                << name << "'\n";
            layer.runReverse(bundle);

            _formatInputDeltasForLayer(localInputDeltas, *node);

            deltas.clear();

            size_t index = 0;
            for(auto& predecessor : node->forwardPredecessors)
            {
                layerToOutputDeltas[predecessor->name].push_back(
                    std::move(localInputDeltas[index]));

                ++index;
            }

            // remaining outputs go to the sublayer outputs
            for(; index < localInputDeltas.size(); ++index)
            {
                inputDeltas.push_back(std::move(localInputDeltas[index]));
            }

            std::reverse(localGradients.begin(), localGradients.end());
            reverseGradients.push_back(std::move(localGradients));
        }

        std::reverse(reverseGradients.begin(), reverseGradients.end());

        gradients.push_back(reverseGradients);
    }

public:
    void runForwardIterateTimesteps(Bundle& bundle)
    {
        if(layers.empty())
        {
            return;
        }

        auto& inputActivations  = bundle[ "inputActivations"].get<MatrixVector>();
        auto& outputActivations = bundle["outputActivations"].get<MatrixVector>();

        // Get the total number of timesteps
        size_t timesteps = inputActivations.front().size().back();

        StringToMatrixMap layerToInputActivations;

        util::log("SubgraphLayer") << "Running forward propagation through "
            << timesteps << " timesteps.\n";

        // Step through each timestep
        for(size_t timestep = 0; timestep < timesteps; ++timestep)
        {
            _updateSublayerInputActivationsForTimestep(layerToInputActivations,
                inputActivations, timestep);

            // Evaluate layers for this timestep in order
            for(auto& node : layers)
            {
                auto& name  = node.name;
                auto& layer = *node.layer;

                // get the inputs
                auto& inputs = layerToInputActivations[name];

                _formatInputActivationsForLayer(inputs, node);

                // run the layer
                Bundle localBundle;

                localBundle["inputActivations"] = inputs;

                MatrixVector& localOutputs = localBundle["outputActivations"].get<MatrixVector>();

                util::log("SubgraphLayer") << " Running forward propagation on layer '"
                    << name << "' on timestep " << timestep << "\n";

                layer.runForward(localBundle);

                inputs.clear();

                _routeOutputActivationsForLayerAndTimestep(layerToInputActivations,
                    outputActivations, localOutputs, node, timestep);
            }
        }
    }
    void runReverseIterateTimesteps(Bundle& bundle)
    {
        auto& gradients    = bundle[   "gradients"].get<MatrixVector>();
        auto& inputDeltas  = bundle[ "inputDeltas"].get<MatrixVector>();
        auto& outputDeltas = bundle["outputDeltas"].get<MatrixVector>();

        if(layers.empty())
        {
            return;
        }

        // Set the output deltas for the last layer
        std::map<std::string, MatrixVector> layerToOutputDeltas;

        // Get the total number of timesteps
        size_t timesteps = outputDeltas.front().size().back();

        MatrixVector sublayerGradients;

        // step through each timestep
        for(size_t t = 0; t < timesteps; ++t)
        {
            size_t timestep = timesteps - t - 1;

            _updateSublayerOutputDeltasForTimestep(layerToOutputDeltas,
                outputDeltas, timestep);

            if(t == 0)
            {
                _zeroPadTimeConnections(layerToOutputDeltas,
                    outputDeltas.front().size()[outputDeltas.front().size().size()-2],
                    outputDeltas.front().precision());
            }

            Bundle localBundle;

            localBundle["gradients"] = MatrixVector();

            auto& localGradients = localBundle["gradients"].get<MatrixVector>();

            // Evaluate the layers in reverse order
            for(auto node = layers.rbegin(); node != layers.rend(); ++node)
            {
                auto& name  = node->name;
                auto& layer = *node->layer;

                // get the output deltas
                auto& deltas = layerToOutputDeltas[name];

                _formatOutputDeltasForLayer(deltas, *node);

                // run the layer
                localBundle["outputDeltas"] = deltas;
                localBundle["inputDeltas"] = MatrixVector();
                auto& localInputDeltas = localBundle["inputDeltas"].get<MatrixVector>();

                util::log("SubgraphLayer") << " Running reverse propagation on layer '"
                    << name << "' on timestep " << timestep << "\n";

                layer.runReverse(localBundle);

                layer.popReversePropagationData();

                deltas.clear();

                _routeInputDeltasForLayerAndTimestep(layerToOutputDeltas,
                    inputDeltas, localInputDeltas, *node, t);
            }

            // accumulate into the gradients
            if(t == 0)
            {
                sublayerGradients.push_back(std::move(localGradients));
            }
            else
            {
                for(size_t i = 0; i < localGradients.size(); ++i)
                {
                    apply(sublayerGradients[i], sublayerGradients[i],
                        localGradients[i], matrix::Add());
                }
            }
        }

        gradients.push_back(sublayerGradients);
    }

public:
    void addLayer(const std::string& layerName, std::unique_ptr<Layer>&& layer)
    {
        assert(layerNames.count(layerName) == 0);

        if(!layers.empty())
        {
            layers.back().isLastLayer = false;
        }

        auto layerIterator = layers.insert(layers.end(), Node(layerName, std::move(layer)));

        if(layers.size() == 1)
        {
            layers.front().isFirstLayer = true;
        }

        layers.back().isLastLayer = true;

        layerNames.insert(std::make_pair(layerName, layerIterator));

        weights.push_back(layerIterator->layer->weights());
    }

    void addForwardConnection(const std::string& node, const std::string& successor)
    {
        auto currentNode   = layerNames.find(node);
        auto successorNode = layerNames.find(successor);

        assert(currentNode   != layerNames.end());
        assert(successorNode != layerNames.end());

        currentNode->second->forwardSuccessors.push_back(successorNode->second);
        successorNode->second->forwardPredecessors.push_back(currentNode->second);
    }

    void addTimeConnection(const std::string& node, const std::string& successor)
    {
        auto currentNode   = layerNames.find(node);
        auto successorNode = layerNames.find(successor);

        assert(currentNode   != layerNames.end());
        assert(successorNode != layerNames.end());

        currentNode->second->timeSuccessors.push_back(successorNode->second);
        successorNode->second->timePredecessors.push_back(currentNode->second);
    }

public:
    bool doAnyTimeConnectionsExist() const
    {
        for(auto& layer : layers)
        {
            if(!layer.timeSuccessors.empty())
            {
                return true;
            }
        }

        return false;
    }

public:
    std::unique_ptr<Layer>& getLayer(const std::string& name)
    {
        auto layer = layerNames.find(name);

        assert(layer != layerNames.end());

        return layer->second->layer;
    }

public:
    void clear()
    {
        layers.clear();
        layerNames.clear();

        weights.clear();
    }

private:
    Matrix _collapseActivations(const Matrix& m)
    {
        size_t miniBatch   = m.size()[m.size().size() - 2];
        size_t timesteps   = m.size()[m.size().size() - 1];
        size_t activations = m.size().product() / (miniBatch * timesteps);

        return reshape(m, {activations, miniBatch, timesteps});
    }

    void _formatInputActivationsForLayer(MatrixVector& inputActivations, const Node& node)
    {
        if(node.layer->getSupportsMultipleInputsAndOutputs())
        {
            return;
        }

        if(node.totalPredecessors() < 2)
        {
            return;
        }

        auto result = _collapseActivations(inputActivations.front());

        for(size_t i = 1; i < inputActivations.size(); ++i)
        {
            result = concatenate(result, _collapseActivations(inputActivations[i]), 0);
        }

        inputActivations.clear();

        inputActivations.push_back(result);
    }

    void _formatOutputActivationsForLayer(MatrixVector& outputActivations, const Node& node)
    {
        if(node.layer->getSupportsMultipleInputsAndOutputs())
        {
            return;
        }

        if(node.forwardSuccessors.size() < 2)
        {
            return;
        }

        assert(outputActivations.size() == 1);

        auto outputData = _collapseActivations(outputActivations.front());

        // format output data according to successors
        MatrixVector newOutputActivations;

        size_t currentActivation = 0;

        for(auto& successor : node.forwardSuccessors)
        {
            size_t nextActivation = currentActivation + successor->getInputActivationCount();

            newOutputActivations.push_back(slice(outputData, {currentActivation, 0, 0},
                {nextActivation, outputData.size()[1], outputData.size()[2]}));

            currentActivation = nextActivation;
        }

        outputActivations = std::move(newOutputActivations);
    }

private:
    Matrix _extractTimestep(const Matrix& matrix, size_t timestep)
    {
        auto begin = zeros(matrix.size());
        auto end   = matrix.size();

        begin.back() = timestep;
        end.back()   = timestep + 1;

        return slice(matrix, begin, end);
    }

    void _updateSublayerInputActivationsForTimestep(StringToMatrixMap& layerToInputMap,
        const MatrixVector& inputActivations, size_t timestep)
    {
        auto& firstLayerName = layers.front().name;

        assert(inputActivations.size() == 1);

        auto& firstLayerInputs = layerToInputMap[firstLayerName];

        auto inputsForTimestep = _extractTimestep(inputActivations.front(), timestep);

        if(firstLayerInputs.empty())
        {
            firstLayerInputs.push_back(std::move(inputsForTimestep));
        }
        else
        {
            size_t miniBatch = inputsForTimestep.size()[inputsForTimestep.size().size() - 2];

            assert(!layers.front().timePredecessors.empty());

            size_t splits = layers.front().timePredecessors.size() + 1;

            size_t splitSize = (layers.front().layer->getInputCount() + splits - 1) / splits;

            // handle forward inputs
            auto inputSlice = slice(inputsForTimestep,
                {0, 0, 0}, {splitSize, miniBatch, 1});

            MatrixVector newFirstLayerInputs;

            newFirstLayerInputs.push_back(inputSlice);

            // handle time inputs
            for(size_t predecessorId = 0;
                (predecessorId < layers.front().timePredecessors.size()) &&
                (predecessorId < firstLayerInputs.size()); ++predecessorId)
            {
                auto existingInput = firstLayerInputs[predecessorId];

                size_t beginActivation = splitSize * predecessorId;
                size_t endActivation   = std::min(beginActivation + splitSize,
                    layers.front().layer->getInputCount());

                auto inputSlice = slice(inputsForTimestep,
                    {beginActivation, 0, 0},
                    {endActivation, miniBatch, 1});

                auto modifiedInput = apply(Matrix(existingInput), inputSlice, matrix::Add());

                newFirstLayerInputs.push_back(modifiedInput);
            }

            // handle remaining time inputs
            for(size_t predecessorId = firstLayerInputs.size();
                predecessorId < layers.front().timePredecessors.size(); ++predecessorId)
            {
                size_t beginActivation = splitSize * predecessorId;
                size_t endActivation   = std::min(beginActivation + splitSize,
                    layers.front().layer->getInputCount());

                auto inputSlice = slice(inputsForTimestep,
                    {beginActivation, 0, 0},
                    {endActivation, miniBatch, 1});

                newFirstLayerInputs.push_back(inputSlice);
            }

            firstLayerInputs = std::move(newFirstLayerInputs);
        }
    }

    void _routeOutputActivationsForLayerAndTimestep(StringToMatrixMap& layerToInputMap,
        MatrixVector& outputActivations, const MatrixVector& localOutputActivations,
        const Node& node, size_t timestep)
    {
        if(node.layer->getSupportsMultipleInputsAndOutputs())
        {
            return;
        }

        assert(localOutputActivations.size() == 1);

        auto outputData = _collapseActivations(localOutputActivations.front());

        // format output data according to successors
        size_t currentActivation = 0;

        for(auto& successor : node.forwardSuccessors)
        {
            size_t nextActivation = currentActivation + successor->getInputActivationCount();

            util::log("SubgraphLayer") << " Routing output activations ["
                << currentActivation << ", " << nextActivation
                << "] to forward successor '" << successor->name << "'\n";

            auto outputSlice = slice(outputData, {currentActivation, 0, 0},
                {nextActivation, outputData.size()[1], outputData.size()[2]});

            currentActivation = nextActivation;

            layerToInputMap[successor->name].push_back(outputSlice);
        }

        for(auto& successor : node.timeSuccessors)
        {
            size_t nextActivation = currentActivation + successor->getInputActivationCount();

            util::log("SubgraphLayer") << " Routing output activations ["
                << currentActivation << ", " << nextActivation
                << "] to time successor '" << successor->name << "'\n";

            auto outputSlice = slice(outputData, {currentActivation, 0, 0},
                {nextActivation, outputData.size()[1], outputData.size()[2]});

            currentActivation = nextActivation;

            layerToInputMap[successor->name].push_back(outputSlice);
        }

        size_t nextActivation = outputData.size().front();

        if(nextActivation > currentActivation)
        {
            auto outputSlice = slice(outputData, {currentActivation, 0, 0},
                {nextActivation, outputData.size()[1], outputData.size()[2]});

            util::log("SubgraphLayer") << " Routing output activations ["
                << currentActivation << ", " << nextActivation
                << "] to sublayer output.\n";

            if(timestep == 0)
            {
                outputActivations.push_back(outputSlice);
            }
            else
            {
                // find the correct output activation
                size_t outputActivationIndex = 0;

                for(; outputActivationIndex < outputActivations.size(); ++outputActivationIndex)
                {
                    if(outputActivations[outputActivationIndex].size().back() <= timestep)
                    {
                        break;
                    }
                }

                assert(outputActivationIndex < outputActivations.size());

                outputActivations[outputActivationIndex] = concatenate(
                    outputActivations[outputActivationIndex], outputSlice, 2);
            }
        }
    }

private:
    void _formatOutputDeltasForLayer(MatrixVector& outputDeltas, const Node& node)
    {
        if(node.layer->getSupportsMultipleInputsAndOutputs())
        {
            return;
        }

        if(node.totalSuccessors() < 2)
        {
            return;
        }

        auto result = _collapseActivations(outputDeltas.back());

        for(size_t i = 1; i < outputDeltas.size(); ++i)
        {
            result = concatenate(result, _collapseActivations(
                outputDeltas[outputDeltas.size() - i - 1]), 0);
        }

        outputDeltas.clear();

        outputDeltas.push_back(result);
    }

    void _formatInputDeltasForLayer(MatrixVector& inputDeltas, const Node& node)
    {
        if(node.layer->getSupportsMultipleInputsAndOutputs())
        {
            return;
        }

        if(node.forwardPredecessors.size() < 2)
        {
            return;
        }

        assert(inputDeltas.size() == 1);

        auto inputDeltasData = _collapseActivations(inputDeltas.front());

        // format input deltas according to successors
        MatrixVector newInputDeltas;

        size_t currentActivation = 0;

        for(auto predecessorIterator = node.forwardPredecessors.rbegin();
            predecessorIterator != node.forwardPredecessors.rend(); ++predecessorIterator)
        {
            auto& predecessor = *predecessorIterator;

            size_t nextActivation = currentActivation + predecessor->getOutputActivationCount();

            newInputDeltas.push_back(slice(inputDeltasData, {currentActivation, 0, 0},
                {nextActivation, inputDeltasData.size()[1], inputDeltasData.size()[2]}));

            currentActivation = nextActivation;
        }

        inputDeltas = std::move(newInputDeltas);
    }

private:
    void _zeroPadTimeConnections(StringToMatrixMap& layerToOutputDeltas,
        size_t miniBatchSize, const matrix::Precision& precision)
    {
        for(auto& node : layers)
        {
            auto& layerOutputDeltas = layerToOutputDeltas[node.name];

            for(size_t i = 0; i < node.timeSuccessors.size(); ++i)
            {
                util::log("SubgraphLayer") << " Zero padding output deltas with ["
                    << 0 << ", " << node.getOutputActivationCount()
                    << "] for '" << node.name << "'\n";

                layerOutputDeltas.push_back(
                    zeros({node.getOutputActivationCount(), miniBatchSize, 1}, precision));
            }
        }
    }

    void _updateSublayerOutputDeltasForTimestep(StringToMatrixMap& layerToOutputDeltas,
        const MatrixVector& outputDeltas, size_t timestep)
    {
        if(outputDeltas.empty())
        {
            return;
        }

        auto& lastLayerName = layers.back().name;

        assert(outputDeltas.size() <= 1);

        auto& lastLayerOutputDeltas = layerToOutputDeltas[lastLayerName];

        auto outputDeltasForTimestep = _extractTimestep(outputDeltas.front(), timestep);

        util::log("SubgraphLayer") << " Appending single timestep from sublayer output deltas ["
            << 0 << ", " << outputDeltasForTimestep.size().front()
            << "] for '" << lastLayerName << "'\n";

        lastLayerOutputDeltas.push_front(outputDeltasForTimestep);
    }

    void _routeInputDeltasForLayerAndTimestep(StringToMatrixMap& layerToOutputDeltas,
        MatrixVector& sublayerInputDeltas, const MatrixVector& localInputDeltas, const Node& node,
        size_t timesteps)
    {
        if(node.layer->getSupportsMultipleInputsAndOutputs())
        {
            return;
        }

        assert(localInputDeltas.size() == 1);

        auto inputDeltas = _collapseActivations(localInputDeltas.front());

        // format input deltas according to predecessors
        size_t totalActivations = 0;

        for(auto& predecessor : node.forwardPredecessors)
        {
            totalActivations += predecessor->getOutputActivationCount();
        }

        for(auto& predecessor : node.timePredecessors)
        {
            totalActivations += predecessor->getOutputActivationCount();
        }

        size_t currentActivation = 0;

        if(inputDeltas.size().front() > totalActivations)
        {
            size_t nextActivation = inputDeltas.size().front() - totalActivations;

            auto inputDeltasSlice = slice(inputDeltas, {currentActivation, 0, 0},
                {nextActivation, inputDeltas.size()[1], inputDeltas.size()[2]});

            util::log("SubgraphLayer") << " Routing input deltas ["
                << currentActivation << ", " << nextActivation
                << "] to sublayer input.\n";

            if(timesteps == 0)
            {
                sublayerInputDeltas.push_back(inputDeltasSlice);
            }
            else
            {
                // find the correct output activation
                size_t inputDeltasIndex = 0;

                for(; inputDeltasIndex < sublayerInputDeltas.size(); ++inputDeltasIndex)
                {
                    if(sublayerInputDeltas[inputDeltasIndex].size().back() <= timesteps)
                    {
                        break;
                    }
                }

                assert(inputDeltasIndex < sublayerInputDeltas.size());

                sublayerInputDeltas[inputDeltasIndex] = concatenate(
                    inputDeltasSlice, sublayerInputDeltas[inputDeltasIndex], 2);
            }

            currentActivation = nextActivation;
        }

        for(auto& predecessor : node.timePredecessors)
        {
            size_t nextActivation = currentActivation + predecessor->getOutputActivationCount();

            util::log("SubgraphLayer") << " Routing input deltas ["
                << currentActivation << ", " << nextActivation
                << "] to time successor '" << predecessor->name << "'\n";

            auto inputDeltasSlice = slice(inputDeltas, {currentActivation, 0, 0},
                {nextActivation, inputDeltas.size()[1], inputDeltas.size()[2]});

            currentActivation = nextActivation;

            layerToOutputDeltas[predecessor->name].push_back(inputDeltasSlice);
        }

        for(auto& predecessor : node.forwardPredecessors)
        {
            size_t nextActivation = currentActivation + predecessor->getOutputActivationCount();

            util::log("SubgraphLayer") << " Routing input deltas ["
                << currentActivation << ", " << nextActivation
                << "] to forward predecessor '" << predecessor->name << "'\n";

            auto inputDeltasSlice = slice(inputDeltas, {currentActivation, 0, 0},
                {nextActivation, inputDeltas.size()[1], inputDeltas.size()[2]});

            currentActivation = nextActivation;

            layerToOutputDeltas[predecessor->name].push_back(inputDeltasSlice);
        }

    }

private:
    void _copy(const SubgraphLayerImplementation& i)
    {
        for(auto& node : i.layers)
        {
            addLayer(node.name, node.layer->clone());
        }

        for(auto& node : i.layers)
        {
            for(auto& successor : node.forwardSuccessors)
            {
                addForwardConnection(node.name, successor->name);
            }

            for(auto& successor : node.timeSuccessors)
            {
                addTimeConnection(node.name, successor->name);
            }
        }
    }

private:
    NodePointerList _findReadyLayers()
    {
        NodeSet readyNodes;

        for(auto& layer : layerNames)
        {
            readyNodes.insert(layer.first);
        }

        for(auto& layer : layers)
        {
            for(auto& forwardConnection : layer.forwardSuccessors)
            {
                readyNodes.erase(forwardConnection->name);
            }
        }

        // all nodes should not have an in-edge in an acyclic graph
        assert(!readyNodes.empty());

        NodePointerList readyNodeList;

        for(auto& readyNode : readyNodes)
        {
            readyNodeList.push_back(layerNames[readyNode]);
        }

        return readyNodeList;
    }

    bool _allPredecessorsVisited(const NodeList::iterator node, const NodeSet& visited)
    {
        for(auto& predecessor : node->forwardPredecessors)
        {
            if(visited.count(predecessor->name) == 0)
            {
                return false;
            }
        }

        return true;
    }
};

SubgraphLayer::SubgraphLayer()
: _implementation(std::make_unique<SubgraphLayerImplementation>())
{

}

SubgraphLayer::~SubgraphLayer()
{

}

SubgraphLayer::SubgraphLayer(const SubgraphLayer& l)
: Layer(l), _implementation(std::make_unique<SubgraphLayerImplementation>(*l._implementation))
{

}

SubgraphLayer& SubgraphLayer::operator=(const SubgraphLayer& l)
{
    Layer::operator=(l);

    *_implementation = *l._implementation;

    return *this;
}

void SubgraphLayer::initialize()
{
    for(auto& layer : _implementation->layers)
    {
        layer.layer->initialize();
    }
}

void SubgraphLayer::runForwardImplementation(Bundle& bundle)
{
    // if there are no through-timestep connections, do all timesteps in one shot
    if(!_implementation->doAnyTimeConnectionsExist())
    {
        _implementation->runForwardAllTimesteps(bundle);
    }
    else
    {
        _implementation->runForwardIterateTimesteps(bundle);
    }

}

void SubgraphLayer::runReverseImplementation(Bundle& bundle)
{
    // if there are no through-timestep connections, do all timesteps in one shot
    if(!_implementation->doAnyTimeConnectionsExist())
    {
        _implementation->runReverseAllTimesteps(bundle);
    }
    else
    {
        _implementation->runReverseIterateTimesteps(bundle);
    }
}

MatrixVector& SubgraphLayer::weights()
{
    return _implementation->weights;
}

const MatrixVector& SubgraphLayer::weights() const
{
    return _implementation->weights;
}

const matrix::Precision& SubgraphLayer::precision() const
{
    return _implementation->front().precision();
}

double SubgraphLayer::computeWeightCost() const
{
    double cost = 0.0;

    for(auto& layer : _implementation->layers)
    {
        cost += layer.layer->computeWeightCost();
    }

    return cost;
}

Dimension SubgraphLayer::getInputSize() const
{
    return _implementation->getInputSize();
}

Dimension SubgraphLayer::getOutputSize() const
{
    return _implementation->getOutputSize();
}

size_t SubgraphLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t SubgraphLayer::getOutputCount() const
{
    return getOutputSize().product();
}

size_t SubgraphLayer::totalNeurons() const
{
    size_t total = 0;

    for(auto& node : _implementation->layers)
    {
        total += node.layer->totalNeurons();
    }

    return total;
}

size_t SubgraphLayer::totalConnections() const
{
    size_t total = 0;

    for(auto& node : _implementation->layers)
    {
        total += node.layer->totalConnections();
    }

    return total;
}

size_t SubgraphLayer::getFloatingPointOperationCount() const
{
    size_t total = 0;

    for(auto& node : _implementation->layers)
    {
        total += node.layer->getFloatingPointOperationCount();
    }

    return total;
}

size_t SubgraphLayer::getActivationMemory() const
{
    size_t total = 0;

    for(auto& node : _implementation->layers)
    {
        total += node.layer->getActivationMemory();
    }

    return total;
}

void SubgraphLayer::save(util::OutputTarArchive& archive, util::PropertyTree& properties) const
{
    auto& sublayers = properties["sublayers"];

    auto& forwardConnections = properties["forward-connections"];
    auto& timeConnections    = properties["time-connections"];

    for(auto& node : _implementation->layers)
    {
        auto& layerProperties = sublayers[node.name];

        auto& layer = node.layer;

        layerProperties["type"] = layer->getTypeName();

        layer->save(archive, layerProperties);

        for(auto& successor : node.forwardSuccessors)
        {
            forwardConnections[node.name] = successor->name;
        }

        for(auto& successor : node.timeSuccessors)
        {
            timeConnections[node.name] = successor->name;
        }
    }

    saveLayer(archive, properties);
}

void SubgraphLayer::load(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    auto& sublayers = properties["sublayers"];

    for(auto& layerProperties : sublayers)
    {
        auto layerName = layerProperties.key();

        addLayer(layerName, LayerFactory::create(layerProperties["type"]));

        _implementation->getLayer(layerName)->load(archive, layerProperties);
    }

    auto& forwardConnections = properties["forward-connections"];

    for(auto& connection : forwardConnections)
    {
        addForwardConnection(connection.key(), connection.value());
    }

    auto& timeConnections = properties["forward-connections"];

    for(auto& connection : timeConnections)
    {
        addTimeConnection(connection.key(), connection.value());
    }

    prepareSubgraphForEvaluation();

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> SubgraphLayer::clone() const
{
    return std::make_unique<SubgraphLayer>(*this);
}

std::string SubgraphLayer::getTypeName() const
{
    return "SubgraphLayer";
}

void SubgraphLayer::addLayer(const std::string& layerName, std::unique_ptr<Layer>&& layer)
{

    util::log("SubgraphLayer") << "Adding layer '" << layerName <<
        "' " << layer->shapeString() << "\n";
    _implementation->addLayer(layerName, std::move(layer));
}

void SubgraphLayer::addForwardConnection(const std::string& source, const std::string& destination)
{
    util::log("SubgraphLayer") << "Adding forward connection '" << source <<
        "' -> '" << destination << "'\n";

    _implementation->addForwardConnection(source, destination);
}

void SubgraphLayer::addTimeConnection(const std::string& source, const std::string& destination)
{
    util::log("SubgraphLayer") << "Adding time connection '" << source <<
        "' -> '" << destination << "'\n";

    _implementation->addTimeConnection(source, destination);
}

void SubgraphLayer::prepareSubgraphForEvaluation()
{
    _implementation->prepareSubgraphForEvaluation();
}

}

}


