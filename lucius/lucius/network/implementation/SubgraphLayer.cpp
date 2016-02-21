/*  \file   SubgraphLayer.h
    \author Gregory Diamos
    \date   January 20, 2016
    \brief  The interface file for the SubgraphLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/SubgraphLayer.h>

#include <lucius/network/interface/LayerFactory.h>

#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/Dimension.h>

#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/PropertyTree.h>

// Standard Library Includes
#include <list>

namespace lucius
{
namespace network
{

typedef matrix::MatrixVector MatrixVector;
typedef matrix::Dimension Dimension;

class SubgraphLayerImplementation
{
public:

    class Node
    {
    public:
        typedef std::list<Node> NodeList;
        typedef std::list<NodeList::iterator> NodePointerList;

    public:
        Node(const std::string& name, std::unique_ptr<Layer>&& layer)
        : name(name), layer(std::move(layer))
        {

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

    };

    typedef Node::NodeList NodeList;
    typedef Node::NodePointerList NodePointerList;
    typedef std::map<std::string, NodeList::iterator> NameToLayerMap;
    typedef std::set<std::pair<std::string, std::string>> ConnectionSet;
    typedef std::set<std::string> NodeSet;

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
    void prepareSubgraphForEvaluation()
    {
        // sort layers in topological order according to forward connections
        auto frontier = _findReadyLayers();

        NodeSet visited;

        // Create the new layer list and name map
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

        NodeList newLayerList;

        for(auto& layer : newOrder)
        {
            newLayerList.push_back(std::move(*layer));
        }

        layers = std::move(newLayerList);
    }

public:
    void runForwardAllTimesteps(MatrixVector& outputActivations,
        const MatrixVector& inputActivations)
    {
        if(layers.empty())
        {
            return;
        }

        // Set the input activations for the first layer
        std::map<std::string, MatrixVector> layerToInputActivations;

        layerToInputActivations[layers.front().name] = inputActivations;

        // Evaluate the layers in order
        for(auto& layer : layers)
        {
            auto& name = layer.name;

            // get the inputs
            auto& inputs = layerToInputActivations[name];

            // run the layer
            MatrixVector localOutputs;

            layer.layer->runForward(localOutputs, inputs);

            inputs.clear();

            assert(localOutputs.size() <= layer.forwardSuccessors.size());

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

    void runForwardIterateTimesteps(MatrixVector& outputActivations,
        const MatrixVector& inputActivations)
    {
        if(layers.empty())
        {
            return;
        }

        // Get the total number of timesteps
        size_t timesteps = inputActivations.front().size().back();

        // Set the input activations for the first timestep
        std::map<std::string, MatrixVector> layerToInputActivations;

        layerToInputActivations[layers.front().name] = inputActivations;

        // Step through each timestep
        for(size_t timestep = 0; timestep < timesteps; ++timestep)
        {
            // Evaluate layers for this timestep in order
            for(auto& node : layers)
            {
                auto& name  = node.name;
                auto& layer = *node.layer;

                // get the inputs
                auto& inputs = layerToInputActivations[name];

                // run the layer
                MatrixVector localOutputs;

                layer.runForward(localOutputs, inputs);

                inputs.clear();

                assert(localOutputs.size() <= node.forwardSuccessors.size());

                size_t index = 0;

                // handle data sent along forward connections
                for(auto& successor : node.forwardSuccessors)
                {
                    layerToInputActivations[successor->name].push_back(
                        std::move(localOutputs[index++]));
                }

                // handle data sent along through-time connections
                for(auto& successor : node.timeSuccessors)
                {
                    layerToInputActivations[successor->name].push_back(
                        std::move(localOutputs[index++]));
                }

                // outputs go to the sublayer outputs (to be saved for backprop)
                if(timestep == 0)
                {
                    for(size_t i = index; i < localOutputs.size(); ++i)
                    {
                        auto size = localOutputs[i].size();

                        size.back() = timesteps;

                        outputActivations.push_back(zeros(size, localOutputs[i].precision()));
                    }
                }
                else
                {
                    for(size_t i = index; i < localOutputs.size(); ++i)
                    {
                        auto begin = localOutputs[i].size();
                        auto end   = localOutputs[i].size();

                        begin.back() = timestep;
                        end.back()   = timestep + 1;

                        auto outputSlice = slice(outputActivations[i - index], begin, end);

                        copy(outputSlice, localOutputs[i]);
                    }
                }
            }
        }
    }

public:
    void runReverseAllTimesteps(MatrixVector& gradients, MatrixVector& inputDeltas,
        const MatrixVector& outputDeltas)
    {
        if(layers.empty())
        {
            return;
        }

        // Set the output deltas for the last layer
        std::map<std::string, MatrixVector> layerToOutputDeltas;

        layerToOutputDeltas[layers.back().name] = outputDeltas;

        // Evaluate the layers in reverse order
        for(auto node = layers.rbegin(); node != layers.rend(); ++node)
        {
            auto& name  = node->name;
            auto& layer = *node->layer;

            // get the inputs
            auto& deltas = layerToOutputDeltas[name];

            // run the layer
            MatrixVector localInputDeltas;
            MatrixVector localGradients;

            layer.runReverse(gradients, localInputDeltas, deltas);

            deltas.clear();

            assert(localInputDeltas.size() <=  node->forwardPredecessors.size());

            size_t index = 0;
            for(auto predecessorIterator = node->forwardPredecessors.rbegin();
                predecessorIterator != node->forwardPredecessors.rend(); ++predecessorIterator)
            {
                auto& predecessor = *predecessorIterator;

                layerToOutputDeltas[predecessor->name].push_back(
                    std::move(localInputDeltas[index]));

                ++index;
            }

            // remaining outputs go to the sublayer outputs
            for(; index < localInputDeltas.size(); ++index)
            {
                inputDeltas.push_back(std::move(localInputDeltas[index]));
            }

            gradients.push_back(std::move(localGradients));
        }

    }

    void runReverseIterateTimesteps(MatrixVector& gradients, MatrixVector& inputDeltas,
        const MatrixVector& outputDeltas)
    {
        if(layers.empty())
        {
            return;
        }

        // Set the output deltas for the last layer
        std::map<std::string, MatrixVector> layerToOutputDeltas;

        // Get the total number of timesteps
        size_t timesteps = outputDeltas.front().size().back();

        layerToOutputDeltas[layers.back().name] = outputDeltas;

        // step through each timestep
        size_t gradientStart = 0;

        for(size_t t = 0; t < timesteps; ++t)
        {
            // Evaluate the layers in reverse order
            for(auto node = layers.rbegin(); node != layers.rend(); ++node)
            {
                auto& name  = node->name;
                auto& layer = *node->layer;

                // get the inputs
                auto& deltas = layerToOutputDeltas[name];

                // run the layer
                MatrixVector localInputDeltas;
                MatrixVector localGradients;

                layer.runReverse(gradients, localInputDeltas, deltas);

                deltas.clear();

                assert(localInputDeltas.size() <= node->forwardPredecessors.size());

                size_t index = 0;
                for(auto predecessorIterator = node->forwardPredecessors.rbegin();
                    predecessorIterator != node->forwardPredecessors.rend(); ++predecessorIterator)
                {
                    auto& predecessor = *predecessorIterator;

                    layerToOutputDeltas[predecessor->name].push_back(
                        std::move(localInputDeltas[index]));

                    ++index;
                }

                for(auto predecessorIterator = node->timePredecessors.rbegin();
                    predecessorIterator != node->timePredecessors.rend(); ++predecessorIterator)
                {
                    auto& predecessor = *predecessorIterator;

                    layerToOutputDeltas[predecessor->name].push_back(
                        std::move(localInputDeltas[index]));

                    ++index;
                }

                // remaining outputs go to the sublayer outputs
                for(; index < localInputDeltas.size(); ++index)
                {
                    inputDeltas.push_back(std::move(localInputDeltas[index]));
                }

                // accumulate into the gradients
                if(t == 0)
                {
                    gradientStart = gradients.size();
                    gradients.push_back(std::move(localGradients));
                }
                else
                {
                    for(size_t i = 0; i < localGradients.size(); ++i)
                    {
                        apply(gradients[i + gradientStart], gradients[i + gradientStart],
                            localGradients[i], matrix::Add());
                    }
                }
            }
        }
    }

public:
    void addLayer(const std::string& layerName, std::unique_ptr<Layer>&& layer)
    {
        assert(layerNames.count(layerName) == 0);

        auto layerIterator = layers.insert(layers.end(), Node(layerName, std::move(layer)));

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
: _implementation(std::make_unique<SubgraphLayerImplementation>(*l._implementation))
{

}

SubgraphLayer& SubgraphLayer::operator=(const SubgraphLayer& l)
{
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

void SubgraphLayer::runForwardImplementation(MatrixVector& outputActivations,
    const MatrixVector& inputActivations)
{
    // if there are no through-timestep connections, do all timesteps in one shot
    if(_implementation->doAnyTimeConnectionsExist())
    {
        _implementation->runForwardAllTimesteps(outputActivations, inputActivations);
    }
    else
    {
        _implementation->runForwardIterateTimesteps(outputActivations, inputActivations);
    }

}

void SubgraphLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& inputDeltas,
    const MatrixVector& outputDeltas)
{
    // if there are no through-timestep connections, do all timesteps in one shot
    if(_implementation->doAnyTimeConnectionsExist())
    {
        _implementation->runReverseAllTimesteps(gradients, inputDeltas,
            outputDeltas);
    }
    else
    {
        _implementation->runReverseIterateTimesteps(gradients, inputDeltas,
            outputDeltas);
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
    return _implementation->front().getInputSize();
}

Dimension SubgraphLayer::getOutputSize() const
{
    return _implementation->back().getOutputSize();
}

size_t SubgraphLayer::getInputCount() const
{
    return _implementation->front().getInputCount();
}

size_t SubgraphLayer::getOutputCount() const
{
    return _implementation->back().getOutputCount();
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
    _implementation->addLayer(layerName, std::move(layer));
}

void SubgraphLayer::addForwardConnection(const std::string& source, const std::string& destination)
{
    _implementation->addForwardConnection(source, destination);
}

void SubgraphLayer::addTimeConnection(const std::string& source, const std::string& destination)
{
    _implementation->addTimeConnection(source, destination);
}

void SubgraphLayer::prepareSubgraphForEvaluation()
{
    _implementation->prepareSubgraphForEvaluation();
}

}

}


