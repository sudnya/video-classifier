/*  \file   SubgraphLayer.h
    \author Gregory Diamos
    \date   January 20, 2016
    \brief  The interface file for the SubgraphLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

namespace lucius
{
namespace network
{

class SubgraphLayerImplementation
{
public:
    class Node;

    typedef std::list<Node> NodeList;
    typedef std::list<Node::iterator> NodePointerList;

    class Node
    {
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

    typedef std::map<std::string, LayerList::iterator> NameToLayerMap;
    typedef std::set<std::pair<std::string, std::string>> ConnectionSet;
    typedef std::set<std::string> NodeSet;

public:
    NodeList       layers;
    NameToLayerMap layerNames;

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
                    frontier.insert(successor);
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

        layerToInputActivations[layers.front().first] = inputActivations;

        // Evaluate the layers in order
        for(auto& nameAndLayer : layers)
        {
            auto& name = nameAndLayer.first;

            // get the inputs
            auto& inputs = layerToInputActivations[name];

            // run the layer
            MatrixVector localOutputs;

            nameAndLayer.second->runForward(localOutputs, inputs);

            inputs.clear();

            assert(localOutputs.size() <= nameAndLayer.second->forwardSuccessors.size());

            // initial outputs go to successor outputs
            size_t index = 0;
            for(auto& successor : nameAndLayer.second->forwardSuccessors)
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

        layerToInputActivations[layers.front().first] = inputActivations;

        // Step through each timestep
        for(size_t timestep = 0; timestep < timesteps; ++timestep)
        {
            // Evaluate layers for this timestep in order
            for(auto& nameAndLayer : layers)
            {
                auto& name  = nameAndLayer.first;
                auto& layer = nameAndLayer.second;

                // get the inputs
                auto& inputs = layerToInputActivations[name];

                // run the layer
                MatrixVector localOutputs;

                layer->runForward(localOutputs, inputs);

                inputs.clear();

                assert(localOutputs.size() <= layer->forwardSuccessors.size());

                size_t index = 0;

                // handle data sent along forward connections
                for(auto& successor : layer->forwardSuccessors)
                {
                    layerToInputActivations[successor->name].push_back(
                        std::move(localOutputs[index++]));
                }

                // handle data sent along through-time connections
                for(auto& successor : layer->timeSuccessors)
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

        layerToOutputDeltas[layers.back().first] = outputDeltas;

        // Evaluate the layers in reverse order
        for(auto nameAndLayer = layers.rbegin(); nameAndLayer != layers.rend(); ++nameAndLayer)
        {
            auto& name  = nameAndLayer->first;
            auto& layer = nameAndLayer->second;

            // get the inputs
            auto& deltas = layerToOutputDeltas[name];

            // run the layer
            MatrixVector localInputDeltas;
            MatrixVector localGradients;

            layer->runReverse(gradients, localInputDeltas, deltas);

            deltas.clear();

            assert(localInputDeltas.size() <= layer->forwardPredecessors.size());

            size_t index = 0;
            for(auto predecessorIterator = layer->forwardPredecessors.rbegin();
                predecessorIterator != layer->forwardPredecessors.rend(); ++predecessorIterator)
            {
                auto& predecessor = *predecessorIterator;

                layerToOutputDeltas[predecessor->name].push_back(
                    std::move(localInputDeltas[index]));

                ++index;
            }

            // remaining outputs go to the sublayer outputs
            for(; index < localOutputs.size(); ++index)
            {
                inputDeltas.push_back(std::move(localInputDeltas[index]));
            }

            gradients.push_back(std::move(localGradients));
        }

    }

    void runReverseIterateTimesteps(MatrixVector& gradients, MatrixVector& inputDeltas,
        const MatrixVector& outputActivations, const MatrixVector& outputDeltas)
    {
        if(layers.empty())
        {
            return;
        }

        // Set the output deltas for the last layer
        std::map<std::string, MatrixVector> layerToOutputDeltas;

        // Get the total number of timesteps
        size_t timesteps = outputDeltas.front().size().back();

        layerToOutputDeltas[layers.back().first] = outputDeltas;

        // step through each timestep
        size_t gradientStart = 0;

        for(size_t t = 0; t < timesteps; ++t)
        {
            size_t timestep = timesteps - t - 1;

            // Evaluate the layers in reverse order
            for(auto nameAndLayer = layers.rbegin(); nameAndLayer != layers.rend(); ++nameAndLayer)
            {
                auto& name  = nameAndLayer->first;
                auto& layer = nameAndLayer->second;

                // get the inputs
                auto& deltas = layerToOutputDeltas[name];

                // run the layer
                MatrixVector localInputDeltas;
                MatrixVector localGradients;

                layer->runReverse(gradients, localInputDeltas, deltas);

                deltas.clear();

                assert(localInputDeltas.size() <= layer->forwardPredecessors.size());

                size_t index = 0;
                for(auto predecessorIterator = layer->forwardPredecessors.rbegin();
                    predecessorIterator != layer->forwardPredecessors.rend(); ++predecessorIterator)
                {
                    auto& predecessor = *predecessorIterator;

                    layerToOutputDeltas[predecessor->name].push_back(
                        std::move(localInputDeltas[index]));

                    ++index;
                }

                for(auto predecessorIterator = layer->timePredecessors.rbegin();
                    predecessorIterator != layer->timePredecessors.rend(); ++predecessorIterator)
                {
                    auto& predecessor = *predecessorIterator;

                    layerToOutputDeltas[predecessor->name].push_back(
                        std::move(localInputDeltas[index]));

                    ++index;
                }

                // remaining outputs go to the sublayer outputs
                for(; index < localOutputs.size(); ++index)
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

private:
    NodeSet _findReadyLayers()
    {
        NodeSet readyNodes;

        for(auto& layer : layerNames)
        {
            readyNodes.insert(layer.first);
        }

        for(auto& connection : forwardConnection)
        {
            readyNodes.erase(connection.second);
        }

        // all nodes should not have an in-edge in an acyclic graph
        assert(!readyNodes.empty());

        return readyNodes;
    }

    ConnectionSet _getMatchingConnections(const ConnectionSet& connections,
        const std::string& sourceName)
    {
        auto range = connections.equal_range(sourceName);

        return ConnectionSet(range.first, range.second);
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

SubgraphLayer::SubgraphLayer& operator=(const SubgraphLayer& l)
{
    *_implementation = *l._implementation;
}

void SubgraphLayer::initialize()
{
    for(auto& layer : _implementation->layers)
    {
        layer->initialize();
    }
}

void SubgraphLayer::runForwardImplementation(MatrixVector& outputActivations,
    const MatrixVector& inputActivations)
{
    // if there are no through-timestep connections, do all timesteps in one shot
    if(_implementation->timeConnections.empty())
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
    const MatrixVector& outputActivations,
    const MatrixVector& outputDeltas)
{
    // if there are no through-timestep connections, do all timesteps in one shot
    if(_implementation->timeConnections.empty())
    {
        _implementation->runReverseAllTimesteps(gradients, inputDeltas,
            outputActivations, outputDeltas);
    }
    else
    {
        _implementation->runReverseIterateTimesteps(gradients, inputDeltas,
            outputActivations, outputDeltas);
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
    return _implementation->front()->precision();
}

double SubgraphLayer::computeWeightCost() const
{
    double cost = 0.0;

    for(auto& layer : _implementation->layers)
    {
        cost += layer->computeWeightCost();
    }
}

Dimension SubgraphLayer::getInputSize() const
{
    return _implementation->front()->getInputSize();
}

Dimension SubgraphLayer::getOutputSize() const
{
    return _implementation->back()->getOutputSize();
}

size_t SubgraphLayer::getInputCount() const
{
    return _implementation->front()->getInputCount();
}

size_t SubgraphLayer::getOutputCount() const
{
    return _implementation->back()->getOutputCount();
}

size_t SubgraphLayer::totalNeurons() const
{
    size_t total = 0;

    for(auto& layer : _implementation->layers)
    {
        total += layer.second->totalNeurons();
    }

    return total;
}

size_t SubgraphLayer::totalConnections() const
{
    size_t total = 0;

    for(auto& layer : _implementation->layers)
    {
        total += layer.second->totalConnections();
    }
}

size_t SubgraphLayer::getFloatingPointOperationCount() const
{
    size_t total = 0;

    for(auto& layer : _implementation->layers)
    {
        total += layer.second->getFloatingPointOperationCount();
    }
}

size_t SubgraphLayer::getActivationMemory() const
{
    size_t total = 0;

    for(auto& layer : _implementation->layers)
    {
        total += layer.second->getActivationMemory();
    }

    return total;
}

void SubgraphLayer::save(util::OutputTarArchive& archive, util::PropertyTree& properties) const
{
    auto& sublayers = properties["sublayers"];

    for(auto& layerName : _implementation->layerNames)
    {
        auto& layerPropertyies = sublayers[layerName];

        auto& layer = _implementation->getLayer(layerName);

        layerProperties["type"] = layer->getTypeName();

        layer->save(archive, layerProperties);
    }

    auto& forwardConnections = properties["forward-connections"];

    for(auto& connection : _implementation->forwardConnections)
    {
        forwardConnections[connection.first] = connection.second;
    }

    auto& timeConnections = properties["time-connections"];

    for(auto& connection : _implementation->timeConnections)
    {
        timeConnections[connection.first] = connection.second;
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
    assert(_implementation->layerNames.count(layerName) == 0);

    auto layerIterator = _implementation->layers.insert(
        _implementation->layers.end(), std::move(layer));

    _implementation->layerNames.insert(std::make_pair(layerName, layerIterator));
}

void SubgraphLayer::addForwardConnection(const std::string& source, const std::string& destination)
{
    assert(_impementation->layerNames.count(source)      != 0);
    assert(_impementation->layerNames.count(destination) != 0);

    _implementation->forwardConnections.insert(std::make_pair(source, destination));
}

void SubgraphLayer::addTimeConnection(const std::string& source, const std::string& destination)
{
    assert(_impementation->layerNames.count(source)      != 0);
    assert(_impementation->layerNames.count(destination) != 0);

    _implementation->timeConnections.insert(std::make_pair(source, destination));
}

void SubgraphLayer::prepareSubgraphForEvaluation()
{
    _implementation->prepareSubgraphForEvaluation();
}

}

}


