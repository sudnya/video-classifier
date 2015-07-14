/*    \file   Model.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Model class.
*/

// Lucious Includes
#include <lucious/model/interface/Model.h>

#include <lucious/network/interface/NeuralNetwork.h>

#include <lucious/matrix/interface/Matrix.h>

#include <lucious/util/interface/TarArchive.h>
#include <lucious/util/interface/PropertyTree.h>

#include <lucious/util/interface/debug.h>
#include <lucious/util/interface/json.h>

// Standard Library Includes
#include <sstream>
#include <map>

namespace lucious
{

namespace model
{

Model::Model(const std::string& path)
: _path(path), _loaded(false)
{

}

Model::Model()
: _path("unknown-path"), _loaded(true)
{

}

const Model::NeuralNetwork& Model::getNeuralNetwork(
    const std::string& name) const
{
    auto network = _neuralNetworkMap.find(name);

    assertM(network != _neuralNetworkMap.end(), "Invalid neural network name "
        + name);

    return *network->second;
}

Model::NeuralNetwork& Model::getNeuralNetwork(
    const std::string& name)
{
    auto network = _neuralNetworkMap.find(name);

    assertM(network != _neuralNetworkMap.end(), "Invalid neural network name "
        + name);

    return *network->second;
}

bool Model::containsNeuralNetwork(const std::string& name) const
{
    return _neuralNetworkMap.count(name) != 0;
}

void Model::setNeuralNetwork(
    const std::string& name, const NeuralNetwork& n)
{
    if(!containsNeuralNetwork(name))
    {
        _neuralNetworkMap[name] = _neuralNetworks.insert(_neuralNetworks.end(), n);
    }
    else
    {
        *_neuralNetworkMap[name] = n;
    }
}

void Model::setOutputLabel(size_t output, const std::string& label)
{
    _outputLabels[output] = label;
}

std::string Model::getOutputLabel(size_t output) const
{
    auto label = _outputLabels.find(output);

    assert(label != _outputLabels.end());

    return label->second;
}

size_t Model::getOutputCount() const
{
    return _outputLabels.size();
}

void Model::save(std::ostream& stream) const
{
    util::OutputTarArchive tar(stream);

    util::PropertyTree properties("metadata");

    auto& networks = properties["networks"];

    for(auto& network : _neuralNetworkMap)
    {
        auto& networkProperties = networks[network.first];

        network.second->save(tar, networkProperties);
    }

    auto& outputLabels = properties["output-labels"];

    for(auto& outputLabel : _outputLabels)
    {
        outputLabels[std::to_string(outputLabel.first)] = outputLabel.second;
    }

    if(!_attributes.empty())
    {
        auto& attributes = properties["attributes"];

        for(auto& attribute : _attributes)
        {
            attributes[attribute.first] = attribute.second;
        }
    }

    std::stringstream json;

    properties.saveJson(json);

    tar.addFile("metadata.txt", json);
}

void Model::load(std::istream& stream)
{
    if(_loaded) return;

    _loaded = true;

    clear();

    util::log("Model") << "Loading classification-model from '"
        << _path << "'\n";

    util::InputTarArchive tar(stream);

    std::stringstream metadataText;

    tar.extractFile("metadata.txt", metadataText);

    util::log("Model") << " metadata text: " << metadataText.str() << "\n";

    auto metadataObject = util::PropertyTree::loadJson(metadataText);

    util::log("Model") << " metadata object: " << metadataObject.jsonString() << "\n";

    auto& metadata = metadataObject["metadata"];

    auto& networks = metadata["networks"];

    for(auto& network : networks)
    {
        auto name = network.key();

        setNeuralNetwork(name, network::NeuralNetwork());

        getNeuralNetwork(name).load(tar, network);
    }

    auto& attributes = metadata["attributes"];

    for(auto& attribute : attributes)
    {
        _attributes[attribute.key()] = attribute.value();
    }

    auto& labels = metadata["output-labels"];

    for(auto& label : labels)
    {
        _outputLabels[label.key<size_t>()] = label.value();
    }
}

void Model::save() const
{
    std::ofstream stream(_path);

    if(!stream.is_open())
    {
        throw std::runtime_error("Could not open path '" +
            _path + "' to write a model file.");
    }

    save(stream);
}

void Model::load()
{
    std::ifstream stream(_path);

    if(!stream.is_open())
    {
        throw std::runtime_error("Failed to open path '" +
            _path + "' for reading model file.");
    }

    load(stream);
}

void Model::clear()
{
    _loaded = false;
    _neuralNetworks.clear();
    _neuralNetworkMap.clear();
    _outputLabels.clear();
    _attributes.clear();
}

Model::iterator Model::begin()
{
    return _neuralNetworks.begin();
}

Model::const_iterator Model::begin() const
{
    return _neuralNetworks.begin();
}

Model::iterator Model::end()
{
    return _neuralNetworks.end();
}

Model::const_iterator Model::end() const
{
    return _neuralNetworks.end();
}

Model::reverse_iterator Model::rbegin()
{
    return _neuralNetworks.rbegin();
}

Model::const_reverse_iterator Model::rbegin() const
{
    return _neuralNetworks.rbegin();
}

Model::reverse_iterator Model::rend()
{
    return _neuralNetworks.rend();
}

Model::const_reverse_iterator Model::rend() const
{
    return _neuralNetworks.rend();
}

}

}


