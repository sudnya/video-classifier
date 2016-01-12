/*  \file   ModelSpecification.cpp
    \date   Saturday April 26, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ModelSpecification class.
*/

// Lucius Includes
#include <lucius/model/interface/ModelSpecification.h>
#include <lucius/model/interface/Model.h>

#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/Layer.h>
#include <lucius/network/interface/LayerFactory.h>
#include <lucius/network/interface/CostFunctionFactory.h>

#include <lucius/matrix/interface/Dimension.h>

#include <lucius/util/interface/PropertyTree.h>
#include <lucius/util/interface/ParameterPack.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{

namespace model
{

class ModelSpecificationImplementation
{
public:
    ModelSpecificationImplementation();

public:
    void parseSpecification(const std::string& specification);
    void initializeModel(Model& model);

private:
    std::unique_ptr<util::PropertyTree> _specification;
};


ModelSpecification::ModelSpecification(const std::string& specification)
: _implementation(new ModelSpecificationImplementation)
{
    if(!specification.empty())
    {
        parseSpecification(specification);
    }
}

ModelSpecification::~ModelSpecification() = default;

void ModelSpecification::parseSpecification(const std::string& specification)
{
    _implementation->parseSpecification(specification);
}

void ModelSpecification::initializeModel(Model& model)
{
    _implementation->initializeModel(model);
}

ModelSpecificationImplementation::ModelSpecificationImplementation()
{

}

void ModelSpecificationImplementation::parseSpecification(const std::string& specification)
{
    std::stringstream jsonStream(specification);

    *_specification = util::PropertyTree::loadJson(jsonStream);
}

static void checkSections(const util::PropertyTree& specification)
{
    if(!specification.exists("layer-types"))
    {
        throw std::runtime_error("Specification is missing a 'layer-types' section.");
    }

    if(!specification.exists("networks"))
    {
        throw std::runtime_error("Specification is missing a 'networks' section.");
    }

    if(!specification.exists("cost-function"))
    {
        throw std::runtime_error("Specification is missing a 'cost-function' section.");
    }
}

static void loadModelAttributes(Model& model, const util::PropertyTree& specification)
{
    auto attributes = specification.get("model-attributes");

    for(auto& attribute : attributes)
    {
        model.setAttribute(attribute.key(), attribute.value());
    }
}

typedef std::map<std::string, util::ParameterPack> TypeMap;

static void loadNetwork(Model& model, const util::PropertyTree& network, const TypeMap& types)
{
    network::NeuralNetwork neuralNetwork;

    auto layers = network.get("layers");

    for(auto& layer : layers)
    {
        auto layerName = layer.key();

        auto layerType = types.find(layerName);

        if(layerType == types.end())
        {
            throw std::runtime_error("Unknown layer type name '" + layerName + "'.");
        }

        neuralNetwork.addLayer(network::LayerFactory::create(layerName, layerType->second));
    }

    neuralNetwork.initialize();

    model.setNeuralNetwork(network.key(), neuralNetwork);
}

static void loadType(TypeMap& types, const util::PropertyTree& type)
{
    if(types.count(type.key()) != 0)
    {
        throw std::runtime_error("Duplicate layer type name '" + type.key() + "'.");
    }

    auto attributes = type.get();

    util::ParameterPack pack;

    for(auto& property : attributes)
    {
        pack.insert(property.key(), property.value());
    }

    types.insert(std::make_pair(type.key(), pack));
}

static TypeMap loadTypes(const util::PropertyTree& specification)
{
    auto types = specification.get("layer-types");

    TypeMap typeMap;

    for(auto& type : types)
    {
        loadType(typeMap, type);
    }

    return typeMap;
}

static void loadNetworks(Model& model, const util::PropertyTree& specification)
{
    auto types = loadTypes(specification);

    auto networks = specification.get("networks");

    for(auto& network : networks)
    {
        loadNetwork(model, network, types);
    }
}

static void loadCostFunction(Model& model, const util::PropertyTree& specification)
{
    auto name = specification.get<std::string>("cost-function.name");

    auto& network = model.getNeuralNetwork("Classifier");

    network.setCostFunction(network::CostFunctionFactory::create(name));
}

void ModelSpecificationImplementation::initializeModel(Model& model)
{
    model.clear();

    checkSections(*_specification);

    loadModelAttributes(model, *_specification);
    loadNetworks(model, *_specification);
    loadCostFunction(model, *_specification);
}

}

}

