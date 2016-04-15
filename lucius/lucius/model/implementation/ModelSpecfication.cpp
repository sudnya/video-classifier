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
#include <lucius/network/interface/SubgraphLayer.h>
#include <lucius/network/interface/CostFunctionFactory.h>

#include <lucius/matrix/interface/Dimension.h>

#include <lucius/database/interface/SampleDatabase.h>

#include <lucius/util/interface/PropertyTree.h>
#include <lucius/util/interface/ParameterPack.h>
#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{

namespace model
{

typedef network::Layer Layer;
typedef network::SubgraphLayer SubgraphLayer;
typedef matrix::Dimension Dimension;

class ModelSpecificationImplementation
{
public:
    ModelSpecificationImplementation();

public:
    void parseSpecification(const std::string& specification);
    void initializeModel(Model& model);

private:
    util::PropertyTree _specification;
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

    _specification = util::PropertyTree::loadJson(jsonStream);
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
    if(!specification.exists("model-attributes"))
    {
        return;
    }

    auto attributes = specification.get("model-attributes");

    for(auto& attribute : attributes)
    {
        if(attribute.isList())
        {
            continue;
        }

        model.setAttribute(attribute.key(), attribute.value());
    }
}

typedef std::map<std::string, util::ParameterPack> TypeMap;

matrix::Dimension computeInputSize(const util::PropertyTree& model)
{
    if(!model.exists("model-attributes"))
    {
        return matrix::Dimension(1, 1, 1, 1, 1);
    }

    auto attributes = model.get("model-attributes");

    matrix::Dimension result;

    if(attributes.exists("ResolutionX"))
    {
        result.push_back(attributes.get<size_t>("ResolutionY"));
        result.push_back(attributes.get<size_t>("ResolutionX"));
        result.push_back(attributes.get<size_t>("ColorComponents"));
        result.push_back(1);
        result.push_back(1);
    }
    else if(attributes.exists("FrameDuration"))
    {
        result.push_back(attributes.get<size_t>("FrameDuration"));
        result.push_back(1);
        result.push_back(1);
        result.push_back(1);
        result.push_back(1);
    }
    else
    {
        throw std::runtime_error("Unknown input data type.");
    }

    return result;
}

static void appendInputSize(util::ParameterPack& parameters, const matrix::Dimension& inputSize)
{
    parameters.insert("InputSizeWidth", inputSize[0]);

    if(inputSize.size() > 3)
    {
        parameters.insert("InputSizeHeight", inputSize[1]);
    }

    if(inputSize.size() > 4)
    {
        parameters.insert("InputSizeChannels", inputSize[2]);
    }

    if(inputSize.size() > 1)
    {
        parameters.insert("InputSizeBatch",    inputSize[inputSize.size()-2]);
    }

    if(inputSize.size() > 2)
    {
        parameters.insert("InputSizeTimesteps", inputSize[inputSize.size()-1]);
    }
}

static void setupOutputLayerParameters(model::Model& model,
    util::ParameterPack& layerParameters, const util::PropertyTree& specification)
{
    if(!specification.exists("infer-outputs-from"))
    {
        return;
    }

    auto datasetPath = specification.get<std::string>("infer-outputs-from");

    database::SampleDatabase inputDatabase(datasetPath);

    if(specification.exists("model-attributes.Graphemes"))
    {
        for(auto& grapheme : specification.get("model-attributes.Graphemes"))
        {
            inputDatabase.addGrapheme(grapheme.key());
        }

        inputDatabase.addGrapheme("-SEPARATOR-");

        model.setAttribute("UsesGraphemes", "1");
    }

    if(specification.exists("model-attributes.DelimiterGrapheme"))
    {
        inputDatabase.setDelimiterGrapheme(
            specification.get("model-attributes.DelimiterGrapheme").value());
    }
    else
    {
        inputDatabase.setDelimiterGrapheme("END");
    }

    model.setAttribute("DelimiterGrapheme", inputDatabase.getDelimiterGrapheme());

    inputDatabase.load();

    auto labels = inputDatabase.getAllPossibleLabels();

    size_t index = 0;

    for(auto& label : labels)
    {
        model.setOutputLabel(index++, label);
    }

    layerParameters.insert("OutputSize", labels.size());
}

static void populateSubgraphLayer(std::unique_ptr<Layer>& layer,
    const util::PropertyTree& specification, const util::PropertyTree& submodules,
    const Dimension& size, const TypeMap& types)
{
    SubgraphLayer& subgraph = dynamic_cast<SubgraphLayer&>(*layer);

    Dimension inputSize = size;

    util::log("ModelSpecification") << "  Building subgraph layer with input size "
        << inputSize.toString() << "\n";

    // Add submodules
    for(auto& submodule : submodules)
    {
        auto layerName = submodule.key();
        auto type = submodule.get<std::string>("Type");

        auto layerType = types.find(type);

        if(layerType == types.end())
        {
            throw std::runtime_error("Unknown layer type name '" + type + "'.");
        }

        auto layerParameters = layerType->second;

        appendInputSize(layerParameters, inputSize);

        auto layerTypeName = layerType->second.get<std::string>("Type");

        util::log("ModelSpecification") << "   Building sublayer '" << layerName <<
            "' with input size " << inputSize.toString() << " and parameters '" <<
            layerParameters.toString() << "'\n";

        auto createdLayer = network::LayerFactory::create(layerTypeName, layerParameters);

        if(!createdLayer)
        {
            throw std::runtime_error("Failed to create layer type name '" + layerName +
                "' with parameters '" + layerType->second.toString() + "'");
        }

        if(layerTypeName == "SubgraphLayer")
        {
            auto submodules = specification.get("layer-types." + layerName + ".Submodules");

            populateSubgraphLayer(createdLayer, specification, submodules, inputSize, types);
        }

        subgraph.addLayer(layerName, std::move(createdLayer));

        inputSize = subgraph.getOutputSize();
    }

    // Connect submodules
    for(auto& submodule : submodules)
    {
        auto name = submodule.key();

        if(submodule.exists("ForwardConnections"))
        {
            auto forwardConnections = submodule.get("ForwardConnections");

            for(auto& forwardConnection : forwardConnections)
            {
                subgraph.addForwardConnection(name, forwardConnection.key());
            }
        }

        if(submodule.exists("TimeConnections"))
        {
            auto timeConnections = submodule.get("TimeConnections");

            for(auto& timeConnection : timeConnections)
            {
                subgraph.addTimeConnection(name, timeConnection.key());
            }
        }
    }

    subgraph.prepareSubgraphForEvaluation();
}

static void loadNetwork(Model& model, const util::PropertyTree& network,
    const util::PropertyTree& specification, const TypeMap& types)
{
    network::NeuralNetwork neuralNetwork;

    auto layers = network.get("layers");

    auto inputSize = computeInputSize(specification);

    util::log("ModelSpecification") << "Building network with input size "
        << inputSize.toString() << "\n";

    for(auto& layer : layers)
    {
        auto layerName = layer.key();

        auto layerType = types.find(layerName);

        if(layerType == types.end())
        {
            throw std::runtime_error("Unknown layer type name '" + layerName + "'.");
        }

        auto layerParameters = layerType->second;

        appendInputSize(layerParameters, inputSize);

        bool isLastLayer = neuralNetwork.size() == (layers.size() - 1);

        if(isLastLayer)
        {
            setupOutputLayerParameters(model, layerParameters, specification);
        }

        auto layerTypeName = layerType->second.get<std::string>("Type");

        util::log("ModelSpecification") << " Building layer '" << layerName << "' with input size "
            << inputSize.toString() << " and parameters '" << layerParameters.toString() << "'\n";

        auto createdLayer = network::LayerFactory::create(layerTypeName, layerParameters);

        if(!createdLayer)
        {
            throw std::runtime_error("Failed to create layer type name '" + layerName +
                "' with parameters '" + layerType->second.toString() + "'");
        }

        if(layerTypeName == "SubgraphLayer")
        {
            auto submodules = specification.get("layer-types." + layerName + ".Submodules");

            populateSubgraphLayer(createdLayer, specification, submodules, inputSize, types);
        }

        neuralNetwork.addLayer(std::move(createdLayer));

        inputSize = neuralNetwork.getOutputSize();
    }

    util::log("ModelSpecification") << " Created network \n" << neuralNetwork.shapeString();

    neuralNetwork.initialize();

    model.setNeuralNetwork(network.key(), neuralNetwork);
}

static void loadType(TypeMap& types, const util::PropertyTree& type)
{
    if(types.count(type.key()) != 0)
    {
        throw std::runtime_error("Duplicate layer type name '" + type.key() + "'.");
    }

    util::ParameterPack pack;

    for(auto& property : type)
    {
        if(property.key() == "Submodules")
        {
            continue;
        }

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
        loadNetwork(model, network, specification, types);
    }
}

static void loadCostFunction(Model& model, const util::PropertyTree& specification)
{
    auto name = specification.get<std::string>("cost-function.name");

    auto& network = model.getNeuralNetwork("Classifier");

    auto costFunction = network::CostFunctionFactory::create(name);

    if(costFunction == nullptr)
    {
        throw std::runtime_error("Failed to create neural network cost function '" + name + "'.");
    }

    network.setCostFunction(costFunction);
}

void ModelSpecificationImplementation::initializeModel(Model& model)
{
    checkSections(_specification);

    loadModelAttributes(model, _specification);
    loadNetworks(model, _specification);
    loadCostFunction(model, _specification);
}

}

}

