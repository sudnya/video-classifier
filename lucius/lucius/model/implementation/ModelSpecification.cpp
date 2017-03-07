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
#include <lucius/network/interface/CostFunction.h>
#include <lucius/network/interface/LayerFactory.h>
#include <lucius/network/interface/SubgraphLayer.h>
#include <lucius/network/interface/ControllerLayer.h>
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
typedef network::ControllerLayer ControllerLayer;
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

database::SampleDatabase::StringVector getAllPossibleGraphemes(const util::PropertyTree& specification)
{
    database::SampleDatabase::StringVector labels;

    if(!specification.exists("infer-outputs-from"))
    {
        if(specification.exists("model-attributes.Graphemes"))
        {
            for(auto& grapheme : specification.get("model-attributes.Graphemes"))
            {
                labels.push_back(grapheme.key());
            }
        }

        return labels;
    }

    auto datasetPath = specification.get<std::string>("infer-outputs-from");

    database::SampleDatabase inputDatabase(datasetPath);

    if(specification.exists("model-attributes.Graphemes"))
    {
        for(auto& grapheme : specification.get("model-attributes.Graphemes"))
        {
            inputDatabase.addGrapheme(grapheme.key());
        }
    }

    inputDatabase.load();

    labels = inputDatabase.getAllPossibleLabels();

    return labels;
}

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
    else if(attributes.exists("SegmentSize"))
    {
        result.push_back(getAllPossibleGraphemes(model).size());
        result.push_back(1);
        result.push_back(1);
        result.push_back(1); // minibatch
        result.push_back(1); // timesteps

    }
    else
    {
        throw std::runtime_error("Unknown input data type.");
    }

    return result;
}

static void appendInputSize(util::ParameterPack& parameters, const matrix::Dimension& inputSize)
{
    if(inputSize.size() > 0)
    {
        parameters.insert("InputSizeWidth", inputSize[0]);
    }

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
        parameters.insert("InputSizeBatch", inputSize[inputSize.size() - 2]);
    }

    if(inputSize.size() > 2)
    {
        parameters.insert("InputSizeTimesteps", inputSize[inputSize.size() - 1]);
    }
}

static void setupOutputLayerParameters(model::Model& model,
    util::ParameterPack& layerParameters, const util::PropertyTree& specification)
{
    auto labels = getAllPossibleGraphemes(specification);
    size_t index = 0;

    if(specification.exists("model-attributes.Graphemes"))
    {
        model.setAttribute("UsesGraphemes", "1");
    }

    bool needsSeparator = specification.exists("model-attributes.Graphemes") &&
        ((specification.exists("cost-function") &&
        specification.get<std::string>("cost-function") == "CTCCostFunction") ||
        (specification.exists("model-attributes.UsesSeparatorToken") &&
         specification.get<bool>("model-attributes.UsesSeparatorToken")));

    bool needsUnknown = specification.exists("model-attributes.Graphemes") &&
        ((specification.exists("cost-function") &&
        specification.get<std::string>("cost-function") == "SoftmaxCostFunction") ||
        (specification.exists("model-attributes.UsesUnknownToken") &&
        specification.get<bool>("model-attributes.UsesUnknownToken")));


    // The 0th network output must be a separator for CTC
    if(needsSeparator)
    {
        model.setOutputLabel(index++, "-SEPARATOR-");
    }

    if(needsUnknown)
    {
        model.setOutputLabel(index++, "-UNKOWN-");
    }

    for(auto& label : labels)
    {
        model.setOutputLabel(index++, label);
    }

    if(needsSeparator)
    {
        labels.push_back("-SEPARATOR-");
    }

    if(needsUnknown)
    {
        labels.push_back("-UNKOWN-");
    }

    layerParameters.insert("OutputSize", labels.size());
}

static std::unique_ptr<Layer> createLayer(const util::PropertyTree& layerType,
    const util::PropertyTree& layerConfiguration,
    const util::ParameterPack& layerParameters)
{
    auto layerTypeName = layerConfiguration.get<std::string>("Type");

    util::log("ModelSpecification") << " Building layer '" << layerTypeName
        << "' with parameters '" << layerParameters.toString() << "'\n";

    auto createdLayer = network::LayerFactory::create(layerTypeName, layerParameters);

    if(!createdLayer)
    {
        throw std::runtime_error("Failed to create layer type name '" + layerTypeName +
            "' with parameters '" + layerParameters.toString() + "'");
    }

    return createdLayer;
}

static util::ParameterPack getLayerParameters(const util::PropertyTree& layer,
    const TypeMap& types)
{
    auto layerName = layer.key();

    auto layerType = types.find(layerName);

    if(layerType == types.end())
    {
        throw std::runtime_error("Unknown layer type name '" + layerName + "'.");
    }

    return layerType->second;
}

static std::unique_ptr<Layer> createLayer(Model& model, const util::PropertyTree& layer,
    const util::PropertyTree& specification, const Dimension& size, const TypeMap& types,
    bool isLastLayer)
{
    auto layerParameters = getLayerParameters(layer, types);

    appendInputSize(layerParameters, size);

    if(isLastLayer)
    {
        setupOutputLayerParameters(model, layerParameters, specification);
    }

    auto layerConfiguration = specification.get("layer-types." + layer.key());

    return createLayer(layer, layerConfiguration, layerParameters);
}

static std::unique_ptr<Layer> createLayer(const util::PropertyTree& layer,
    const util::PropertyTree& specification,
    const Dimension& size, const TypeMap& types)
{
    util::ParameterPack parameters = getLayerParameters(layer, types);

    appendInputSize(parameters, size);

    auto layerConfiguration = specification.get("layer-types." + layer.key());

    return createLayer(layer, layerConfiguration, parameters);
}

static void populateLayer(std::unique_ptr<Layer>& layer,
    const util::PropertyTree& specification,
    const util::PropertyTree& layerProperties,
    const Dimension& size, const TypeMap& types);

static void populateSubgraphLayer(std::unique_ptr<Layer>& layer,
    const util::PropertyTree& specification,
    const util::PropertyTree& layerProperties,
    const Dimension& size, const TypeMap& types)
{
    SubgraphLayer& subgraph = dynamic_cast<SubgraphLayer&>(*layer);

    util::log("ModelSpecification") << "  Building subgraph layer with input size "
        << size.toString() << "\n";

    auto submodules = specification.get("layer-types." + layerProperties.key() +
        ".Submodules");

    auto inputSize = size;

    // Add submodules
    for(auto& submodule : submodules)
    {
        auto createdLayer = createLayer(submodule, specification, inputSize, types);

        inputSize = Dimension();

        populateLayer(createdLayer, specification, submodule, inputSize, types);
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

static void populateController(std::unique_ptr<Layer>& layer,
    const util::PropertyTree& specification,
    const util::PropertyTree& layerProperties,
    const Dimension& size, const TypeMap& types)
{
    ControllerLayer& layerWithController = dynamic_cast<ControllerLayer&>(*layer);

    if(!specification.exists("layer-types." + layerProperties.key() + ".Controller"))
    {
        throw std::runtime_error("No 'Controller' field in layer type '" +
            layerProperties.key() + "'.");
    }

    auto controller = specification.get("layer-types." + layerProperties.key()
        + ".Controller").get();

    layerWithController.setController(createLayer(controller, specification, size, types));
}

static bool hasController(const std::string& layerType)
{
    return (layerType == "MemoryReaderLayer") || (layerType == "MemoryWriterLayer");
}

static void populateLayer(std::unique_ptr<Layer>& layer,
    const util::PropertyTree& specification,
    const util::PropertyTree& layerProperties,
    const Dimension& size, const TypeMap& types)
{
    if(!specification.exists("layer-types." + layerProperties.key()))
    {
        throw std::runtime_error("No layer type name '" + layerProperties.key() + "'.");
    }

    auto layerConfiguration = specification.get("layer-types." + layerProperties.key());

    auto layerTypeName = layerConfiguration.get<std::string>("Type");

    if(layerTypeName == "SubgraphLayer")
    {
        populateSubgraphLayer(layer, specification, layerProperties, size, types);
    }
    else if(hasController(layerTypeName))
    {
        populateController(layer, specification, layerProperties, size, types);
    }
}

static Dimension loadSubcomponent(Model& model, network::NeuralNetwork& neuralNetwork,
    const util::PropertyTree& layer, const util::PropertyTree& specification,
    const Dimension& size, const TypeMap& types, bool isLastLayer)
{
    auto createdLayer = createLayer(model, layer, specification, size, types, isLastLayer);

    populateLayer(createdLayer, specification, layer, size, types);

    neuralNetwork.addLayer(std::move(createdLayer));

    return neuralNetwork.getOutputSize();
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
        bool isLastLayer = neuralNetwork.size() == (layers.size() - 1);

        inputSize = loadSubcomponent(model, neuralNetwork, layer, specification,
            inputSize, types, isLastLayer);
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

    if(!costFunction)
    {
        throw std::runtime_error("Failed to create neural network cost function '" + name + "'.");
    }

    network.setCostFunction(std::move(costFunction));
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

