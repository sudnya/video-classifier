/*  \file   ModelBuilder.cpp
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ModelBuilder class.
*/

// Lucius Includes
#include <lucius/model/interface/ModelBuilder.h>
#include <lucius/model/interface/Model.h>
#include <lucius/model/interface/ModelSpecification.h>
#include <lucius/model/interface/BuiltInSpecifications.h>

#include <lucius/network/interface/NeuralNetwork.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace model
{

typedef lucius::network::NeuralNetwork NeuralNetwork;
typedef lucius::network::Layer Layer;
typedef lucius::matrix::Matrix Matrix;

static void initializeModelFromSpecification(Model* model, const std::string& specification)
{
    ModelSpecification modelSpecification;

    modelSpecification.parseSpecification(specification);

    modelSpecification.initializeModel(*model);
}

static void buildConvolutionalFastModel(Model* model, size_t outputs)
{
    auto specification = BuiltInSpecifications::getConvolutionalFastModelSpecification(outputs);

    initializeModelFromSpecification(model, specification);
}

std::unique_ptr<Model> ModelBuilder::create(const std::string& path)
{
    auto model = std::make_unique<Model>(path);

    size_t x      = util::KnobDatabase::getKnobValue("ModelBuilder::ResolutionX",     32);
    size_t y      = util::KnobDatabase::getKnobValue("ModelBuilder::ResolutionY",     32);
    size_t colors = util::KnobDatabase::getKnobValue("ModelBuilder::ColorComponents", 3 );

    model->setAttribute("ResolutionX",     x     );
    model->setAttribute("ResolutionY",     y     );
    model->setAttribute("ColorComponents", colors);

    size_t classifierOutputSize = util::KnobDatabase::getKnobValue(
        "Classifier::NeuralNetwork::Outputs", 1);

    // (FastModel, ConvolutionalCPUModel, ConvolutionalGPUModel)
    auto modelType = util::KnobDatabase::getKnobValue("ModelType", "ConvolutionalFastModel");

    util::log("ModelBuilder") << "Creating ...\n";

    if(modelType == "ConvolutionalFastModel")
    {
        buildConvolutionalFastModel(model.get(), classifierOutputSize);
    }
    else
    {
        throw std::runtime_error("Unknown model named " + modelType);
    }

    return model;
}

std::unique_ptr<Model> ModelBuilder::create(const std::string& path,
    const std::string& specification)
{
    auto model = std::make_unique<Model>(path);

    initializeModelFromSpecification(model.get(), specification);

    return model;
}

}

}


