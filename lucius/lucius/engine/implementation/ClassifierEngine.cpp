/*  \file   ClassifierEngine.cpp
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ClassifierEngine class.
*/

// Lucius Includes
#include <lucius/engine/interface/ClassifierEngine.h>

#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/Bundle.h>

#include <lucius/model/interface/Model.h>

#include <lucius/results/interface/ResultProcessorFactory.h>
#include <lucius/results/interface/ResultProcessor.h>
#include <lucius/results/interface/LabelMatchResult.h>
#include <lucius/results/interface/LabelResult.h>
#include <lucius/results/interface/CostResult.h>
#include <lucius/results/interface/ResultVector.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/CTCOperations.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace engine
{

ClassifierEngine::ClassifierEngine()
: _shouldUseLabeledData(false)
{
    setResultProcessor(results::ResultProcessorFactory::create("LabelMatchResultProcessor"));
    setEpochs(1);
}

ClassifierEngine::~ClassifierEngine()
{

}

void ClassifierEngine::setModel(Model* model)
{
    if(model->hasAttribute("UsesGraphemes") && model->getAttribute<bool>("UsesGraphemes"))
    {
        setResultProcessor(results::ResultProcessorFactory::create(
            "GraphemeMatchResultProcessor"));
    }

    Engine::setModel(model);
}

void ClassifierEngine::setUseLabeledData(bool shouldUse)
{
    _shouldUseLabeledData = shouldUse;
}

static util::StringVector convertActivationsToGraphemeLabels(matrix::Matrix&& activations,
    const model::Model& model)
{
    assert(activations.size().size() >= 3);

    if(activations.size().size() > 3)
    {
        activations = reshape(activations,
            {activations.size().front(),
             activations.size()[1],
             activations.size().product() / (activations.size().front() * activations.size()[1])});
    }

    size_t timesteps     = activations.size()[2];
    size_t miniBatchSize = activations.size()[1];
    size_t graphemes     = activations.size()[0];

    util::StringVector labels;

    for(size_t miniBatch = 0; miniBatch < miniBatchSize; ++miniBatch)
    {
        std::string currentLabel;

        size_t currentGrapheme = graphemes;

        for(size_t timestep = 0; timestep < timesteps; ++timestep)
        {
            size_t maxGrapheme = 0;
            double maxValue    = std::numeric_limits<double>::min();

            for(size_t grapheme = 0; grapheme < graphemes; ++grapheme)
            {
                if(activations(grapheme, miniBatch, timestep) >= maxValue)
                {
                    maxValue    = activations(grapheme, miniBatch, timestep);
                    maxGrapheme = grapheme;
                }
            }

            if(maxGrapheme != currentGrapheme)
            {
                currentGrapheme = maxGrapheme;
                auto newGrapheme = model.getOutputLabel(maxGrapheme);

                currentLabel.insert(currentLabel.end(), newGrapheme.begin(), newGrapheme.end());
            }

        }

        labels.push_back(currentLabel);
    }

    return labels;
}

static util::StringVector convertActivationsToLabels(matrix::Matrix&& activations,
    const model::Model& model)
{
    if(model.hasAttribute("UsesGraphemes") && model.getAttribute<bool>("UsesGraphemes"))
    {
        return convertActivationsToGraphemeLabels(std::move(activations), model);
    }

    if(activations.size().size() > 2)
    {
        activations = reshape(activations,
            {activations.size().front(),
             activations.size().product() / activations.size().front()});
    }

    size_t samples = activations.size()[1];
    size_t columns = activations.size()[0];

    util::StringVector labels;

    for(size_t sample = 0; sample < samples; ++sample)
    {
        size_t maxColumn = 0;
        double maxValue  = 0.0f;

        for(size_t column = 0; column < columns; ++column)
        {
            if(activations(column, sample) >= maxValue)
            {
                maxValue  = activations(column, sample);
                maxColumn = column;
            }
        }

        labels.push_back(model.getOutputLabel(maxColumn));
    }

    return labels;
}

static util::StringVector getReferenceLabels(network::Bundle& bundle,
    const model::Model& model)
{
    if(bundle.contains("referenceLabels"))
    {
        util::StringVector labels;

        auto& referenceLabels = bundle["referenceLabels"].get<matrix::LabelVector>();

        for(auto& referenceLabel : referenceLabels)
        {
            labels.push_back(std::string());

            for(auto& grapheme : referenceLabel)
            {
                labels.back() += model.getOutputLabel(grapheme);
            }
        }

        return labels;
    }

    auto& referenceActivations =
        bundle["referenceActivations"].get<matrix::MatrixVector>().front();

    return convertActivationsToLabels(std::move(referenceActivations), model);
}

results::ResultVector compareWithReference(double cost, size_t iteration,
    const util::StringVector& labels, const util::StringVector& references)
{
    results::ResultVector result;

    for(auto label = labels.begin(), reference = references.begin();
        label != labels.end(); ++reference, ++label)
    {
        result.push_back(new results::LabelMatchResult(*label, *reference));
    }

    result.push_back(new results::CostResult(cost, iteration));

    return result;
}

results::ResultVector recordLabels(const util::StringVector& labels)
{
    results::ResultVector result;

    for(auto label : labels)
    {
        result.push_back(new results::LabelResult(label));
    }

    return result;
}

ClassifierEngine::ResultVector ClassifierEngine::runOnBatch(const Bundle& input)
{
    auto network = getAggregateNetwork();

    network->setIsTraining(false);

    auto bundle = network->runInputs(input);

    auto& result = bundle["outputActivations"].get<matrix::MatrixVector>().front();

    auto labels = convertActivationsToLabels(std::move(result), *getModel());

    ClassifierEngine::ResultVector results;

    if(_shouldUseLabeledData)
    {
        bundle = network->getCost(input);

        auto cost = bundle["cost"].get<double>();

        results = compareWithReference(cost * labels.size(), getIteration(), labels,
            getReferenceLabels(bundle, *getModel()));
    }
    else
    {
        results = recordLabels(labels);
    }

    restoreAggregateNetwork();

    return results;
}

bool ClassifierEngine::requiresLabeledData() const
{
    return _shouldUseLabeledData;
}

}

}


