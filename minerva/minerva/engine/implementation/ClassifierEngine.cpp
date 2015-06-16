/*  \file   ClassifierEngine.cpp
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the ClassifierEngine class.
*/

// Minerva Includes
#include <minerva/engine/interface/ClassifierEngine.h>

#include <minerva/network/interface/NeuralNetwork.h>

#include <minerva/model/interface/Model.h>

#include <minerva/results/interface/ResultProcessorFactory.h>
#include <minerva/results/interface/LabelMatchResult.h>
#include <minerva/results/interface/LabelResult.h>
#include <minerva/results/interface/ResultVector.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixTransformations.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace engine
{

ClassifierEngine::ClassifierEngine()
: _shouldUseLabeledData(true)
{
    setResultProcessor(results::ResultProcessorFactory::create("LabelMatchResultProcessor"));
    setEpochs(1);
}

ClassifierEngine::~ClassifierEngine()
{

}

void ClassifierEngine::setUseLabeledData(bool shouldUse)
{
    _shouldUseLabeledData = shouldUse;
}

util::StringVector convertActivationsToLabels(matrix::Matrix&& activations,
    const model::Model& model)
{
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

results::ResultVector compareWithReference(const util::StringVector& labels,
    const util::StringVector& references)
{
    results::ResultVector result;

    for(auto label = labels.begin(), reference = references.begin(); label != labels.end(); ++reference, ++label)
    {
        result.push_back(new results::LabelMatchResult(*label, *reference));
    }

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

ClassifierEngine::ResultVector ClassifierEngine::runOnBatch(Matrix&& input, Matrix&& reference)
{
    auto network = getAggregateNetwork();

    auto result = network->runInputs(std::move(input));

    auto labels = convertActivationsToLabels(std::move(result), *getModel());

    restoreAggregateNetwork();

    if(_shouldUseLabeledData)
    {
        return compareWithReference(labels, convertActivationsToLabels(std::move(reference), *getModel()));
    }

    return recordLabels(labels);
}

bool ClassifierEngine::requiresLabeledData() const
{
    return _shouldUseLabeledData;
}

}

}


