/*  \file   BatchNormalizationLayer.cpp
    \author Gregory Diamos
    \date   September 23, 2015
    \brief  The interface for the BatchNormalizationLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/BatchNormalizationLayer.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/FileOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/PropertyTree.h>

namespace lucius
{
namespace network
{

typedef matrix::Matrix       Matrix;
typedef matrix::Dimension    Dimension;
typedef matrix::MatrixVector MatrixVector;

BatchNormalizationLayer::BatchNormalizationLayer()
: BatchNormalizationLayer(0)
{

}

BatchNormalizationLayer::BatchNormalizationLayer(size_t inputs)
: BatchNormalizationLayer(inputs, matrix::Precision::getDefaultPrecision())
{

}

BatchNormalizationLayer::BatchNormalizationLayer(size_t inputs,
    const matrix::Precision& precision)
: _parameters(new MatrixVector({Matrix({inputs}, precision), Matrix({inputs}, precision)})),
  _gamma((*_parameters)[0]), _beta((*_parameters)[1]),
  _internal_parameters(new MatrixVector({Matrix({inputs}, precision),
    Matrix({inputs}, precision), Matrix({inputs}, precision)})),
  _means((*_internal_parameters)[0]), _variances((*_internal_parameters)[1]),
  _sumOfSquaresOfDifferences((*_internal_parameters)[2]), _samples(0)
{

}
BatchNormalizationLayer::~BatchNormalizationLayer()
{

}

BatchNormalizationLayer::BatchNormalizationLayer(const BatchNormalizationLayer& l)
: _parameters(std::make_unique<MatrixVector>(*l._parameters)),
  _gamma((*_parameters)[0]), _beta((*_parameters)[1]),
  _internal_parameters(std::make_unique<MatrixVector>(*l._parameters)),
  _means((*_internal_parameters)[0]), _variances((*_internal_parameters)[1]),
  _sumOfSquaresOfDifferences((*_internal_parameters)[2]), _samples(l._samples)
{

}

BatchNormalizationLayer& BatchNormalizationLayer::operator=(const BatchNormalizationLayer& l)
{
    if(&l == this)
    {
        return *this;
    }

    _parameters = std::move(std::make_unique<MatrixVector>(*l._parameters));
    _internal_parameters = std::move(std::make_unique<MatrixVector>(*l._internal_parameters));
    _samples = l._samples;

    return *this;
}

void BatchNormalizationLayer::initialize()
{
    ones(_gamma);
    zeros(_beta);

    zeros(_means);
    zeros(_variances);
    zeros(_sumOfSquaresOfDifferences);

    _samples = 0;
}

static Matrix foldTime(const Matrix& input)
{
    assert(input.size().size() < 4);

    if(input.size().size() == 3)
    {
        auto size = input.size();
        size_t timesteps = size.back();

        size.pop_back();

        size.back() *= timesteps;

        return reshape(input, size);
    }

    return input;
}

static Matrix unfoldTime(const Matrix& result, const Dimension& inputSize)
{
    if(inputSize.size() <= 2)
    {
        return result;
    }

    assert(inputSize.size() == 3);

    size_t layerSize = result.size()[0];
    size_t miniBatch = inputSize[1];
    size_t timesteps = inputSize[2];

    return reshape(result, {layerSize, miniBatch, timesteps});
}

static void computeMeansAndVariances(Matrix& means, Matrix& variances,
    Matrix& sumOfSquareDifferences, const Matrix& activations, size_t existingSamples)
{
    size_t samples = existingSamples + activations.size()[1];

    auto newMeans = apply(reduce(activations, {1}, matrix::Add()),
        matrix::Multiply(1.0 / activations.size()[1]));

    auto deltas = apply(Matrix(newMeans), means, matrix::Subtract());

    apply(means, means, apply(deltas,
        matrix::Multiply(static_cast<double>(existingSamples) / samples)), matrix::Add());

    auto newDifferences = broadcast(activations, newMeans, {1}, matrix::Subtract());
    auto newSumOfSquareDifferences = reduce(apply(newDifferences, matrix::Square()), {1},
        matrix::Square());

    auto deltaUpdate = apply(apply(deltas, matrix::Square()),
        matrix::Multiply(static_cast<double>(existingSamples) * activations.size()[1] / samples));

    apply(sumOfSquareDifferences, sumOfSquareDifferences, newSumOfSquareDifferences,
        matrix::Add());
    apply(sumOfSquareDifferences, sumOfSquareDifferences, deltaUpdate,
        matrix::Add());

    apply(variances, sumOfSquareDifferences, matrix::Multiply(1.0 / (samples - 1)));
}

static Matrix normalize(const Matrix& activations, const Matrix& means, const Matrix& variances)
{
    auto shifted = broadcast(activations, means, {1}, matrix::Subtract());

    auto deviation = apply(apply(variances, matrix::Add(std::numeric_limits<double>::epsilon())),
        matrix::Square());

    return apply(Matrix(shifted), deviation, matrix::Divide());
}

static Matrix scaleAndShift(const Matrix& activations, const Matrix& gamma, const Matrix& beta)
{
    return broadcast(broadcast(activations, gamma, {1}, matrix::Multiply()),
        beta, {1}, matrix::Add());
}

void BatchNormalizationLayer::runForwardImplementation(MatrixVector& activations) const
{
    auto inputActivations = foldTime(activations.back());

    computeMeansAndVariances(_means, _variances, _sumOfSquaresOfDifferences,
        inputActivations, _samples);

    auto normalizedActivations = normalize(inputActivations, _means, _variances);

    auto outputActivations = unfoldTime(scaleAndShift(normalizedActivations, _gamma, _beta),
        activations.back().size());

    activations.push_back(std::move(activations));
}

Matrix BatchNormalizationLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& activations,
    const Matrix& deltasWithTime) const
{

    // Get the output activations
    auto outputActivations = foldTime(activations.back());

    // deallocate memory for the output activations
    activations.pop_back();

    // Get the input activations and deltas
    auto inputActivations = foldTime(activations.back());
    auto deltas = foldTime(deltasWithTime);

    // Get sizes
    size_t miniBatchSize = outputActivations.size()[1];

    const_cast<size_t&>(_samples) += miniBatchSize;

    // Compute derivatives:

    //  dl/dx^ = gamma * dl/dy
    auto xHatDeltas = broadcast(deltas, _gamma, {1}, matrix::Multiply());

    //  inputMinusMean = input - mean
    auto inputMinusMean = broadcast(inputActivations, _means, {1}, matrix::Subtract());

    // variancePlusEpsilon = variance + epsilon
    auto variancePlusEpsilon = apply(_variances,
        matrix::Add(std::numeric_limits<double>::epsilon()));

    //  dl/dVariance = sum_mini_batch(dl/dx^ * (inputMinusMean) * (-1.0/2.0) * (variance + epsilon)^(-3.0/2.0))
    auto variancePlusEpsilonPowHalf = apply(apply(variancePlusEpsilon, matrix::Pow(-3.0/2.0)),
        matrix::Multiply(-1.0/2.0));

    auto variancePlusEpsilonPowHalfTimesInputMinusMean = broadcast(inputMinusMean,
        variancePlusEpsilonPowHalf, {1}, matrix::Multiply());

    auto varianceDeltas = reduce(
        apply(Matrix(xHatDeltas), variancePlusEpsilonPowHalfTimesInputMinusMean,
        matrix::Multiply()), {1}, matrix::Add());

    // dl/dMean = (-1.0 / sqrt(variance + epsilon)) * sum_over_mini_batch(xHatDeltas) +
    //            d/dVariance * sum_over_mini_batch(-2.0 * inputMinusMean) / miniBatchSize
    auto sqrtVariancePlusEpsilon = apply(variancePlusEpsilon, matrix::Sqrt());

    auto negativeInverseSqrtVariancePlusEpsilon = apply(apply(sqrtVariancePlusEpsilon,
        matrix::Inverse()), matrix::Multiply(-1.0));

    auto leftMeanDeltas = apply(Matrix(negativeInverseSqrtVariancePlusEpsilon),
        reduce(xHatDeltas, {1}, matrix::Add()), matrix::Multiply());

    auto rightMeanDeltas = apply(Matrix(varianceDeltas),
        reduce(apply(inputMinusMean, matrix::Multiply(-2.0 / miniBatchSize)), {1}, matrix::Add()),
        matrix::Multiply());

    auto meanDeltas = apply(Matrix(leftMeanDeltas), rightMeanDeltas, matrix::Add());

    // dl/dx = dl/dx^ * 1.0 / sqrt(variance + epsilon) +
    //         dl/dVariance * 2.0 (x - mean) / miniBatchSize +
    //         dl/dmean * 1.0 / miniBatchSize

    auto leftInputDeltas = broadcast(xHatDeltas,
        apply(sqrtVariancePlusEpsilon, matrix::Inverse()), {1}, matrix::Multiply());

    auto middleInputDeltas = apply(apply(inputMinusMean, matrix::Multiply(2.0 / miniBatchSize)),
        matrix::Multiply());

    auto rightInputDeltas = apply(meanDeltas, matrix::Multiply(1.0 / miniBatchSize));

    auto inputDeltas = broadcast(broadcast(leftInputDeltas, middleInputDeltas, {1}, matrix::Add()),
        rightInputDeltas, {1}, matrix::Add());

    // dl/dBeta = sum_over_mini_batch(dl/dy)
    auto betaDeltas = reduce(deltas, {1}, matrix::Add());

    gradients.push_back(betaDeltas);

    // dl/dGamma = sum_over_mini_batch(xHat) * gamma
    auto gammaDeltas = apply(reduce(inputActivations, {1}, matrix::Add()),
        _gamma, matrix::Multiply());

    gradients.push_back(gammaDeltas);

    return unfoldTime(inputDeltas, outputActivations.size());
}

MatrixVector& BatchNormalizationLayer::weights()
{
    return *_parameters;
}

const MatrixVector& BatchNormalizationLayer::weights() const
{
    return *_parameters;
}

const matrix::Precision& BatchNormalizationLayer::precision() const
{
    return _gamma.precision();
}

double BatchNormalizationLayer::computeWeightCost() const
{
    return 0.0;
}

Dimension BatchNormalizationLayer::getInputSize() const
{
    return {getInputCount(), 1, 1};
}

Dimension BatchNormalizationLayer::getOutputSize() const
{
    return {getOutputCount(), 1, 1};
}

size_t BatchNormalizationLayer::getInputCount() const
{
    return _gamma.elements();
}

size_t BatchNormalizationLayer::getOutputCount() const
{
    return _gamma.elements();
}

size_t BatchNormalizationLayer::totalNeurons() const
{
    return getInputCount();
}

size_t BatchNormalizationLayer::totalConnections() const
{
    return getInputCount();
}

size_t BatchNormalizationLayer::getFloatingPointOperationCount() const
{
    return 2 * getInputCount();
}

void BatchNormalizationLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    properties["gamma"] = properties.path() + "." + properties.key() + ".gamma.npy";
    properties["beta"]  = properties.path() + "." + properties.key() + ".beta.npy";
    properties["means"] = properties.path() + "." + properties.key() + ".means.npy";

    properties["variances"] = properties.path() + "." + properties.key() + ".variances.npy";
    properties["sumOfSquaresOfDifferences"] =
        properties.path() + "." + properties.key() + ".sumOfSquaresOfDifferences.npy";
    properties["samples"] = _samples;

    saveToArchive(archive, properties["gamma"], _gamma);
    saveToArchive(archive, properties["beta"],  _beta);

    saveToArchive(archive, properties["means"], _beta);
    saveToArchive(archive, properties["variances"], _variances);
    saveToArchive(archive, properties["sumOfSquaresOfDifferences"], _sumOfSquaresOfDifferences);

    saveLayer(archive, properties);
}

void BatchNormalizationLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    _gamma = matrix::loadFromArchive(archive, properties["gamma"]);
    _beta  = matrix::loadFromArchive(archive, properties["beta"]);

    _means = matrix::loadFromArchive(archive, properties["beta"]);
    _variances = matrix::loadFromArchive(archive, properties["variances"]);
    _sumOfSquaresOfDifferences =
        matrix::loadFromArchive(archive, properties["sumOfSquaresOfDifferences"]);
    _samples = properties.get<size_t>("samples");

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> BatchNormalizationLayer::clone() const
{
    return std::make_unique<BatchNormalizationLayer>(getInputCount(), precision());
}

std::unique_ptr<Layer> BatchNormalizationLayer::mirror() const
{
    return clone();
}

std::string BatchNormalizationLayer::getTypeName() const
{
    return "BatchNormalizationLayer";
}

}

}




