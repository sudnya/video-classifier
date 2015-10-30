/*  \file   BatchNormalizationLayer.cpp
    \author Gregory Diamos
    \date   September 23, 2015
    \brief  The interface for the BatchNormalizationLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/BatchNormalizationLayer.h>

#include <lucius/network/interface/ActivationFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/FileOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/util/interface/debug.h>
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
: BatchNormalizationLayer({inputs, 1, 1})
{

}

BatchNormalizationLayer::BatchNormalizationLayer(const Dimension& size)
: BatchNormalizationLayer(size, matrix::Precision::getDefaultPrecision())
{

}

BatchNormalizationLayer::BatchNormalizationLayer(const Dimension& size,
    const matrix::Precision& precision)
: _parameters(new MatrixVector({Matrix({size[0]}, precision), Matrix({size[0]}, precision)})),
  _gamma((*_parameters)[0]), _beta((*_parameters)[1]),
  _internal_parameters(new MatrixVector({Matrix({size[0]}, precision),
    Matrix({size[0]}, precision)})),
  _means((*_internal_parameters)[0]), _variances((*_internal_parameters)[1]),
  _samples(0),
  _inputSize(std::make_unique<matrix::Dimension>(size))
{

}

BatchNormalizationLayer::~BatchNormalizationLayer()
{

}

BatchNormalizationLayer::BatchNormalizationLayer(const BatchNormalizationLayer& l)
: _parameters(std::make_unique<MatrixVector>(*l._parameters)),
  _gamma((*_parameters)[0]), _beta((*_parameters)[1]),
  _internal_parameters(std::make_unique<MatrixVector>(*l._internal_parameters)),
  _means((*_internal_parameters)[0]), _variances((*_internal_parameters)[1]),
  _samples(l._samples),
  _inputSize(std::make_unique<matrix::Dimension>(*l._inputSize))
{

}

BatchNormalizationLayer& BatchNormalizationLayer::operator=(const BatchNormalizationLayer& l)
{
    if(&l == this)
    {
        return *this;
    }

    _parameters = std::move(std::make_unique<MatrixVector>(*l._parameters));
    _inputSize  = std::move(std::make_unique<matrix::Dimension>(*l._inputSize));

    _internal_parameters = std::move(std::make_unique<MatrixVector>(*l._internal_parameters));
    _samples             = l._samples;

    return *this;
}

void BatchNormalizationLayer::initialize()
{
    ones(_gamma);
    zeros(_beta);

    zeros(_means);
    ones(_variances);

    _samples = 0;
}

static Matrix foldTime(const Matrix& input)
{
    auto size = input.size();

    size_t batch = 1;

    for(size_t i = 1; i < size.size(); ++i)
    {
        batch *= size[i];
    }

    return reshape(input, {size[0], batch});
}

static Matrix unfoldTime(const Matrix& result, const Dimension& inputSize)
{
    return reshape(result, inputSize);
}

static Matrix computeMeans(const Matrix& activations)
{
    return apply(reduce(activations, {1}, matrix::Add()),
        matrix::Multiply(1.0 / activations.size()[1]));
}

static Matrix computeVariances(const Matrix& activations, const Matrix& means)
{
    auto differences = broadcast(activations, means, {1}, matrix::Subtract());
    auto sumOfSquareDifferences = reduce(apply(differences, matrix::Square()), {1},
        matrix::Add());

    return apply(sumOfSquareDifferences, matrix::Multiply(1.0 / (activations.size()[1])));
}

static void updateMeansAndVariances(Matrix& means, Matrix& variances,
    Matrix& onlineMeans, Matrix& onlineVariances,
    const Matrix& activations, size_t& existingSamples)
{
    // Xb
    means = computeMeans(activations);
    variances = computeVariances(activations, means);

    size_t totalSamples = existingSamples + activations.size()[1];

    // delta = Xb - Xa
    auto deltas = apply(Matrix(means), onlineMeans, matrix::Subtract());

    // Xx
    apply(onlineMeans, onlineMeans, apply(deltas,
        matrix::Multiply(static_cast<double>(activations.size()[1]) / totalSamples)),
        matrix::Add());

    // M2b
    auto differences = broadcast(activations, means, {1}, matrix::Subtract());
    auto sumOfSquareDifferences = reduce(apply(differences, matrix::Square()), {1},
        matrix::Add());

    auto deltaUpdate = apply(apply(deltas, matrix::Square()),
        matrix::Multiply(
            static_cast<double>(existingSamples) * activations.size()[1] / totalSamples));

    // M2a
    auto onlineSumOfSquareDifferences = apply(onlineVariances,
        matrix::Multiply(existingSamples));

    apply(sumOfSquareDifferences, sumOfSquareDifferences, onlineSumOfSquareDifferences,
        matrix::Add());
    apply(sumOfSquareDifferences, sumOfSquareDifferences, deltaUpdate,
        matrix::Add());

    apply(onlineVariances, sumOfSquareDifferences, matrix::Multiply(1.0 / (totalSamples)));

    existingSamples = totalSamples;
}

static Matrix normalize(const Matrix& activations, const Matrix& means, const Matrix& variances)
{
    auto shifted = broadcast(activations, means, {1}, matrix::Subtract());

    auto deviation = apply(apply(variances, matrix::Add(std::numeric_limits<double>::epsilon())),
        matrix::Sqrt());

    return broadcast(Matrix(shifted), deviation, {1}, matrix::Divide());
}

static Matrix scaleAndShift(const Matrix& activations, const Matrix& gamma, const Matrix& beta)
{
    return broadcast(broadcast(activations, gamma, {1}, matrix::Multiply()),
        beta, {1}, matrix::Add());
}

void BatchNormalizationLayer::runForwardImplementation(MatrixVector& activations)
{
    auto inputActivations = foldTime(activations.back());

    util::log("BatchNormalizationLayer") << " Running forward propagation of matrix "
        << inputActivations.shapeString() << " through batch normalization: "
        << _gamma.shapeString() << "\n";

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " input: "
            << inputActivations.debugString();
    }

    Matrix means;
    Matrix variances;

    if(isTraining())
    {
        updateMeansAndVariances(means, variances, _means,
            _variances, inputActivations, _samples);
    }
    else
    {
        means     = _means;
        variances = _variances;
    }

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " means: "
            << means.debugString();
        util::log("BatchNormalizationLayer::Detail") << " variances:  "
            << variances.debugString();
        util::log("BatchNormalizationLayer::Detail") << " online means: "
            << _means.debugString();
        util::log("BatchNormalizationLayer::Detail") << " online variances:  "
            << _variances.debugString();
    }

    auto normalizedActivations = normalize(inputActivations, means, variances);

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " normalized-inputs: "
            << normalizedActivations.debugString();
        util::log("BatchNormalizationLayer::Detail") << " gamma: "
            << _gamma.debugString();
        util::log("BatchNormalizationLayer::Detail") << " beta: "
            << _beta.debugString();
    }

    auto outputActivations = unfoldTime(getActivationFunction()->apply(
        scaleAndShift(normalizedActivations, _gamma, _beta)), activations.back().size());

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " outputs: "
            << outputActivations.debugString();
    }

    activations.push_back(std::move(outputActivations));
}

Matrix BatchNormalizationLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& activations,
    const Matrix& deltasWithTime)
{
    assert(isTraining());

    // Get the output activations
    auto outputActivations = foldTime(activations.back());

    // deallocate memory for the output activations
    activations.pop_back();

    // Get the input activations and deltas
    auto inputActivations = foldTime(activations.back());
    auto deltas = apply(getActivationFunction()->applyDerivative(outputActivations),
        foldTime(deltasWithTime), matrix::Multiply());

    // Get sizes
    size_t miniBatchSize = outputActivations.size()[1];

    // Compute means/variances
    auto means     = computeMeans(inputActivations);
    auto variances = computeVariances(inputActivations, means);

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " means: "
            << means.debugString();
        util::log("BatchNormalizationLayer::Detail") << " variances:  "
            << variances.debugString();
    }


    // Compute derivatives:

    //  dl/dx^ = gamma * dl/dy
    auto xHatDeltas = broadcast(deltas, _gamma, {1}, matrix::Multiply());

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " dl/dx^: "
            << xHatDeltas.debugString();
    }

    //  inputMinusMean = input - mean
    auto inputMinusMean = broadcast(inputActivations, means, {1}, matrix::Subtract());

    // variancePlusEpsilon = variance + epsilon
    auto variancePlusEpsilon = apply(variances,
        matrix::Add(std::numeric_limits<double>::epsilon()));

    //  dl/dVariance = sum_mini_batch(dl/dx^ * (inputMinusMean)) * (-1.0/2.0) * (variance + epsilon)^(-3.0/2.0)
    auto variancePlusEpsilonPowHalf = apply(apply(variancePlusEpsilon, matrix::Pow(-3.0/2.0)),
        matrix::Multiply(-1.0/2.0));

    auto inputMinusMeanTimesXHatDeltas = reduce(
        apply(Matrix(inputMinusMean), xHatDeltas, matrix::Multiply()), {1}, matrix::Add());

    auto varianceDeltas = apply(Matrix(variancePlusEpsilonPowHalf), inputMinusMeanTimesXHatDeltas,
        matrix::Multiply());

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " dl/dVariance: "
            << varianceDeltas.debugString();
    }

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

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " dl/dMean: "
            << meanDeltas.debugString();
    }

    // dl/dx = dl/dx^ * 1.0 / sqrt(variance + epsilon) +
    //         dl/dVariance * 2.0 (x - mean) / miniBatchSize +
    //         dl/dmean * 1.0 / miniBatchSize

    auto leftInputDeltas = broadcast(xHatDeltas,
        apply(sqrtVariancePlusEpsilon, matrix::Inverse()), {1}, matrix::Multiply());

    auto middleInputDeltas = broadcast(
        apply(inputMinusMean, matrix::Multiply(2.0 / miniBatchSize)),
        varianceDeltas, {1}, matrix::Multiply());

    auto rightInputDeltas = apply(meanDeltas, matrix::Multiply(1.0 / miniBatchSize));

    auto inputDeltas = broadcast(apply(Matrix(leftInputDeltas), middleInputDeltas, matrix::Add()),
        rightInputDeltas, {1}, matrix::Add());

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " dl/dx: "
            << inputDeltas.debugString();
    }

    // dl/dGamma = sum_over_mini_batch(xHat) * dl/dy
    auto gammaDeltas = apply(reduce(apply(Matrix(deltas), outputActivations, matrix::Multiply()),
        {1}, matrix::Add()), matrix::Multiply(1.0 / miniBatchSize));

    gradients.push_back(gammaDeltas);

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " dl/dGamma: "
            << gammaDeltas.debugString();
    }

    // dl/dBeta = sum_over_mini_batch(dl/dy)
    auto betaDeltas = apply(reduce(deltas, {1}, matrix::Add()),
        matrix::Multiply(1.0 / miniBatchSize));

    gradients.push_back(betaDeltas);

    if(util::isLogEnabled("BatchNormalizationLayer::Detail"))
    {
        util::log("BatchNormalizationLayer::Detail") << " dl/dBeta: "
            << betaDeltas.debugString();
    }

    return unfoldTime(inputDeltas, deltasWithTime.size());
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
    return *_inputSize;
}

Dimension BatchNormalizationLayer::getOutputSize() const
{
    return getInputSize();
}

size_t BatchNormalizationLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t BatchNormalizationLayer::getOutputCount() const
{
    return getInputCount();
}

size_t BatchNormalizationLayer::totalNeurons() const
{
    return _gamma.elements();
}

size_t BatchNormalizationLayer::totalConnections() const
{
    return totalNeurons();
}

size_t BatchNormalizationLayer::getFloatingPointOperationCount() const
{
    return 2 * totalConnections();
}

void BatchNormalizationLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    properties["gamma"] = properties.path() + "." + properties.key() + ".gamma.npy";
    properties["beta"]  = properties.path() + "." + properties.key() + ".beta.npy";
    properties["means"] = properties.path() + "." + properties.key() + ".means.npy";

    properties["variances"]  = properties.path() + "." + properties.key() + ".variances.npy";
    properties["samples"]    = _samples;
    properties["input-size"] = _inputSize->toString();

    saveToArchive(archive, properties["gamma"], _gamma);
    saveToArchive(archive, properties["beta"],  _beta);

    saveToArchive(archive, properties["means"], _means);
    saveToArchive(archive, properties["variances"], _variances);

    saveLayer(archive, properties);
}

void BatchNormalizationLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    _gamma = matrix::loadFromArchive(archive, properties["gamma"]);
    _beta  = matrix::loadFromArchive(archive, properties["beta"]);

    *_inputSize = Dimension::fromString(properties["input-size"]);

    _means = matrix::loadFromArchive(archive, properties["means"]);
    _variances = matrix::loadFromArchive(archive, properties["variances"]);
    _samples = properties.get<size_t>("samples");

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> BatchNormalizationLayer::clone() const
{
    return std::make_unique<BatchNormalizationLayer>(*this);
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




