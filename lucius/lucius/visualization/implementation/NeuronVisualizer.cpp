/*    \file   NeuronVisualizer.cpp
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the NeuronVisualizer class.
*/

// Minvera Includes
#include <lucius/visualization/interface/NeuronVisualizer.h>

#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/Layer.h>

#include <lucius/video/interface/Image.h>

#include <lucius/optimizer/interface/GeneralDifferentiableSolver.h>
#include <lucius/optimizer/interface/GeneralDifferentiableSolverFactory.h>

#include <lucius/optimizer/interface/CostAndGradientFunction.h>
#include <lucius/optimizer/interface/CostFunction.h>
#include <lucius/optimizer/interface/ConstantConstraint.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/math.h>
#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <random>
#include <cstdlib>

namespace lucius
{

namespace visualization
{

typedef matrix::Matrix Matrix;
typedef network::NeuralNetwork NeuralNetwork;
typedef video::Image Image;
typedef optimizer::ConstantConstraint ConstantConstraint;

NeuronVisualizer::NeuronVisualizer(NeuralNetwork* network)
: _network(network)
{

}

static void visualizeNeuron(NeuralNetwork& , Image& , size_t);

void NeuronVisualizer::visualizeNeuron(Image& image, size_t outputNeuron)
{
    visualization::visualizeNeuron(*_network, image, outputNeuron);
}

static void getTileDimensions(size_t& x, size_t& y, size_t& colors,
    NeuralNetwork& tile);
static size_t sqrtRoundUp(size_t);
static size_t getXPixelsPerTile(NeuralNetwork& );
static size_t getYPixelsPerTile(NeuralNetwork& );
static size_t getColorsPerTile(NeuralNetwork& );

Image NeuronVisualizer::visualizeInputTileForNeuron(size_t outputNeuron)
{
    size_t x      = 0;
    size_t y      = 0;
    size_t colors = 0;

    getTileDimensions(x, y, colors, *_network);

    Image image(x, y, colors, 1);

    visualization::visualizeNeuron(*_network, image, outputNeuron);

    return image;
}

Image NeuronVisualizer::visualizeInputTilesForAllNeurons()
{
    assert(_network->getOutputCount() > 0);

    size_t xTiles = sqrtRoundUp(_network->getOutputCount());
    size_t yTiles = sqrtRoundUp(_network->getOutputCount());

    size_t xPixelsPerTile = getXPixelsPerTile(*_network);
    size_t yPixelsPerTile = getYPixelsPerTile(*_network);

    size_t colors = getColorsPerTile(*_network);

    size_t xPadding = std::max(xPixelsPerTile / 8, 1UL);
    size_t yPadding = std::max(yPixelsPerTile / 8, 1UL);

    size_t x = xTiles * (xPixelsPerTile + xPadding);
    size_t y = yTiles * (yPixelsPerTile + yPadding);

    Image image(x, y, colors, 1);

    for(size_t neuron = 0; neuron != _network->getOutputCount(); ++neuron)
    {
        util::log("NeuronVisualizer") << "Solving for neuron " << neuron << " / "
            << _network->getOutputCount() << "\n";

        size_t xTile = neuron % xTiles;
        size_t yTile = neuron / xTiles;

        size_t xPosition = xTile * (xPixelsPerTile + xPadding);
        size_t yPosition = yTile * (yPixelsPerTile + yPadding);

        image.setTile(xPosition, yPosition, visualizeInputTileForNeuron(neuron));
    }

    return image;
}

static void getTileDimensions(size_t& x, size_t& y, size_t& colors,
    NeuralNetwork& tile)
{
    x = getXPixelsPerTile(tile);
    y = getYPixelsPerTile(tile);

    colors = getColorsPerTile(tile);
}

static size_t sqrtRoundUp(size_t value)
{
    size_t result = std::sqrt((double) value);

    if(result * result < value)
    {
        result += 1;
    }

    return result;
}

static size_t getXPixelsPerTile(NeuralNetwork& network)
{
    return getYPixelsPerTile(network);
}

static size_t getYPixelsPerTile(NeuralNetwork& network)
{
    size_t inputs = network.getInputCount();

    return sqrtRoundUp(inputs / getColorsPerTile(network));
}

static size_t getColorsPerTile(NeuralNetwork& network)
{
    size_t inputs = network.getInputCount();

    return inputs % 3 == 0 ? 3 : 1;
}

static Matrix optimizeWithDerivative(NeuralNetwork*, const Image& , size_t);
static void updateImage(Image& , const Matrix& );

static void visualizeNeuron(NeuralNetwork& network, Image& image, size_t outputNeuron)
{
    auto matrix = optimizeWithDerivative(&network, image, outputNeuron);

    updateImage(image, matrix);
}

void NeuronVisualizer::setNeuralNetwork(NeuralNetwork* network)
{
    _network = network;
}

static Matrix generateRandomImage(NeuralNetwork* network, double range)
{
    return apply(matrix::rand({network->getInputCount()}, network->precision()), matrix::Multiply(range));
}

class CostAndGradientFunction : public optimizer::CostAndGradientFunction
{
public:
    CostAndGradientFunction(NeuralNetwork* n, const Matrix* r)
    : _network(n), _reference(r)
    {

    }


public:
    virtual double computeCostAndGradient(MatrixVector& gradients,
        const MatrixVector& inputs) const
    {
        util::log("NeuronVisualizer::Detail") << " inputs are : " << inputs.front().toString();

        Matrix gradient;
        double newCost = _network->getInputCostAndGradient(gradient, inputs.front(), *_reference);

        gradients.push_back(std::move(gradient));

        util::log("NeuronVisualizer::Detail") << " new gradient is : " << gradients.front().toString();
        util::log("NeuronVisualizer::Detail") << " new cost is : " << newCost << "\n";

        return newCost;
    }

private:
    NeuralNetwork* _network;
    const Matrix*  _reference;
};

static Matrix generateReferenceForNeuron(NeuralNetwork* network,
    size_t neuron)
{
    Matrix reference(1, network->getOutputCount());

    reference(0, 0) = 0.9f;

    return reference;
}

static void addConstraints(optimizer::GeneralDifferentiableSolver& solver)
{
    // constraint values between 0.0f and 255.0f
    solver.addConstraint(ConstantConstraint(1.0f));
    solver.addConstraint(ConstantConstraint(-1.0f, ConstantConstraint::GreaterThanOrEqual));
}

static Matrix optimizeWithDerivative(double& bestCost, NeuralNetwork* network,
    const Matrix& input, size_t neuron)
{
    auto reference = generateReferenceForNeuron(network, neuron);

    auto bestSoFar = matrix::MatrixVector({input});
         bestCost  = network->getCost(input, reference);

    std::string solverType = util::KnobDatabase::getKnobValue(
        "NeuronVisualizer::SolverType", "LBFGSSolver");

    std::unique_ptr<optimizer::GeneralDifferentiableSolver> solver(optimizer::GeneralDifferentiableSolverFactory::create(solverType));

    assert(solver != nullptr);

    addConstraints(*solver);

    util::log("NeuronVisualizer") << " Initial inputs are   : " << input.toString();
    util::log("NeuronVisualizer") << " Initial reference is : " << generateReferenceForNeuron(network, neuron).toString();
    util::log("NeuronVisualizer") << " Initial output is    : " << network->runInputs(input).toString();
    util::log("NeuronVisualizer") << " Initial cost is      : " << bestCost << "\n";

    CostAndGradientFunction costAndGradient(network, &reference);

    bestCost = solver->solve(bestSoFar, costAndGradient);

    util::log("NeuronVisualizer") << "  solver produced new cost: " << bestCost << ".\n";
    util::log("NeuronVisualizer") << "  final input is : " << bestSoFar.toString();
    util::log("NeuronVisualizer") << "  final output is : " << network->runInputs(bestSoFar.front()).toString();

    return bestSoFar.front();
}

static Matrix optimizeWithDerivative(NeuralNetwork* network,
    const Image& image, size_t neuron)
{
    double range = util::KnobDatabase::getKnobValue("NeuronVisualizer::InputRange", 0.01f);

    util::log("NeuronVisualizer") << "Searching for lowest cost inputs...\n";

    auto randomInputs = generateRandomImage(network, range);

    double newCost = std::numeric_limits<double>::max();
    return optimizeWithDerivative(newCost, network, randomInputs, neuron);
}

static void updateImage(Image& image, const Matrix& bestData)
{
    assertM(false, "Not implemented.");
}

}

}


