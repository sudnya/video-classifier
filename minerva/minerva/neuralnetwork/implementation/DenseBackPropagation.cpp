/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation of the dense back propagate data class
 */

#include <minerva/neuralnetwork/interface/DenseBackPropagation.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>

// Minerva Includes
#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <algorithm>
#include <cmath>

namespace minerva
{

namespace neuralnetwork
{

typedef matrix::Matrix Matrix;
typedef matrix::BlockSparseMatrix BlockSparseMatrix;
typedef Matrix::FloatVector FloatVector;
typedef DenseBackPropagation::MatrixVector MatrixVector;

static bool isInMargin(const Matrix& ref, const Matrix& output, float epsilon)
{
	return output.subtract(ref).abs().reduceSum() < (ref.size() * epsilon);
}

static float computeCostForLayer(const Layer& layer, const BlockSparseMatrix& layerInput,
	const BlockSparseMatrix& layerOutput, float lambda)
{
	//J(theta) = -1/m (sum over i, sum over k yi,k * log (h(xl)k) + (1-yik)*log(1-h(xi)k) +
	//		   regularization term lambda/2m sum over l,i,j (theta[i,j])^2
	
	auto hx = layer.runInputs(layerInput);
	
	auto logHx = hx.log();
	
	auto oneMinusHx = hx.negate().add(1.0f);
	
	// add an epsilon to avoid log(0)
	auto logOneMinusHx = oneMinusHx.add(1e-15f).log();
	
	auto oneMinusY = layerOutput.negate().add(1.0f);

	unsigned m = layerInput.rows();

	auto yTemp = layerOutput.elementMultiply(logHx);

	auto yMinusOneTemp = oneMinusY.elementMultiply(logOneMinusHx);
	
	auto sum = yTemp.add(yMinusOneTemp);
	auto cost = sum.multiply(-1.0f/m);

	//TODO Normalize this with lambda!
	float costSum = cost.reduceSum();

	auto weights = layer.getFlattenedWeights();

	costSum += (lambda / (2.0f * m)) * (weights.elementMultiply(weights).reduceSum());

	return costSum;
}

static float computeCostForNetwork(const NeuralNetwork& network, const BlockSparseMatrix& input,
	const BlockSparseMatrix& referenceOutput, float lambda)
{
	//J(theta) = -1/m (sum over i, sum over k y(i,k) * log (h(x)) + (1-y(i,k))*log(1-h(x)) +
	//		   regularization term lambda/2m sum over l,i,j (theta[i,j])^2
	// J = (1/m) .* sum(sum(-yOneHot .* log(hx) - (1 - yOneHot) .* log(1 - hx)));

	unsigned m = input.rows();

	
	auto hx = network.runInputs(input);
	
	auto logHx = hx.add(1e-15f).log();
	auto yTemp = referenceOutput.elementMultiply(logHx);

	auto oneMinusY = referenceOutput.negate().add(1.0f);
	auto oneMinusHx = hx.negate().add(1.0f);
	auto logOneMinusHx = oneMinusHx.add(1e-15f).log(); // add an epsilon to avoid log(0)
	auto yMinusOneTemp = oneMinusY.elementMultiply(logOneMinusHx);

	auto sum = yTemp.add(yMinusOneTemp);

	float costSum = sum.reduceSum() * -1.0f / m;

	if(lambda > 0.0f)
	{
		for(auto& layer : network)
		{
			costSum += (lambda / (2.0f * m)) * ((layer.getWeightsWithoutBias().elementMultiply(
				layer.getWeightsWithoutBias())).reduceSum());
		}
	}
	
	return costSum;
}

static bool gradientChecking(const BlockSparseMatrix& partialDerivatives,
	const Layer& layer, const BlockSparseMatrix& layerInput,
	const BlockSparseMatrix& layerOutput, float epsilon, float lambda)
{
	auto layerWeights = layer.getFlattenedWeights();
	auto flattenedPartialDerivatives = BackPropagation::flatten(partialDerivatives);

	Matrix gradientEstimate(flattenedPartialDerivatives.rows(), flattenedPartialDerivatives.columns());

	util::log("DenseBackPropagation") << "Running gradient checking on " << layerWeights.size() << " weights....\n";

	assertM(layerWeights.rows() == flattenedPartialDerivatives.rows(), "Layer weights has " << layerWeights.rows()
		<< " rows, but the flattened partial derivatives has " << flattenedPartialDerivatives.rows());
	assert(layerWeights.columns() == flattenedPartialDerivatives.columns());

	auto weight = layerWeights.begin();
	auto partialDerivative = flattenedPartialDerivatives.begin();
	for (auto estimate = gradientEstimate.begin(); estimate != gradientEstimate.end(); ++estimate, ++weight, ++partialDerivative)
	{
		// +e
		Layer layerPlus = layer;

		layerPlus.back()[std::distance(layerWeights.begin(), weight)] += epsilon;

		util::log("DenseBackPropagation") << "  layer plus e " << layerPlus.back().toString() << "\n";
		// -e 
		Layer layerMinus = layer;

		layerMinus.back()[std::distance(layerWeights.begin(), weight)] -= epsilon;
		util::log("DenseBackPropagation") << "  layer minus e " << layerMinus.back().toString() << "\n";

		// update derivative value
		float derivative = (computeCostForLayer(layerPlus, layerInput, layerOutput, lambda) -
			computeCostForLayer(layerMinus, layerInput, layerOutput, lambda)) / (2.0f * epsilon);
	
		*estimate = derivative;

		util::log("DenseBackPropagation") << " gradient of weight "
			<< std::distance(layerWeights.begin(), weight) << " out of " << layerWeights.size() 
			<< " weights is " << derivative << ", compared to computed " << *partialDerivative << "\n";
	
		 if (std::abs(derivative - *partialDerivative) > epsilon)
		 {
			return false;
		 }
	}

	return isInMargin(flattenedPartialDerivatives, gradientEstimate, epsilon);
}

DenseBackPropagation::DenseBackPropagation(NeuralNetwork* ann, BlockSparseMatrix* input, BlockSparseMatrix* ref)
 : BackPropagation(ann, input, ref), _lambda(0.0f)
{
	_lambda = util::KnobDatabase::getKnobValue("NeuralNetwork::Lambda", 0.01f);
}

MatrixVector DenseBackPropagation::getCostDerivative(const NeuralNetwork& network,
	const BlockSparseMatrix& input,
	const BlockSparseMatrix& reference) const
{
	return getCostDerivative(network);
}

BlockSparseMatrix DenseBackPropagation::getInputDerivative(const NeuralNetwork& network,
	const BlockSparseMatrix& input, const BlockSparseMatrix&) const
{
	return getInputDerivative(network, input);
}

float DenseBackPropagation::getCost(const NeuralNetwork& network, const BlockSparseMatrix& input,
	const BlockSparseMatrix& reference) const
{
	return computeCostForNetwork(network, input, reference, _lambda);
}

MatrixVector DenseBackPropagation::getDeltas(const NeuralNetwork& network, const MatrixVector& activations) const
{
	MatrixVector deltas;

	deltas.reserve(activations.size() - 1);
	
	auto i = activations.rbegin();
	auto delta = (*i).subtract(*_referenceOutput);
	++i;

	while (i != activations.rend())
	{
		deltas.push_back(std::move(delta));
		
		unsigned int layerNumber = std::distance(activations.begin(), --(i.base()));
		//util::log ("DenseBackPropagation") << " Layer number: " << layerNumber << "\n";
		auto& layer = network[layerNumber];

		network.formatOutputForLayer(layer, deltas.back());

		auto activationDerivativeOfCurrentLayer = i->sigmoidDerivative();
		auto deltaPropagatedReverse = layer.runReverse(deltas.back());
	   
		delta = deltaPropagatedReverse.elementMultiply(activationDerivativeOfCurrentLayer);

		++i; 
	}

	std::reverse(deltas.begin(), deltas.end());
	
	if (util::isLogEnabled("DenseBackPropagation::Detail"))
	{
		for (auto& delta : deltas)
		{
			util::log("DenseBackPropagation::Detail") << " added delta of size ( " << delta.rows() << " ) rows and ( " << delta.columns() << " )\n" ;
			util::log("DenseBackPropagation::Detail") << " delta contains " << delta.toString() << "\n";
		}
	}
	
	return deltas;
}

BlockSparseMatrix DenseBackPropagation::getInputDelta(const NeuralNetwork& network, const MatrixVector& activations) const
{
	auto i = activations.rbegin();
	auto delta = (*i).subtract(*_referenceOutput);
	++i;

	while (i + 1 != activations.rend())
	{
		unsigned int layerNumber = std::distance(activations.begin(), --(i.base()));
		auto& layer = network[layerNumber];

		network.formatOutputForLayer(layer, delta);

		auto activationDerivativeOfCurrentLayer = i->sigmoidDerivative();
		auto deltaPropagatedReverse = layer.runReverse(delta);

		util::log ("DenseBackPropagation") << " Computing input delta for layer number: " << layerNumber << "\n";
		delta = deltaPropagatedReverse.elementMultiply(activationDerivativeOfCurrentLayer);

		++i; 
	}

	// Handle the first layer differently because the input does not have sigmoid applied
	unsigned int layerNumber = 0;
	auto& layer = network[layerNumber];

	network.formatOutputForLayer(layer, delta);

	auto activationDerivativeOfCurrentLayer = activations.front();
	auto deltaPropagatedReverse = layer.runReverse(delta);

	util::log ("DenseBackPropagation") << " Computing input delta for layer number: " << layerNumber << "\n";
	delta = deltaPropagatedReverse;//deltaPropagatedReverse.elementMultiply(activationDerivativeOfCurrentLayer);
	
	return delta;	
}

MatrixVector DenseBackPropagation::getActivations(const NeuralNetwork& network, const BlockSparseMatrix& input) const
{
	MatrixVector activations;

	activations.reserve(network.size() + 1);

	auto temp = input;

	activations.push_back(temp);
	//util::log("DenseBackPropagation") << " added activation of size ( " << activations.back().rows()
	// << " ) rows and ( " << activations.back().columns() << " )\n" ;

	for (auto i = network.begin(); i != network.end(); ++i)
	{
		network.formatInputForLayer(*i, activations.back());
	
		activations.push_back((*i).runInputs(activations.back()));
		//util::log("DenseBackPropagation") << " added activation of size ( " << activations.back().rows()
		//<< " ) rows and ( " << activations.back().columns() << " )\n" ;
	}

	//util::log("DenseBackPropagation") << " intermediate stage ( " << activations[activations.size() / 2].toString() << "\n";
	//util::log("DenseBackPropagation") << " final output ( " << activations.back().toString() << "\n";

	return activations;
}

MatrixVector DenseBackPropagation::getCostDerivative(
	const NeuralNetwork& network) const
{
	//get activations in a vector
	auto activations = getActivations(network, *_input);
	//get deltas in a vector
	auto deltas = getDeltas(network, activations);
	
	MatrixVector partialDerivative;

	partialDerivative.reserve(deltas.size());
	
	unsigned int samples = _input->rows();

	//derivative of layer = activation[i] * delta[i+1] - for some layer i
	auto layer = network.begin();
	for (auto i = deltas.begin(), j = activations.begin(); i != deltas.end() && j != activations.end(); ++i, ++j, ++layer)
	{
		auto transposedDelta = (*i).transpose();

		transposedDelta.setRowSparse();

		//there will be one less delta than activation
		auto unnormalizedPartialDerivative = (transposedDelta.multiply(*j));
		auto normalizedPartialDerivative = unnormalizedPartialDerivative.multiply(1.0f/samples);
		
		// add in the regularization term
		auto weights = layer->getWeightsWithoutBias();

		auto lambdaTerm = weights.multiply(_lambda/samples);
		
		partialDerivative.push_back(lambdaTerm.add(normalizedPartialDerivative.transpose()));

		//util::log("DenseBackPropagation") << " computed derivative for layer " << std::distance(deltas.begin(), i)
		//	<< " (" << partialDerivative.back().rows()
		//	<< " rows, " << partialDerivative.back().columns() << " columns).\n";
		//util::log("DenseBackPropagation") << " PD contains " << partialDerivative.back().toString() << "\n";
	
	}//this loop ends after all activations are done. and we don't need the last delta (ref-output anyway)

	bool performGradientChecking = util::KnobDatabase::getKnobValue("NeuralNetwork::DoGradientChecking", false);

	if (performGradientChecking)
	{ 
		float epsilon = util::KnobDatabase::getKnobValue("NeuralNetwork::GradientCheckingEpsilon", 0.05f);
		bool isInRange = gradientChecking(partialDerivative.back(), network.back(), *(++activations.rbegin()),
			*_referenceOutput, epsilon, _lambda);
		assertM(isInRange, "Gradient checking indicates gradient descent is wrong\n");
	}

	return partialDerivative;	
}

BlockSparseMatrix DenseBackPropagation::getInputDerivative(
	const NeuralNetwork& network,
	const BlockSparseMatrix& input) const
{
	//get activations in a vector
	auto activations = getActivations(network, input);
	//get deltas in a vector
	auto delta = getInputDelta(network, activations);
	
	util::log("DenseBackPropagation") << "Input delta: " << delta.toString();
	unsigned int samples = input.rows();

	auto partialDerivative = delta;

	auto normalizedPartialDerivative = partialDerivative.multiply(1.0f/samples);

	util::log("DenseBackPropagation") << "Input derivative: " << normalizedPartialDerivative.toString();

	return normalizedPartialDerivative;
}

}//end neuralnetwork

}//end minerva


