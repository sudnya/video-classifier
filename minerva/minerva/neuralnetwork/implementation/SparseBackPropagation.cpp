/*! \file   SparseBackPropagation.cpp
	\author Gregory Diamos
	\date   Sunday December 22, 2013
	\brief  The source file for the SparseBackPropagation class.
*/

// Minerva Includes
#include <minerva/neuralnetwork/interface/SparseBackPropagation.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <algorithm>

namespace minerva
{

namespace neuralnetwork
{

typedef matrix::Matrix Matrix;
typedef matrix::BlockSparseMatrix BlockSparseMatrix;
typedef Matrix::FloatVector FloatVector;
typedef SparseBackPropagation::MatrixVector MatrixVector;

static MatrixVector computeCostDerivative(const NeuralNetwork& network, const BlockSparseMatrix& input,
	const BlockSparseMatrix& referenceOutput, float lambda, float sparsity, float sparsityWeight);
static BlockSparseMatrix computeInputDerivative(const NeuralNetwork& network, const BlockSparseMatrix& input,
	const BlockSparseMatrix& referenceOutput, float lambda, float sparsity, float sparsityWeight);
static float computeCostForNetwork(const NeuralNetwork& network, const BlockSparseMatrix& input,
	const BlockSparseMatrix& referenceOutput, float lambda, float sparsity, float sparsityWeight);
static Matrix getAverageActivations(const NeuralNetwork& network, const BlockSparseMatrix& input);

SparseBackPropagation::SparseBackPropagation(NeuralNetwork* ann,
	BlockSparseMatrix* input,
	BlockSparseMatrix* ref)
: BackPropagation(ann, input, ref), _lambda(0.0f), _sparsity(0.0f), _sparsityWeight(0.0f)
{
	_lambda         = util::KnobDatabase::getKnobValue("NeuralNetwork::Lambda",         0.05f );
	_sparsity       = util::KnobDatabase::getKnobValue("NeuralNetwork::Sparsity",       0.05f);
	_sparsityWeight = util::KnobDatabase::getKnobValue("NeuralNetwork::SparsityWeight", 0.10f );
}

MatrixVector SparseBackPropagation::getCostDerivative(const NeuralNetwork& neuralNetwork,
	const BlockSparseMatrix& input, const BlockSparseMatrix& reference) const
{
	return neuralnetwork::computeCostDerivative(neuralNetwork, input, reference, _lambda, _sparsity, _sparsityWeight);
}

BlockSparseMatrix SparseBackPropagation::getInputDerivative(const NeuralNetwork& network,
	const BlockSparseMatrix& input, const BlockSparseMatrix& reference) const
{
	return neuralnetwork::computeInputDerivative(network, input, reference, _lambda, _sparsity, _sparsityWeight);
}

float SparseBackPropagation::getCost(const NeuralNetwork& network,
	const BlockSparseMatrix& input, const BlockSparseMatrix& reference) const
{
	return neuralnetwork::computeCostForNetwork(network, input, reference, _lambda, _sparsity, _sparsityWeight);
}

static float computeCostForNetwork(const NeuralNetwork& network, const BlockSparseMatrix& input,
	const BlockSparseMatrix& referenceOutput, float lambda, float sparsity, float sparsityWeight)
{
	//J(theta) = -1/m (sum over i, sum over k y(i,k) * log (h(x)) + (1-y(i,k))*log(1-h(x)) +
	//		   regularization term lambda/2m sum over l,i,j (theta[i,j])^2
	// J = (1/m) .* sum(sum(-yOneHot .* log(hx) - (1 - yOneHot) .* log(1 - hx)));

	const float epsilon = 1.0e-6f;

	unsigned m = input.rows();

	auto hx = network.runInputs(input);
	
	auto logHx = hx.add(epsilon).log();
	auto yTemp = referenceOutput.elementMultiply(logHx);

	auto oneMinusY = referenceOutput.negate().add(1.0f);
	auto oneMinusHx = hx.negate().add(1.0f);
	auto logOneMinusHx = oneMinusHx.add(epsilon).log(); // add an epsilon to avoid log(0)
	auto yMinusOneTemp = oneMinusY.elementMultiply(logOneMinusHx);

	auto sum = yTemp.add(yMinusOneTemp);

	float costSum = sum.reduceSum() * -1.0f / m;

	if(lambda > 0.0f)
	{
		auto weights = network.getFlattenedWeights();

		costSum += (lambda / (2.0f * m)) * ((weights.elementMultiply(weights)).reduceSum());
	}

	if(sparsityWeight > 0.0f)
	{
		// The average activation of each neuron over all samples
		auto averageActivations = getAverageActivations(network, input);

		// Get the KL divergence of each activation
		auto klDivergence = averageActivations.klDivergence(sparsity);
	
		// Add it into the cost
		costSum += sparsityWeight * klDivergence.reduceSum();
	}
	
	return costSum;
}

static MatrixVector getDeltas(const NeuralNetwork& network, const MatrixVector& activations, const BlockSparseMatrix& reference,
	float sparsity, float sparsityWeight)
{
	MatrixVector deltas;
	
	auto i = activations.rbegin();
	auto delta = (*i).subtract(reference);
	++i;

	while (i != activations.rend())
	{
		deltas.push_back(delta);
		
		unsigned int layerNumber = std::distance(activations.begin(), --(i.base()));
		//util::log ("SparseBackPropagation") << " Layer number: " << layerNumber << "\n";
		auto& layer = network[layerNumber];
		auto& activation = *i;		

		size_t samples = i->rows();

		network.formatOutputForLayer(layer, deltas.back());

		auto activationDerivativeOfCurrentLayer = activation.sigmoidDerivative();
		auto deltaPropagatedReverse = layer.runReverse(deltas.back());
		
		// add in the sparsity term
		auto klDivergenceDerivative = activation.reduceSumAlongRows().multiply(1.0f/samples).klDivergenceDerivative(sparsity);

		auto sparsityTerm = klDivergenceDerivative.multiply(sparsityWeight/samples);
	   
		delta = deltaPropagatedReverse.elementMultiply(activationDerivativeOfCurrentLayer).addBroadcastRow(sparsityTerm);

		++i; 
	}

	std::reverse(deltas.begin(), deltas.end());
	for (auto& delta : deltas)
	{
		util::log("SparseBackPropagation") << " added delta of size ( " << delta.rows() << " ) rows and ( " << delta.columns() << " )\n" ;
		util::log("SparseBackPropagation") << " delta contains " << delta.toString() << "\n";
	}
	return deltas;
}

static BlockSparseMatrix getInputDelta(const NeuralNetwork& network, const MatrixVector& activations,
	const BlockSparseMatrix& reference, float sparsity, float sparsityWeight)
{
	auto i = activations.rbegin();
	auto delta = (*i).subtract(reference);
	++i;

	while (i + 1 != activations.rend())
	{
		unsigned int layerNumber = std::distance(activations.begin(), --(i.base()));
		auto& layer = network[layerNumber];

		network.formatOutputForLayer(layer, delta);

		size_t samples = i->rows();

		auto activationDerivativeOfCurrentLayer = i->sigmoidDerivative();
		auto deltaPropagatedReverse = layer.runReverse(delta);

		// add in the sparsity term
		auto klDivergenceDerivative = i->reduceSumAlongRows().multiply(1.0f/samples).klDivergenceDerivative(sparsity);

		auto sparsityTerm = klDivergenceDerivative.multiply(sparsityWeight/samples);

		util::log ("SparseBackPropagation") << " Computing input delta for layer number: " << layerNumber << "\n";
		delta = deltaPropagatedReverse.elementMultiply(activationDerivativeOfCurrentLayer).addBroadcastRow(sparsityTerm);

		++i; 
	}

	// Handle the first layer differently because the input does not have sigmoid applied
	unsigned int layerNumber = 0;
	auto& layer = network[layerNumber];

	network.formatOutputForLayer(layer, delta);

	auto activationDerivativeOfCurrentLayer = activations.front();
	auto deltaPropagatedReverse = layer.runReverse(delta);

	util::log ("SparseBackPropagation") << " Computing input delta for layer number: " << layerNumber << "\n";
	delta = deltaPropagatedReverse;
	
	return delta;	
}

static Matrix getAverageActivations(const NeuralNetwork& network, const BlockSparseMatrix& input)
{
	assert(!network.empty());

	Matrix activations(1, network.totalActivations() - network.getOutputCount());

	auto temp = input;

	size_t position = 0;

	for (auto i = network.begin(); i != --network.end(); ++i)
	{
		network.formatInputForLayer(*i, temp);
		
		temp = (*i).runInputs(temp);
	
		auto reduced = temp.reduceSumAlongRows().multiply(1.0f/temp.rows());
	
		for(auto& matrix : reduced)
		{
			assert(position + matrix.size() <= activations.size());

			std::memcpy(&activations.data()[position],
				&matrix.data()[0],
				matrix.size() * sizeof(float));
			
			position += matrix.size();
		}
	}

	return activations;
}

static MatrixVector getActivations(const NeuralNetwork& network, const BlockSparseMatrix& input) 
{
	MatrixVector activations;

	auto temp = input;

	activations.push_back(temp);
	//util::log("SparseBackPropagation") << " added activation of size ( " << activations.back().rows()
	// << " ) rows and ( " << activations.back().columns() << " )\n" ;

	for (auto i = network.begin(); i != network.end(); ++i)
	{
		network.formatInputForLayer(*i, activations.back());
	
		activations.push_back((*i).runInputs(activations.back()));
		//util::log("SparseBackPropagation") << " added activation of size ( " << activations.back().rows()
		//<< " ) rows and ( " << activations.back().columns() << " )\n" ;
	}

	util::log("SparseBackPropagation") << " intermediate stage ( " << activations[activations.size() / 2].toString() << "\n";
	util::log("SparseBackPropagation") << " final output ( " << activations.back().toString() << "\n";

	return activations;
}

static MatrixVector computeCostDerivative(const NeuralNetwork& network, const BlockSparseMatrix& input,
	const BlockSparseMatrix& referenceOutput, float lambda, float sparsity, float sparsityWeight)
{
	//get activations in a vector
	auto activations = getActivations(network, input);
	//get deltas in a vector
	auto deltas = getDeltas(network, activations, referenceOutput, sparsity, sparsityWeight);
	
	MatrixVector partialDerivative;
	
	unsigned int samples = input.rows();

	//derivative of layer = activation[i] * delta[i+1] - for some layer i
	auto layer = network.begin();
	for (auto i = deltas.begin(), j = activations.begin(); i != deltas.end() && j != activations.end(); ++i, ++j, ++layer)
	{
		auto& activation = *j;
		auto& delta      = *i;

		auto  transposedDelta = delta.transpose();

		transposedDelta.setRowSparse();

		// there will be one less delta than activation
		auto unnormalizedPartialDerivative = (transposedDelta.multiply(activation)).transpose();
		auto normalizedPartialDerivative = unnormalizedPartialDerivative.multiply(1.0f/samples);
		
		// add in the regularization term
		auto weights = layer->getWeightsWithoutBias();

		auto lambdaTerm = weights.multiply(lambda/samples);
		
		auto regularizedPartialDerivative = lambdaTerm.add(normalizedPartialDerivative);
		
		partialDerivative.push_back(regularizedPartialDerivative);
	
		util::log("SparseBackPropagation") << " computed derivative for layer "
			<< std::distance(deltas.begin(), i)
			<< " (" << partialDerivative.back().rows()
			<< " rows, " << partialDerivative.back().columns() << " columns).\n";
		util::log("SparseBackPropagation") << " PD contains " << partialDerivative.back().toString() << "\n";
	
	}//this loop ends after all activations are done. and we don't need the last delta (ref-output anyway)
	
	return partialDerivative;
}

static BlockSparseMatrix computeInputDerivative(const NeuralNetwork& network,
	const BlockSparseMatrix& input,
	const BlockSparseMatrix& referenceOutput,
	float lambda, float sparsity, float sparsityWeight)
{
	//get activations in a vector
	auto activations = getActivations(network, input);
	//get deltas in a vector
	auto delta = getInputDelta(network, activations, referenceOutput, sparsity, sparsityWeight);
	
	util::log("DenseBackPropagation") << "Input delta: " << delta.toString();
	unsigned int samples = input.rows();

	auto partialDerivative = delta;

	auto normalizedPartialDerivative = partialDerivative.multiply(1.0f/samples);

	util::log("DenseBackPropagation") << "Input derivative: " << normalizedPartialDerivative.toString();

	return normalizedPartialDerivative;
}
	
}

}

