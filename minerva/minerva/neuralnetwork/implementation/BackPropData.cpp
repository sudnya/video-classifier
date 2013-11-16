/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation of the back propagate data class
 */

#include <minerva/neuralnetwork/interface/BackPropData.h>
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
typedef BackPropData::MatrixVector MatrixVector;

static bool isInMargin(const Matrix& ref, const Matrix& output, float epsilon)
{
    return output.subtract(ref).abs().reduceSum() < (ref.size() * epsilon);
}

static float computeCost(const Layer& layer, const BlockSparseMatrix& layerInput, const BlockSparseMatrix& layerOutput, float lambda)
{
    //J(theta) = -1/m (sum over i, sum over k yi,k * log (h(xl)k) + (1-yik)*log(1-h(xi)k) + regularization term lambda/2m sum over l,i,j (theta[i,j])^2
    
    auto hx = layer.runInputs(layerInput);
    
    auto logHx = hx.log();
    
    auto oneMinusHx = hx.negate().add(1.0f);
    
    // add an epsilon to avoid log(0)
    auto logOneMinusHx = oneMinusHx.add(0.0001f).log();
    
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

static float computeCost(const NeuralNetwork& network, const BlockSparseMatrix& input,
	const BlockSparseMatrix& referenceOutput, float lambda)
{
	//J(theta) = -1/m (sum over i, sum over k y(i,k) * log (h(x)) + (1-y(i,k))*log(1-h(x)) + regularization term lambda/2m sum over l,i,j (theta[i,j])^2
	// J = (1/m) .* sum(sum(-yOneHot .* log(hx) - (1 - yOneHot) .* log(1 - hx)));

    unsigned m = input.rows();
	
    auto hx = network.runInputs(input);
    
    auto logHx = hx.add(0.000001f).log();
    auto yTemp = referenceOutput.elementMultiply(logHx);

    auto oneMinusY = referenceOutput.negate().add(1.0f);
    auto oneMinusHx = hx.negate().add(1.0f);
    auto logOneMinusHx = oneMinusHx.add(0.000001f).log(); // add an epsilon to avoid log(0)
    auto yMinusOneTemp = oneMinusY.elementMultiply(logOneMinusHx);

    auto sum = yTemp.add(yMinusOneTemp);

    float costSum = sum.reduceSum() * -1.0f / m;

	if(lambda > 0.0f)
	{
		auto weights = network.getFlattenedWeights();

		costSum += (lambda / (2.0f * m)) * ((weights.elementMultiply(weights)).reduceSum());
	}
	
    return costSum;
	
}

static Matrix flatten(const BlockSparseMatrix& blockedMatrix)
{
	Matrix result;

	for(auto& matrix : blockedMatrix)
	{
		result.appendRows(matrix);
	}

	return result;
} 

static bool gradientChecking(const BlockSparseMatrix& partialDerivatives, const Layer& layer, const BlockSparseMatrix& layerInput,
	const BlockSparseMatrix& layerOutput, float epsilon, float lambda)
{
    Matrix gradientEstimate(partialDerivatives.rows(), partialDerivatives.columns());

    auto layerWeights = layer.getFlattenedWeights();
	auto flattenedPartialDerivatives = flatten(partialDerivatives);

    util::log("BackPropData") << "Running gradient checking on " << layerWeights.size() << " weights....\n";

    assert(layerWeights.rows()    == partialDerivatives.rows());
    assert(layerWeights.columns() == partialDerivatives.columns());

    auto weight = layerWeights.begin();
    auto partialDerivative = flattenedPartialDerivatives.begin();
    for (auto estimate = gradientEstimate.begin(); estimate != gradientEstimate.end(); ++estimate, ++weight, ++partialDerivative)
    {
        // +e
        Layer layerPlus = layer;

        layerPlus.back()[std::distance(layerWeights.begin(), weight)] += epsilon;

        util::log("BackPropData") << "  layer plus e " << layerPlus.back().toString() << "\n";
        // -e 
        Layer layerMinus = layer;

        layerMinus.back()[std::distance(layerWeights.begin(), weight)] -= epsilon;
        util::log("BackPropData") << "  layer minus e " << layerMinus.back().toString() << "\n";

        // update derivative value
        float derivative = (computeCost(layerPlus, layerInput, layerOutput, lambda) - computeCost(layerMinus, layerInput, layerOutput, lambda)) / (2.0f * epsilon);
    
        *estimate = derivative;

        util::log("BackPropData") << " gradient of weight " << std::distance(layerWeights.begin(), weight) << " out of " << layerWeights.size() 
            << " weights is " << derivative << ", compared to computed " << *partialDerivative << "\n";
    
         if (std::abs(derivative - *partialDerivative) > epsilon)
         {
            return false;
         }
    }

    return isInMargin(flattenedPartialDerivatives, gradientEstimate, epsilon);
}

BackPropData::BackPropData(NeuralNetwork* ann, const BlockSparseMatrix& input, const BlockSparseMatrix& ref)
 : m_neuralNetworkPtr(ann), m_input(input), m_referenceOutput(ref), m_lambda(0.0f)
{
	m_lambda = util::KnobDatabase::getKnobValue("BackPropData::Lambda", 0.1f);
}

MatrixVector BackPropData::getCostDerivative() const
{
    return getCostDerivative(*m_neuralNetworkPtr);
}

static size_t getElementCount(const MatrixVector& matrices)
{
	size_t size = 0;
	
	for(auto& matrix : matrices)
	{
		size += matrix.size();
	}
	
	return size;
}

static Matrix flatten(const MatrixVector& matrices)
{
	FloatVector flattenedData(getElementCount(matrices));
	
	size_t position = 0;
	
	for(auto& blockedMatrix : matrices)
	{
		for(auto& matrix : blockedMatrix)
		{
			auto data = matrix.data();
			
			std::memcpy(&flattenedData[position], data.data(),
				sizeof(float) * matrix.size());
			
			position += matrix.size();
		}
	}

	return Matrix(1, getElementCount(matrices), flattenedData);
}

NeuralNetwork* BackPropData::getNeuralNetworkPtr()
{
    return m_neuralNetworkPtr;
}

Matrix BackPropData::getFlattenedWeights() const
{
    return m_neuralNetworkPtr->getFlattenedWeights();
}

Matrix BackPropData::getFlattenedCostDerivative() const
{
    return flatten(getCostDerivative());
}

void BackPropData::setFlattenedWeights(const Matrix& weights)
{
    *m_neuralNetworkPtr = createNetworkFromWeights(weights);
}

float BackPropData::computeCostForNewFlattenedWeights(const Matrix& weights) const
{
	auto network = createNetworkFromWeights(weights);

    return computeCost(network, m_input, m_referenceOutput, m_lambda);
}

float BackPropData::computeAccuracyForNewFlattenedWeights(const Matrix& weights) const
{
	auto network = createNetworkFromWeights(weights);

    return network.computeAccuracy(m_input, m_referenceOutput);
}

Matrix BackPropData::computePartialDerivativesForNewFlattenedWeights(const Matrix& weights) const
{
    auto network = createNetworkFromWeights(weights);

	return flatten(getCostDerivative(network));
}

MatrixVector BackPropData::getDeltas(const NeuralNetwork& network, const MatrixVector& activations) const
{
    MatrixVector deltas;
    
    auto i = activations.rbegin();
    auto delta = (*i).subtract(m_referenceOutput);
    ++i;

    while (i != activations.rend())
    {
	    deltas.push_back(delta);
        
        unsigned int layerNumber = std::distance(activations.begin(), --(i.base()));
        //util::log ("BackPropData") << " Layer number: " << layerNumber << "\n";
        auto& layer = network[layerNumber];

        auto activationDerivativeOfCurrentLayer = i->sigmoidDerivative();
        auto deltaPropagatedReverse = layer.runReverse(delta);
        
        delta = deltaPropagatedReverse.elementMultiply(activationDerivativeOfCurrentLayer);

        ++i; 
    }

    std::reverse(deltas.begin(), deltas.end());
    //for (auto& delta : deltas)
    //{
    //    util::log("BackPropData") << " added delta of size ( " << delta.rows() << " ) rows and ( " << delta.columns() << " )\n" ;
    //    util::log("BackPropData") << " delta contains " << delta.toString() << "\n";
    //}
    return deltas;
}

MatrixVector BackPropData::getActivations(const NeuralNetwork& network) const
{
	MatrixVector activations;

	auto temp = m_input;
	activations.push_back(temp);
    //util::log("BackPropData") << " added activation of size ( " << activations.back().rows() << " ) rows and ( " << activations.back().columns() << " )\n" ;

	for (auto i = network.begin(); i != network.end(); ++i)
    {
        activations.push_back((*i).runInputs(temp));
        //util::log("BackPropData") << " added activation of size ( " << activations.back().rows() << " ) rows and ( " << activations.back().columns() << " )\n" ;
        temp = activations.back();
    }

    util::log("BackPropData") << " intermediate stage ( " << activations[activations.size() / 2].toString() << "\n";
    util::log("BackPropData") << " final output ( " << activations.back().toString() << "\n";

    return activations;
}

//static float sigmoid(float v)
//{
//	return 1.0f / (1.0f + std::exp(-v));
//}

Matrix BackPropData::sigmoidDerivative(const Matrix& m) const
{
    // f(x) = 1/(1+e^-x)
    // dy/dx = f(x)' = f(x) * (1 - f(x))

	Matrix temp = m;

	for(auto element : temp)
	{
	//	element = sigmoid(element) * (1.0f - sigmoid(element));
		element = element * (1.0f - element);
	}
	
	return temp;
}


NeuralNetwork BackPropData::createNetworkFromWeights(
	const Matrix& weights) const
{
	NeuralNetwork newNetwork(*m_neuralNetworkPtr);
	
	newNetwork.setFlattenedWeights(weights);
		
	return newNetwork;
}


MatrixVector BackPropData::getCostDerivative(const NeuralNetwork& network) const
{
    //get activations in a vector
    auto activations = getActivations(network);
    //get deltas in a vector
    auto deltas = getDeltas(network, activations);
    
    MatrixVector partialDerivative;
    
    unsigned int samples = m_input.rows();

    //derivative of layer = activation[i] * delta[i+1] - for same layer
    auto layer = network.begin();
    for (auto i = deltas.begin(), j = activations.begin(); i != deltas.end() && j != activations.end(); ++i, ++j, ++layer)
    {
        //there will be one less delta than activation
        auto unnormalizedPartialDerivative = (((*i).transpose()).multiply(*j)).transpose();
        auto normalizedPartialDerivative = unnormalizedPartialDerivative.multiply(1.0f/samples);
        
        auto weights = layer->getWeightsWithoutBias();

        auto lambdaTerm = weights.multiply(m_lambda/samples);
        
        partialDerivative.push_back(lambdaTerm.add(normalizedPartialDerivative));
    
		util::log("BackPropData") << " computed derivative for layer " << std::distance(deltas.begin(), i) << " (" << partialDerivative.back().rows()
		       << " rows, " << partialDerivative.back().columns() << " columns).\n";
        util::log("BackPropData") << " PD contains " << partialDerivative.back().toString() << "\n";
    
    }//this loop ends after all activations are done. and we don't need the last delta (ref-output anyway)
    
    bool performGradientChecking = util::KnobDatabase::getKnobValue("NeuralNetwork::DoGradientChecking", false);
    
    if (performGradientChecking)
    { 
        float epsilon = util::KnobDatabase::getKnobValue("NeuralNetwork::GradientCheckingEpsilon", 0.05f);
        bool isInRange = gradientChecking(partialDerivative.back(), network.back(), *(++activations.rbegin()), m_referenceOutput, epsilon, m_lambda);
        assertM(isInRange, "Gradient checking indicates gradient descent is wrong\n");
    }

    return partialDerivative;	
}

}//end neuralnetwork
}//end minerva

