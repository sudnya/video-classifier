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

static bool isInMargin(const Matrix& ref, const Matrix& output, float epsilon)
{
    BackPropData::Matrix differenceMat = output.subtract(ref);
    
    for (auto value = differenceMat.begin(); value != differenceMat.end(); ++value)
    {
        if (std::fabs(*value) > epsilon)
            return false;
    }

    return true;
}

static float computeCost(const Layer& layer, const Matrix& layerInput, const Matrix& layerOutput)
{
    //J(theta) = -1/m (sum over i, sum over k yi,k * log (h(xl)k) + (1-yik)*log(1-h(xi)k) + regularization term lambda/2m sum over l,i,j thetai,j,i^2
    
    auto hx = layer.runInputs(layerInput);
    
    auto logHx = hx.log();
    
    auto oneMinusHx = hx.negate().add(1.0f);
    auto logOneMinusHx = oneMinusHx.log();
    
    auto oneMinusY = layerOutput.negate().add(1.0f);

    unsigned m = layerInput.rows();

    auto yTemp = layerOutput.elementMultiply(logHx);

    auto yMinusOneTemp = oneMinusY.elementMultiply(logOneMinusHx);
    
    auto sum = yTemp.add(yMinusOneTemp);
    auto cost = sum.multiply(-1.0f/m);

    //TODO Normalize this with lambda!
    util::log("BackPropData") << "  1 " << cost.toString() << "\n";
    float costSum = cost.reduceSum();
    util::log("BackPropData") << "  2 " << costSum << "\n";

    return costSum;
}

static bool gradientChecking(const Matrix& partialDerivatives, const Layer& layer, const Matrix& layerInput, const Matrix& layerOutput, float epsilon)
{
    Matrix gradientEstimate(partialDerivatives.rows(), partialDerivatives.columns());

    assertM(layer.size() == 1, "No support for sparse layers yet.");
    
    const Matrix& layerWeights = layer.back();
    
    util::log("BackPropData") << "Running gradient checking on " << layerWeights.size() << " weights....\n";

    assert(layerWeights.rows()    == partialDerivatives.rows());
    assert(layerWeights.columns() == partialDerivatives.columns());

    auto weight = layerWeights.begin();
    auto partialDerivative = partialDerivatives.begin();
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
        float derivative = (computeCost(layerPlus, layerInput, layerOutput) - computeCost(layerMinus, layerInput, layerOutput)) / (2.0f * epsilon);
    
        *estimate = derivative;

        util::log("BackPropData") << " gradient of weight " << std::distance(layerWeights.begin(), weight) << " out of " << layerWeights.size() 
            << " weights is " << derivative << ", compared to computed " << *partialDerivative << "\n";
    
         if (std::abs(derivative - *partialDerivative) > epsilon)
         {
            return false;
         }
    }

    return isInMargin(partialDerivatives, gradientEstimate, epsilon);
}

BackPropData::MatrixVector BackPropData::getCostDerivative()
{
    //get activations in a vector
    auto activations = getActivations();
    //get deltas in a vector
    auto deltas = getDeltas(activations);
    
    MatrixVector partialDerivative;
    
    //derivative of layer = activation[i] * delta[i+1] - for same layer
    for (auto i = deltas.begin(), j = activations.begin(); i != deltas.end() && j != activations.end(); ++i, ++j)
    {
        //there will be one less delta than activation
        auto unnormalizedPartialDerivative = (((*i).transpose()).multiply(*j)).transpose();
        
        partialDerivative.push_back(unnormalizedPartialDerivative.multiply(1.0f/(*j).rows()));
    
		util::log("BackPropData") << " computed derivative for layer " << std::distance(deltas.begin(), i) << " (" << partialDerivative.back().rows()
		       << " rows, " << partialDerivative.back().columns() << " columns).\n";
        util::log("BackPropData") << " PD contains " << partialDerivative.back().toString() << "\n";
    
    }//this loop ends after all activations are done. and we don't need the last delta (ref-output anyway)
    
    bool performGradientChecking = util::KnobDatabase::getKnobValue("NeuralNetwork::DoGradientChecking", false);
    
    if (performGradientChecking)
    { 
        float epsilon = util::KnobDatabase::getKnobValue("NeuralNetwork::GradientCheckingEpsilon", 0.05f);
        bool isInRange = gradientChecking(partialDerivative.back(), m_neuralNetworkPtr->back(), *(++activations.rbegin()), m_referenceOutput, epsilon);
        assertM(isInRange, "Gradient checking indicates gradient descent is wrong\n");
    }

    return partialDerivative;
}

BackPropData::NeuralNetwork* BackPropData::getNeuralNetworkPtr()
{
    return m_neuralNetworkPtr;
}

BackPropData::FloatVector BackPropData::getFlattenedWeights()
{
    assertM(false, "Not implemented.");

    return FloatVector();
}

BackPropData::FloatVector BackPropData::getFlattenedCostDerivative()
{
    assertM(false, "Not implemented.");

    return FloatVector();
}

void BackPropData::setFlattenedWeights(const FloatVector& weights)
{
    assertM(false, "Not implemented.");
}

float BackPropData::computeCostForNewFlattenedWeights(const FloatVector& weights) const
{
    assertM(false, "Not implemented.");

    return 0.0f;
}

BackPropData::FloatVector BackPropData::computePartialDerivativesForNewFlattenedWeights(const FloatVector& weights) const
{
    assertM(false, "Not implemented.");

    return FloatVector();
}


BackPropData::MatrixVector BackPropData::getDeltas(const MatrixVector& activations) const
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
        auto& layer = (*m_neuralNetworkPtr)[layerNumber];

        auto activationDerivativeOfCurrentLayer = sigmoidDerivative(*i);
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


BackPropData::MatrixVector BackPropData::getActivations() const
{
	MatrixVector activations;

	auto temp = m_input;
	activations.push_back(temp);
    //util::log("BackPropData") << " added activation of size ( " << activations.back().rows() << " ) rows and ( " << activations.back().columns() << " )\n" ;

	for (auto i = m_neuralNetworkPtr->begin();
		i != m_neuralNetworkPtr->end(); ++i)
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

BackPropData::Matrix BackPropData::sigmoidDerivative(const Matrix& m) const
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

}//end neuralnetwork
}//end minerva

