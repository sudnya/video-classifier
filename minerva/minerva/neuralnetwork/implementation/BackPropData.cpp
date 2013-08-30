/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation of the back propagate data class
 */

#include <minerva/neuralnetwork/interface/BackPropData.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <algorithm>
#include <cmath>

namespace minerva
{
namespace neuralnetwork
{

bool BackPropData::gradientChecking(const BackPropData::MatrixVector cost, float epsilon)
{
    return false;
}

BackPropData::Matrix BackPropData::getCost()
{
    //J(theta) = -1/m (sum over i, sum over k yi,k * log (h(xl)k) + (1-yik)*log(1-h(xi)k) + regularization term lambda/2m sum over l,i,j thetai,j,i^2
    
    Matrix cost;
    Matrix yTemp, yMinusOneTemp, sum;

    auto hx = m_neuralNetworkPtr->runInputs(m_input);
    auto logHx = hx.log();
    
    
    
    auto oneMinusHx = hx.negate().add(1.0f);
    auto logOneMinusHx = oneMinusHx.log();

    auto oneMinusY = m_referenceOutput.negate().add(1.0f);

    unsigned m = m_input.rows();

    yTemp = m_referenceOutput.elementMultiply(logHx);
    yMinusOneTemp = oneMinusY.elementMultiply(logOneMinusHx);
    
    sum = yTemp.add(yMinusOneTemp);
    cost = sum.multiply(-1.0f/m);

    //TODO Normalize this with lambda!
    return cost;
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
        partialDerivative.push_back(((*i).transpose()).multiply(*j));
    
		util::log("BackPropData") << " computed derivative for layer " << std::distance(deltas.begin(), i) << " (" << partialDerivative.back().rows()
		        << " rows, " << partialDerivative.back().columns() << " columns).\n";
        util::log("BackPropData") << " PD contains " << i->toString() << "\n";
    
    }//this loop ends after all activations are done. and we don't need the last delta (ref-output anyway)
    
    util::log("BackPropData") << "Computing gradient checking\n";
    float epsilon = 0.0001f;
    bool isInRange = gradientChecking(deltas, epsilon);
    assertM(isInRange, "Gradient checking indicates gradient descent is wrong\n");
    return partialDerivative;
}

BackPropData::NeuralNetwork* BackPropData::getNeuralNetworkPtr()
{
    return m_neuralNetworkPtr;
}

BackPropData::MatrixVector BackPropData::getDeltas(const MatrixVector& activations) const
{
    MatrixVector deltas;
    
    auto i = activations.rbegin();
    auto delta = m_referenceOutput.subtract(*i);
    ++i;

    while (i != activations.rend())
    {
	    deltas.push_back(delta);
        
        unsigned int layerNumber = std::distance(activations.begin(), --(i.base()));
        util::log ("BackPropData") << " Layer number: " << layerNumber << "\n";
        auto& layer = (*m_neuralNetworkPtr)[layerNumber];

        auto activationDerivativeOfCurrentLayer = sigmoidDerivative(*i);
        auto deltaPropagatedReverse = layer.runReverse(delta);
        
        delta = deltaPropagatedReverse.elementMultiply(activationDerivativeOfCurrentLayer);

        ++i; 
    }

    std::reverse(deltas.begin(), deltas.end());
    for (auto& delta : deltas)
    {
        util::log("BackPropData") << " added delta of size ( " << delta.rows() << " ) rows and ( " << delta.columns() << " )\n" ;
        util::log("BackPropData") << " delta contains " << delta.toString() << "\n";
    }
    return deltas;
}


BackPropData::MatrixVector BackPropData::getActivations() const
{
	MatrixVector activations;

	auto temp = m_input;
	activations.push_back(temp);
    util::log("BackPropData") << " added activation of size ( " << activations.back().rows() << " ) rows and ( " << activations.back().columns() << " )\n" ;

	for (auto i = m_neuralNetworkPtr->begin();
		i != m_neuralNetworkPtr->end(); ++i)
    {

        activations.push_back((*i).runInputs(temp));
        util::log("BackPropData") << " added activation of size ( " << activations.back().rows() << " ) rows and ( " << activations.back().columns() << " )\n" ;
        temp = activations.back();
    }

    return activations;
}

static float sigmoid(float v)
{
	return 1.0f / (1.0f + std::exp(-v));
}

BackPropData::Matrix BackPropData::sigmoidDerivative(const Matrix& m) const
{
    // f(x) = 1/(1+e^-x)
    // dy/dx = f(x)' = f(x) * (1 - f(x))

	Matrix temp = m;

	for(auto& element : temp)
	{
		element = sigmoid(element) * (1 - sigmoid(element));
	}
	
	return temp;
}

}//end neuralnetwork
}//end minerva

