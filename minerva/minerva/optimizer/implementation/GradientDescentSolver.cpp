/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation for the Gradient Descent Solver class 
 */


#include <minerva/optimizer/interface/Solver.h>
#include <minerva/optimizer/interface/GradientDescentSolver.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

namespace minerva
{
namespace optimizer
{

typedef minerva::matrix::Matrix Matrix;
typedef Matrix::FloatVector FloatVector;
typedef std::vector<Matrix> MatrixVector;



void GradientDescentSolver::solve()
{
    util::log("GradientDescentSolver") << " Solve\n";
	float learningRate = util::KnobDatabase::getKnobValue<float>("GradientDescentSolver::LearningRate", 0.3f);

    // doing batch descent, so dont need cost
    MatrixVector partialDerivatives = m_backPropDataPtr->getCostDerivative();

    //we have partial derivates, now subtract from each layer's activations
    // for each layer we have one pd
    // Li = Li + alpha*pd
    assertM(m_backPropDataPtr->getNeuralNetworkPtr()->getTotalLayerSize() == partialDerivatives.size(), "each layer should be associated with a partial derivative");
    auto deriv = partialDerivatives.begin();
    for (auto layer = m_backPropDataPtr->getNeuralNetworkPtr()->begin(); layer != m_backPropDataPtr->getNeuralNetworkPtr()->end() && deriv != partialDerivatives.end(); ++layer, ++deriv)
    {
    	assertM(layer->size() == 1, "Only dense matrices supported for now.");
    
        // change the neuron value for each matrix in this layer
        for (auto layerWeight = layer->begin(); layerWeight != layer->end(); ++layerWeight)
        {
            auto weightUpdates = deriv->multiply(learningRate);
            
            // Don't change the bias weights
            auto weightAndBiasUpdates = weightUpdates.appendRows(Matrix(1, weightUpdates.columns(), FloatVector(weightUpdates.columns(), 0.0f)));
        
            (*layerWeight) = (*layerWeight).subtract(weightAndBiasUpdates);
        }
    }
}


}
}


