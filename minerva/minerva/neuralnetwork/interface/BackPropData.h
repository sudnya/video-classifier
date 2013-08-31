/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface of the back propagate data class
 */

#pragma once

#include <vector>

#include <minerva/neuralnetwork/interface/Layer.h>
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

namespace minerva
{
namespace neuralnetwork
{
class BackPropData
{
    public:
        typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;
        typedef minerva::matrix::Matrix Matrix;
        typedef std::vector<minerva::matrix::Matrix> MatrixVector;
    public:
        BackPropData(NeuralNetwork* ann, Matrix input, Matrix ref) : m_neuralNetworkPtr(ann), m_input(input), m_referenceOutput(ref)
        {
        }
    public:
        MatrixVector getCostDerivative();
        bool testDerivative();
        NeuralNetwork* getNeuralNetworkPtr();

    private:
        MatrixVector getDeltas(const MatrixVector& m) const;
        MatrixVector getActivations() const;
        Matrix sigmoidDerivative(const Matrix& m) const;
    private:
        NeuralNetwork* m_neuralNetworkPtr;
        Matrix m_input;
        Matrix m_referenceOutput;

};

}
}
