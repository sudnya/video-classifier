/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The interface of the Neural Network class 
 */

#pragma once

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/neuralnetwork/interface/Layer.h>

#include <string>
#include <vector>
#include <map>

namespace minerva
{
namespace neuralnetwork
{

class NeuralNetwork
{
    public:
        typedef minerva::matrix::Matrix Matrix;
        typedef minerva::neuralnetwork::Layer Layer;

        typedef std::map<unsigned, std::string> NeuronToLabelMap;
        
        typedef std::vector<Layer> LayerVector;
        typedef std::vector<Matrix> MatrixList;

        typedef LayerVector::iterator       iterator;
        typedef LayerVector::const_iterator const_iterator;

    public:
        NeuralNetwork()
        {
        }
    
        void initializeRandomly();
        void backPropagate(const Matrix& input, const Matrix& reference);
        
        Matrix runInputs(const Matrix& m);
        
    public:
        std::string getLabelForOutputNeuron(unsigned int idx) const;
        void setLabelForOutputNeuron(unsigned int idx, const std::string& label);

	public:
		void mirror();
		void cutOffSecondHalf();

    public:
        unsigned getInputCount()  const;
        unsigned getOutputCount() const;

    public:
        void addLayer(const Layer&);
        unsigned getTotalLayerSize() const;
        
              LayerVector* getLayers();
        const LayerVector* getLayers() const;

	public:
		void resize(size_t layers);

    public:
        iterator       begin();
        const_iterator begin() const;

        iterator       end();
        const_iterator end() const;

    public:
              Layer& operator[](size_t index);
        const Layer& operator[](size_t index) const;

    public:
              Layer& back();
        const Layer& back() const;
    
    public:
              Layer& front();
        const Layer& front() const;
    
    public:
        unsigned int size() const;
        bool         empty() const;
    
    private:
        LayerVector m_layers;
        NeuronToLabelMap m_labels;

};
}//end neural network
}//end minerva

