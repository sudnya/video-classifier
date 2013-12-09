/*	\file   NeuronVisualizer.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NeuronVisualizer class.
*/

#pragma once

// Forward Declaration
namespace minerva { namespace neuralnetwork { class NeuralNetwork; } }
namespace minerva { namespace video         { class Image;         } }

namespace minerva
{

namespace visualization
{

class NeuronVisualizer
{
public:
	typedef neuralnetwork::NeuralNetwork NeuralNetwork;
	typedef video::Image Image;

public:
	NeuronVisualizer(const NeuralNetwork* network);

public:
	void visualizeNeuron(Image& image, unsigned int outputNeuron);

public:
	void setNeuralNetwork(const NeuralNetwork* network);
	
private:
	const NeuralNetwork* _network;

};

}

}



