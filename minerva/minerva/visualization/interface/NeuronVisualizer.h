/*	\file   NeuronVisualizer.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NeuronVisualizer class.
*/

#pragma once

// Standard Library Includes
#include <cstddef>

// Forward Declaration
namespace minerva { namespace network { class NeuralNetwork; } }
namespace minerva { namespace video   { class Image;         } }

namespace minerva
{

namespace visualization
{

class NeuronVisualizer
{
public:
	typedef network::NeuralNetwork NeuralNetwork;
	typedef video::Image Image;

public:
	NeuronVisualizer(const NeuralNetwork* network);

public:
	void visualizeNeuron(Image& image, size_t outputNeuron);
	Image visualizeInputTileForNeuron(size_t outputNeuron);
	Image visualizeInputTilesForAllNeurons();

public:
	void setNeuralNetwork(const NeuralNetwork* network);
	
private:
	const NeuralNetwork* _network;
};

}

}



