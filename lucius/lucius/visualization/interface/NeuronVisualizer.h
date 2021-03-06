/*    \file   NeuronVisualizer.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the NeuronVisualizer class.
*/

#pragma once

// Standard Library Includes
#include <cstddef>

// Forward Declaration
namespace lucius { namespace network { class NeuralNetwork; } }
namespace lucius { namespace video   { class Image;         } }

namespace lucius
{

namespace visualization
{

class NeuronVisualizer
{
public:
    typedef network::NeuralNetwork NeuralNetwork;
    typedef video::Image Image;

public:
    NeuronVisualizer(NeuralNetwork* network);

public:
    void visualizeNeuron(Image& image, size_t outputNeuron);
    Image visualizeInputTileForNeuron(size_t outputNeuron);
    Image visualizeInputTilesForAllNeurons();

public:
    void setNeuralNetwork(NeuralNetwork* network);

private:
    NeuralNetwork* _network;
};

}

}



