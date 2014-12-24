/*	\file   BuiltInSpecifications.cpp
	\date   Saturday April 26, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the BuiltInSpecifications class.
*/

// Minerva Includes 
#include <minerva/model/interface/BuiltInSpecifications.h>

// Standard Library Includes
#include <sstream>

namespace minerva
{

namespace model
{

std::string BuiltInSpecifications::getConvolutionalFastModelSpecification(size_t outputCount)
{
	std::stringstream stream;
	
	stream << 
		"{\n"
		"\"name\" : \"fast-model\",\n"
		"\n"
		"\"x\" : 32,\n"
		"\"y\" : 32,\n"
		"\"colors\" : 3,\n"
		"\"costFunction\" : \"SquaredError\",\n"
		"\n"
		"\"outputs\" : [ ";
	
	for(size_t output = 0; output < outputCount; ++output)
	{
		if(output != 0) stream << ", ";

		stream << "output" << output;
	}

	stream <<
		" ],\n"
		"\n"
		"\"networks\" : [\n"
		"	{\n"
		"		\"name\" : \"FeatureSelector\",\n"
		"		\n"
		"		\"layers\" : [\n"
		"			{\n"
		"				\"type : \"FeedForwardTiled\",\n"
		"				\"activationFunction : \"RectifiedLinear\",\n"
		"				\"tiles\" : 64,\n"
		"				\"inputsPerTile\" : 48,\n"
		"				\"outputsPerTile\" : 48,\n"
		"				\"tileSpacing\" : 24\n"
		"			},\n"
		"			{\n"
		"				\"type : \"FeedForwardTiled\",\n"
		"				\"activationFunction : \"RectifiedLinear\",\n"
		"				\"tiles\" : 128,\n"
		"				\"inputsPerTile\" : 48,\n"
		"				\"outputsPerTile\" : 12\n"
		"			},\n"
		"			{\n"
		"				\"type : \"FeedForwardTiled\",\n"
		"				\"activationFunction : \"RectifiedLinear\",\n"
		"				\"tiles\" : 32,\n"
		"				\"inputsPerTile\" : 48,\n"
		"				\"outputsPerTile\" : 48\n"
		"			}\n"
		"		]\n"
		"	},\n"
		"	{\n"
		"		\"name\" : \"Classifier\",\n"
		"		\"layers\" : [\n"
		"			{\n"
		"				\"type : \"FeedForwardTiled\",\n"
		"				\"activationFunction : \"RectifiedLinear\",\n"
		"				\"tiles\" : 32,\n"
		"				\"inputsPerTile\" : 48,\n"
		"				\"outputsPerTile\" : 8\n"
		"			},\n"
		"			{\n"
		"				\"type : \"FeedForwardTiled\",\n"
		"				\"activationFunction : \"RectifiedLinear\",\n"
		"				\"tiles\" : 1,\n"
		"				\"inputsPerTile\" : 256,\n"
		"				\"outputsPerTile\" : 256\n"
		"			},\n"
		"			{\n"
		"				\"type : \"FeedForwardTiled\",\n"
		"				\"activationFunction : \"RectifiedLinear\",\n"
		"				\"tiles\" : 1,\n"
		"				\"inputsPerTile\" : 256,\n"
		"				\"outputsPerTile\" : 1\n"
		"			}\n"
		"		]\n"
		"	\n}"
		"	]\n"
		" }\n\n";
	
	return stream.str();
}

}

}


