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
		"\"name\" : \"FastModel\",\n"
		"\n"
		"input : {\n"
		"   \"type\"   : \"Visual\","
		"	\"x\"      : 32,\n"
		"	\"y\"      : 32,\n"
		"	\"colors\" : 3\n"
		"},\n"
		"\n"
		"\"costFunction\" : \"SquaredError\",\n"
		"\n"
		"\"outputs\" : [ \"";
	
	for(size_t output = 0; output < outputCount; ++output)
	{
		if(output != 0) stream << ", ";

		stream << "output" << output << "\"";
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
		"				\"name\" : \"FeatureSelector0\",\n"
		"				\"type\" : \"FeedForwardTiled\",\n"
		"				\"activationFunction\" : \"RectifiedLinear\",\n"
		"				\"tiles\" : 1,\n"
		"				\"inputsPerTile\" : 48,\n"
		"				\"outputsPerTile\" : 48,\n"
		"				\"tileSpacing\" : 24\n"
		"			},\n"
		"			{\n"
		"				\"name\" : \"FeatureSelector1\",\n"
		"				\"type\" : \"FeedForwardTiled\",\n"
		"				\"activationFunction\" : \"RectifiedLinear\",\n"
		"				\"tiles\" : 1,\n"
		"				\"inputsPerTile\" : 48,\n"
		"				\"outputsPerTile\" : 12\n"
		"			},\n"
		"			{\n"
		"				\"name\" : \"FeatureSelector2\",\n"
		"				\"type\" : \"FeedForwardTiled\",\n"
		"				\"activationFunction\" : \"RectifiedLinear\",\n"
		"				\"tiles\" : 1,\n"
		"				\"inputsPerTile\" : 48,\n"
		"				\"outputsPerTile\" : 48\n"
		"			}\n"
		"		]\n"
		"	},\n"
		"	{\n"
		"		\"name\" : \"Classifier\",\n"
		"		\"layers\" : [\n"
		"			{\n"
		"				\"name\" : \"Classifier0\",\n"
		"				\"type\" : \"FeedForwardTiled\",\n"
		"				\"activationFunction\" : \"RectifiedLinear\",\n"
		"				\"tiles\" : 1,\n"
		"				\"inputsPerTile\" : 256,\n"
		"				\"outputsPerTile\" : 64\n"
		"			},\n"
		"			{\n"
		"				\"name\" : \"Classifier1\",\n"
		"				\"type\" : \"FeedForwardTiled\",\n"
		"				\"activationFunction\" : \"RectifiedLinear\",\n"
		"				\"tiles\" : 1,\n"
		"				\"inputsPerTile\" : 192,\n"
		"				\"outputsPerTile\" : 192\n"
		"			},\n"
		"			{\n"
		"				\"name\" : \"Classifier2\",\n"
		"				\"type\" : \"FeedForwardTiled\",\n"
		"				\"activationFunction\" : \"RectifiedLinear\",\n"
		"				\"tiles\" : 1,\n"
		"				\"inputsPerTile\" : 192,\n"
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


