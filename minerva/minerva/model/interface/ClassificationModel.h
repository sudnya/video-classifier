/*	\file   ClassificationModel.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ClassificationModel class.
*/

#pragma once

// Minerva Includes
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

// Standard Library Includes
#include <string>
#include <map>

namespace minerva
{

namespace model
{

class ClassificationModel
{
public:
	typedef neuralnetwork::NeuralNetwork NeuralNetwork;

public:
	ClassificationModel(const std::string& path = "unknown-path");

public:
	const NeuralNetwork& getNeuralNetwork(const std::string& name) const;
	NeuralNetwork& getNeuralNetwork(const std::string& name);
	void setNeuralNetwork(const std::string& name, const NeuralNetwork& n);
	void setInputImageResolution(unsigned int x, unsigned int y, unsigned int colors);

public:
	unsigned int xPixels() const;
	unsigned int yPixels() const;
	unsigned int colors()  const;

public:
	void save() const;
	void load();

private:
	std::string _path;
	bool        _loaded;

private:
	typedef std::map<std::string, NeuralNetwork> NeuralNetworkMap;

private:
	NeuralNetworkMap _neuralNetworks;

	unsigned int _xPixels;
	unsigned int _yPixels;
	unsigned int _colors;

};

}

}


