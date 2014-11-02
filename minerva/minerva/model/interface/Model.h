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
	typedef std::list<NeuralNetwork> NeuralNetworkList;
	typedef NeuralNetworkList::iterator iterator;
	typedef NeuralNetworkList::const_iterator const_iterator;

public:
	ClassificationModel(const std::string& path);
	ClassificationModel();

public:
	const NeuralNetwork& getNeuralNetwork(const std::string& name) const;
	NeuralNetwork&  getNeuralNetwork(const std::string& name);

public:
	bool containsNeuralNetwork(const std::string& name) const;

public:
	void setNeuralNetwork(const std::string& name, const NeuralNetwork& n);
	void setInputImageResolution(unsigned int x, unsigned int y, unsigned int colors);

public:
	unsigned int xPixels() const;
	unsigned int yPixels() const;
	unsigned int colors()  const;

public:
	void save() const;
	void load();

public:
	void clear();

public:
	iterator       begin();
	const_iterator begin() const;
	
	iterator       end();
	const_iterator end() const;

private:
	std::string _path;
	bool        _loaded;

private:
	typedef std::map<std::string, iterator> NeuralNetworkMap;

private:
	NeuralNetworkList _neuralNetworks;
	NeuralNetworkMap  _neuralNetworkMap;

	unsigned int _xPixels;
	unsigned int _yPixels;
	unsigned int _colors;

};

}

}


