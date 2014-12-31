/*	\file   Model.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Model class.
*/

// Minerva Includes
#include <minerva/model/interface/Model.h>

#include <minerva/network/interface/NeuralNetwork.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/TarArchive.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/json.h>

// Standard Library Includes
#include <sstream>
#include <map>

namespace minerva
{

namespace model
{

Model::Model(const std::string& path)
: _path(path), _loaded(false), _xPixels(0), _yPixels(0), _colors(3)
{

}

Model::Model()
: _path("unknown-path"), _loaded(true), _xPixels(0), _yPixels(0), _colors(3)
{

}

const Model::NeuralNetwork& Model::getNeuralNetwork(
	const std::string& name) const
{
	auto network = _neuralNetworkMap.find(name);
	
	assertM(network != _neuralNetworkMap.end(), "Invalid neural network name "
		+ name);
	
	return *network->second;
}

Model::NeuralNetwork& Model::getNeuralNetwork(
	const std::string& name)
{
	auto network = _neuralNetworkMap.find(name);
	
	assertM(network != _neuralNetworkMap.end(), "Invalid neural network name "
		+ name);
	
	return *network->second;
}

bool Model::containsNeuralNetwork(const std::string& name) const
{
	return _neuralNetworkMap.count(name) != 0;
}

void Model::setNeuralNetwork(
	const std::string& name, const NeuralNetwork& n)
{
	if(!containsNeuralNetwork(name))
	{
		_neuralNetworkMap[name] = _neuralNetworks.insert(_neuralNetworks.end(), n);
	}
	else
	{
		*_neuralNetworkMap[name] = n;
	}
}
	
void Model::setOutputLabel(size_t output, const std::string& label)
{
	_outputLabels[output] = label;
}

std::string Model::getOutputLabel(size_t output) const
{
	auto label = _outputLabels.find(output);
	
	assert(label != _outputLabels.end());
	
	return label->second;
}

size_t Model::getOutputCount() const
{
	return _outputLabels.size();
}

void Model::setInputImageResolution(size_t x, size_t y, size_t c)
{
	_xPixels = x;
	_yPixels = y;
	_colors  = c;
}

size_t Model::xPixels() const
{
	return _xPixels;
}

size_t Model::yPixels() const
{
	return _yPixels;
}

size_t Model::colors() const
{
	return _colors;
}

void Model::save() const
{
	util::TarArchive tar(_path, "w:gz");
	
	// Save input description
	_saveInputDescription(tar);
	
	// Save output description
	_saveOutputDescription(tar);
	
	// Save networks
	for(auto& network : _neuralNetworkMap)
	{
		network.second->save(tar, network.first);
	}
}

void Model::load()
{
	if(_loaded) return;
	
	_loaded = true;
	
	_neuralNetworks.clear();
	
	util::log("Model") << "Loading classification-model from '"
		<< _path << "'\n";
	
	util::TarArchive tar(_path, "r:gz");
	
	_loadInputDescription(tar);
	
	_loadOutputDescription(tar);
	
	auto networks = _getNetworkList(tar);
	
	for(auto networkName : networks)
	{
		setNeuralNetwork(networkName, network::NeuralNetwork());
		
		getNeuralNetwork(networkName).load(tar, networkName);
	}
}
	
void Model::clear()
{
	_neuralNetworks.clear();
}
	
Model::iterator Model::begin()
{
	return _neuralNetworks.begin();
}

Model::const_iterator Model::begin() const
{
	return _neuralNetworks.begin();
}

Model::iterator Model::end()
{
	return _neuralNetworks.end();
}

Model::const_iterator Model::end() const
{
	return _neuralNetworks.end();
}

Model::reverse_iterator Model::rbegin()
{
	return _neuralNetworks.rbegin();
}

Model::const_reverse_iterator Model::rbegin() const
{
	return _neuralNetworks.rbegin();
}

Model::reverse_iterator Model::rend()
{
	return _neuralNetworks.rend();
}

Model::const_reverse_iterator Model::rend() const
{
	return _neuralNetworks.rend();
}
	
void Model::_saveInputDescription(util::TarArchive& tar) const
{
	assertM(false, "Not implemented.");
}

void Model::_saveOutputDescription(util::TarArchive& tar) const
{
	assertM(false, "Not implemented.");
}

void Model::_loadInputDescription(util::TarArchive& tar)
{
	assertM(false, "Not implemented.");
}

void Model::_loadOutputDescription(util::TarArchive& tar)
{
	assertM(false, "Not implemented.");
}

Model::StringVector Model::_getNetworkList(util::TarArchive& tar)
{
	assertM(false, "Not implemented.");
}

}

}


