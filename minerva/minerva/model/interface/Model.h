/*	\file   Model.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Model class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <map>
#include <list>
#include <vector>

// Forward Declarations
namespace minerva { namespace network { class NeuralNetwork; } }
namespace minerva { namespace util    { class TarArchive;    } }

namespace minerva
{

namespace model
{

class Model
{
public:
	typedef network::NeuralNetwork NeuralNetwork;
	typedef std::list<NeuralNetwork> NeuralNetworkList;
	typedef NeuralNetworkList::iterator iterator;
	typedef NeuralNetworkList::const_iterator const_iterator;
	typedef NeuralNetworkList::reverse_iterator reverse_iterator;
	typedef NeuralNetworkList::const_reverse_iterator const_reverse_iterator;

public:
	Model(const std::string& path);
	Model();

public:
	const NeuralNetwork& getNeuralNetwork(const std::string& name) const;
	NeuralNetwork& getNeuralNetwork(const std::string& name);

public:
	bool containsNeuralNetwork(const std::string& name) const;

public:
	void setNeuralNetwork(const std::string& name, const NeuralNetwork& n);
	void setInputImageResolution(unsigned int x, unsigned int y, unsigned int colors);

public:
	void setOutputLabel(size_t output, const std::string& label);
	std::string getOutputLabel(size_t output) const;

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

public:
	reverse_iterator       rbegin();
	const_reverse_iterator rbegin() const;
	
	reverse_iterator       rend();
	const_reverse_iterator rend() const;

private:
	std::string _path;
	bool        _loaded;

private:
	typedef std::map<std::string, iterator> NeuralNetworkMap;
	typedef std::map<size_t, std::string> LabelMap;
	typedef std::vector<std::string> StringVector;

private:
	NeuralNetworkList _neuralNetworks;
	NeuralNetworkMap  _neuralNetworkMap;
	LabelMap          _outputLabels;

	unsigned int _xPixels;
	unsigned int _yPixels;
	unsigned int _colors;
	
private:
	void _saveInputDescription(util::TarArchive& tar) const;
	void _saveOutputDescription(util::TarArchive& tar) const;

private:
	void _loadInputDescription(util::TarArchive& tar);
	void _loadOutputDescription(util::TarArchive& tar);

private:
	StringVector _getNetworkList(util::TarArchive& tar);
	
};

}

}


