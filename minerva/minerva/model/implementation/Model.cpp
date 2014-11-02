/*	\file   ClassificationModel.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ClassificationModel class.
*/

// Minerva Includes
#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/neuralnetwork/interface/Layer.h>

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

ClassificationModel::ClassificationModel(const std::string& path)
: _path(path), _loaded(false), _xPixels(0), _yPixels(0), _colors(3)
{

}

ClassificationModel::ClassificationModel()
: _path("unknown-path"), _loaded(true), _xPixels(0), _yPixels(0), _colors(3)
{

}

const ClassificationModel::NeuralNetwork& ClassificationModel::getNeuralNetwork(
	const std::string& name) const
{
	auto network = _neuralNetworks.find(name);
	
	assertM(network != _neuralNetworks.end(), "Invalid neural network name "
		+ name);
	
	return network->second;
}

ClassificationModel::NeuralNetwork& ClassificationModel::getNeuralNetwork(
	const std::string& name)
{
	auto network = _neuralNetworks.find(name);
	
	assertM(network != _neuralNetworks.end(), "Invalid neural network name "
		+ name);
	
	return network->second;
}

bool ClassificationModel::containsNeuralNetwork(const std::string& name) const
{
	return _neuralNetworks.count(name) != 0;
}

void ClassificationModel::setNeuralNetwork(
	const std::string& name, const NeuralNetwork& n)
{
	assert(n.areConnectionsValid());
	
	_neuralNetworks[name] = n;
}

void ClassificationModel::setInputImageResolution(unsigned int x,
	unsigned int y, unsigned int c)
{
	_xPixels = x;
	_yPixels = y;
	_colors  = c;
}

unsigned int ClassificationModel::xPixels() const
{
	return _xPixels;
}

unsigned int ClassificationModel::yPixels() const
{
	return _yPixels;
}

unsigned int ClassificationModel::colors() const
{
	return _colors;
}

void ClassificationModel::save() const
{
	typedef matrix::Matrix Matrix;
	typedef std::map<std::string, const Matrix*> MatrixMap;
	
	std::stringstream stream;
	
	stream << "{\n";
	
	stream << "\t\"path\": \"" << _path << "\",\n";
	
	stream << "\n";
	stream << "\t\"xPixels\": " << xPixels() << ",\n";
	stream << "\t\"yPixels\": " << yPixels() << ",\n";
	stream << "\t\"colors\": "  << colors()  << ",\n";

	stream << "\n";

	stream << "\t\"neuralnetworks\" : [\n";

	MatrixMap filenameToMatrices;

	for(auto network = _neuralNetworks.begin();
		network != _neuralNetworks.end(); ++network)
	{	
		if(network != _neuralNetworks.begin())
		{
			stream << ",\n";
		}
		
		stream << "\t\t{\n";
		
		stream << "\t\t\t\"name\": \"" << network->first << "\",\n";

        std::string costFunctionType = "dense";

        if(network->second.isUsingSparseCostFunction())
        {
            costFunctionType = "sparse";
        }

		stream << "\t\t\t\"costFunction\": \"" << costFunctionType << "\",\n";

		stream << "\t\t\t\"layers\": [\n";
		
		for(auto layer = network->second.begin();
			layer != network->second.end(); ++layer)
		{
			if(layer != network->second.begin())
			{
				stream << ",\n";
			}
			
			unsigned int index = std::distance(network->second.begin(), layer);
			
			stream << "\t\t\t\t{\n";
			
			stream << "\t\t\t\t\t\"name\": \"layer" << index << "\",\n";
			stream << "\t\t\t\t\t\"step\": " << layer->blockStep() << ",\n";
			stream << "\t\t\t\t\t\"weights\" : [\n";
			
			for(auto matrix = layer->begin(); matrix != layer->end(); ++matrix)
			{
				if(matrix != layer->begin())
				{
					stream << ",\n";
				}
			
				unsigned int matrixIndex =
					std::distance(layer->begin(), matrix);
			
				std::stringstream filename;
				
				filename << network->first << "-layer" << index
					<< "-matrix" << matrixIndex << ".bin";
			
				stream << "\t\t\t\t\t\t\"" << filename.str() << "\"";
				
				filenameToMatrices.insert(
					std::make_pair(filename.str(), &*matrix));
			}

			stream << "],\n";
			stream << "\t\t\t\t\t\"biases\" : [\n";

			for(auto matrix = layer->begin_bias(); matrix != layer->end_bias(); ++matrix)
			{
				if(matrix != layer->begin_bias())
				{
					stream << ",\n";
				}
			
				unsigned int matrixIndex =
					std::distance(layer->begin_bias(), matrix);
			
				std::stringstream filename;
				
				filename << network->first << "-layer" << index
					<< "-bias-matrix" << matrixIndex << ".bin";
			
				stream << "\t\t\t\t\t\t\"" << filename.str() << "\"";
				
				filenameToMatrices.insert(
					std::make_pair(filename.str(), &*matrix));

			}
			
			stream << "]\n";

			
			stream << "\t\t\t\t}";
		}	
		
		stream << "],\n";
		
		stream << "\t\t\t\"output-names\": [\n";
		
		for(unsigned int output = 0;
			output != network->second.getOutputCount(); ++output)
		{
			if(output != 0)
			{
				stream << ",\n";
			}

			stream << "\t\t\t\t\"" << 
				network->second.getLabelForOutputNeuron(output) << "\"";
		}	
		
		stream << "]\n";
		
		stream << "\t\t}";
	}
	
	stream << "]\n";

	stream << "}\n";
	
	util::TarArchive tar(_path, "w:gz");
	
	tar.addFile("model.json", stream);
	
	for(auto matrix : filenameToMatrices)
	{
		std::stringstream stream;
		
		uint64_t rows    = matrix.second->rows();
		uint64_t columns = matrix.second->columns();
		
		stream.write((const char*)&rows,    sizeof(uint64_t));
		stream.write((const char*)&columns, sizeof(uint64_t));

		auto data = matrix.second->data();

		stream.write((const char*)data.data(), matrix.second->size() * sizeof(float));
		
		tar.addFile(matrix.first, stream);
	}
}

void ClassificationModel::load()
{
	if(_loaded) return;
	
	_loaded = true;
	
	_neuralNetworks.clear();
	
	util::log("ClassificationModel") << "Loading classification-model from '"
		<< _path << "'\n";
	
	util::TarArchive tar(_path, "r:gz");
	
	std::stringstream header;
	
	tar.extractFile("model.json", header);
	
	util::json::Parser parser;
	auto headerObject = parser.parse_object(header);

	util::log("ClassificationModel") << " loading header\n";
	
	try
	{
		util::json::Visitor headerVisitor(headerObject);
		
		_xPixels = (int) headerVisitor["xPixels"];		
		_yPixels = (int) headerVisitor["yPixels"];	
		_colors  = (int) headerVisitor["colors" ];

		util::log("ClassificationModel") << "  (" << _xPixels << " xPixels, "
			<< _yPixels << " yPixels, " << _colors << " colors)\n";
		
		util::json::Visitor networksVisitor(
			headerVisitor["neuralnetworks"]);		
	
		for(auto networkObject = networksVisitor.begin_array();
			networkObject != networksVisitor.end_array(); ++networkObject)
		{
			util::json::Visitor networkVisitor(*networkObject);
			
			std::string name = networkVisitor["name"];
			std::string costFunctionType = networkVisitor["costFunction"];

			util::log("ClassificationModel") << "  neural network '"
				<< name << "'\n";
			
			util::json::Visitor layersVisitor(networkVisitor["layers"]);	
			
			auto network = _neuralNetworks.insert(std::make_pair(name,
				neuralnetwork::NeuralNetwork())).first;
		
            if(costFunctionType == "sparse")
            {
                network->second.setUseSparseCostFunction(true);
            }
            else
            {
                network->second.setUseSparseCostFunction(false);
            }

			for(auto layerObject = layersVisitor.begin_array();
				layerObject != layersVisitor.end_array(); ++layerObject)
			{
				network->second.addLayer(neuralnetwork::Layer());
				
				auto& layer = network->second.back();
				
				util::json::Visitor layerVisitor(*layerObject);

				util::log("ClassificationModel") << "   layer "
					<< (std::string)layerVisitor["name"] << "\n";

				size_t step = (int)layerVisitor["step"];
		
				layer.setBlockStep(step);
				
				util::json::Visitor matrixArrayVisitor(layerVisitor["weights"]);	
					
				for(auto weightMatrixObject = matrixArrayVisitor.begin_array();
					weightMatrixObject != matrixArrayVisitor.end_array();
					++weightMatrixObject)
				{
					util::json::Visitor weightMatrixVisitor(
						*weightMatrixObject);
					
					std::string filename = weightMatrixVisitor;
					
					std::stringstream stream;
					
					tar.extractFile(filename, stream);
					
					uint64_t rows    = 0;
					uint64_t columns = 0;
					
					stream.read((char*)&rows,    sizeof(uint64_t));
					stream.read((char*)&columns, sizeof(uint64_t));

					util::log("ClassificationModel") << "    matrix(" << rows
						<< " rows, " << columns << " columns)\n";
					
					layer.push_back(matrix::Matrix());
					
					auto& matrix = layer.back();
					
					matrix.resize(rows, columns);
					
					stream.read((char*)matrix.data().data(), rows * columns * sizeof(float));
				}
				
				matrixArrayVisitor = util::json::Visitor(layerVisitor["biases"]);	
			
				for(auto weightMatrixObject = matrixArrayVisitor.begin_array();
					weightMatrixObject != matrixArrayVisitor.end_array();
					++weightMatrixObject)
				{
					util::json::Visitor weightMatrixVisitor(
						*weightMatrixObject);
					
					std::string filename = weightMatrixVisitor;
					
					std::stringstream stream;
					
					tar.extractFile(filename, stream);
					
					uint64_t rows    = 0;
					uint64_t columns = 0;
					
					stream.read((char*)&rows,    sizeof(uint64_t));
					stream.read((char*)&columns, sizeof(uint64_t));

					util::log("ClassificationModel") << "    bias matrix(" << rows
						<< " rows, " << columns << " columns)\n";
					
					layer.push_back_bias(matrix::Matrix());
					
					auto& matrix = layer.back_bias();
					
					matrix.resize(rows, columns);
					
					stream.read((char*)matrix.data().data(), rows * columns * sizeof(float));
				}
			}
			
			util::json::Visitor outputsVisitor(networkVisitor["output-names"]);	
			
			for(auto outputObject = outputsVisitor.begin_array();
				outputObject != outputsVisitor.end_array(); ++outputObject)
			{
				util::json::Visitor outputVisitor(*outputObject);
				
				network->second.setLabelForOutputNeuron(
					std::distance(outputsVisitor.begin_array(), outputObject),
					outputVisitor);
			}
			
		}
	}
	catch(...)
	{
		delete headerObject;
		_loaded = false;
	
		throw;
	}
	
	delete headerObject;
}
	
void ClassificationModel::clear()
{
	_neuralNetworks.clear();
}

}

}


