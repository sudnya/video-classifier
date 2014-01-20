/*	\file   FeatureExtractorEngine.cpp
	\date   Saturday January 18, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the FeatureExtractorEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/FeatureExtractorEngine.h>
#include <minerva/classifiers/interface/FeatureExtractor.h>

#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace classifiers
{

FeatureExtractorEngine::FeatureExtractorEngine()
{

}

typedef std::vector<std::string> StringVector;
typedef matrix::Matrix Matrix;
typedef video::ImageVector ImageVector;

static void appendFeatures(std::ostream& stream, const Matrix& matrix,
	const StringVector& labels)
{
	assert(labels.size() == matrix.rows());

	auto label = labels.begin();
	for(size_t row = 0; row != matrix.rows(); ++row, ++label)
	{
		stream << "\"" << *label << "\" ";
		
		for(size_t column = 0; column != matrix.columns(); ++column)
		{
			stream << ", " << matrix(row, column);
		}
		
		stream << "\n";
	}
}

static StringVector extractLabels(const ImageVector& images)
{
	StringVector result;
	
	for(auto& image : images)
	{
		if(image.hasLabel())
		{
			result.push_back(image.label());
		}
		else
		{
			result.push_back("");
		}
	}
	
	return result;
}

void FeatureExtractorEngine::runOnImageBatch(ImageVector&& images)
{
	FeatureExtractor extractor(_model);
	
	auto labels   = extractLabels(images);
	auto features = extractor.extract(std::move(images));
	
	appendFeatures(*_outputFile, features, labels);
}

size_t FeatureExtractorEngine::getInputFeatureCount() const
{
	FeatureExtractor extractor(_model);
	
	return extractor.getInputFeatureCount();
}
	
void FeatureExtractorEngine::registerModel()
{
	auto filename = _outputFilename;
	
	if(util::KnobDatabase::knobExists("FeatureExtractor::OutputFilename"))
	{
		filename = util::KnobDatabase::getKnobValue("FeatureExtractor::OutputFilename", "");
	}
	
	_outputFile.reset(new std::ofstream(filename));
}

void FeatureExtractorEngine::closeModel()
{
	_outputFile.reset();
}

}

}

