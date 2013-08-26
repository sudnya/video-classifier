/*	\file   FinalClassifierEngine.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the FinalClassifierEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/FinalClassifierEngine.h>
#include <minerva/classifiers/interface/Classifier.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace classifiers
{

FinalClassifierEngine::FinalClassifierEngine()
{

}	

void FinalClassifierEngine::reportStatistics(std::ostream& stream) const
{
	stream << _statistics.toString();
}

void FinalClassifierEngine::runOnImageBatch(const ImageVector& images)
{
	Classifier classifier(*_model);
	
	auto labels = classifier.classify(images);
	
	assert(labels.size() == images.size());

	auto image = images.begin();
	for(auto label = labels.begin(); label != labels.end() &&
		image != images.end(); ++label, ++image)
	{
		util::log("FinalClassifierEngine") << " Classified '" << image->path()
			<< "' as '" << *label << "'\n";
	}

	// TODO interpret labels, update statistics
}

size_t FinalClassifierEngine::getInputFeatureCount() const
{
	Classifier classifier(*_model);
	
	return classifier.getInputFeatureCount();
}

std::string FinalClassifierEngine::Statistics::toString() const
{
	return "";
}

}

}




