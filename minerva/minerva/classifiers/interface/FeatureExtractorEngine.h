/*	\file   FeatureExtractorEngine.h
	\date   Saturday January 18, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the FeatureExtractorEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierEngine.h>

// Standard Library Includes
#include <memory>
#include <fstream>

namespace minerva
{

namespace classifiers
{

class FeatureExtractorEngine: public ClassifierEngine
{
public:
	FeatureExtractorEngine();

public:
	FeatureExtractorEngine(const FeatureExtractorEngine&) = delete;
	FeatureExtractorEngine& operator=(const FeatureExtractorEngine&) = delete;
	
private:
	virtual void   runOnImageBatch(ImageVector&& images);
	virtual size_t getInputFeatureCount() const;

private:	
	virtual void registerModel();
	virtual void closeModel();

private:
	std::unique_ptr<std::ofstream> _outputFile;
};

}

}




