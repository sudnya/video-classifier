/*	\file   FeatureExtractorEngine.h
	\date   Saturday January 18, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the FeatureExtractorEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/classifiers/interface/Engine.h>

// Standard Library Includes
#include <memory>
#include <fstream>

namespace minerva
{

namespace classifiers
{

class FeatureExtractorEngine: public Engine
{
public:
	FeatureExtractorEngine();

public:
	FeatureExtractorEngine(const FeatureExtractorEngine&) = delete;
	FeatureExtractorEngine& operator=(const FeatureExtractorEngine&) = delete;
	
private:
	virtual ResultVector runOnBatch(Matrix&& matrix, Matrix&& reference);

private:	
	virtual void registerModel();
	virtual void closeModel();
};

}

}




