/*	\file   ClassifierEngine.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ClassifierEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/video/interface/ImageVector.h>

#include <minerva/util/interface/string.h>

// Standard Library Includes
#include <ostream>

// Forward Declarations
namespace minerva { namespace model { class ClassificationModel; } }

namespace minerva
{

namespace classifiers
{

/*! \brief A generic interface to a classifier with support for an arbitrarily
	large input data stream.
 */
class ClassifierEngine
{
public:
	typedef util::StringVector StringVector;
	typedef video::ImageVector ImageVector;

public:
	ClassifierEngine();
	virtual ~ClassifierEngine();

public:
	/*! \brief Load a model from a file */
	void loadModel(const std::string& pathToModelFile);
	
	/*! \brief Run the classifier on all of the contained paths */
	void runOnPaths(const StringVector& paths);
	
public:
	virtual void reportStatistics(std::ostream& stream) const;

protected:
	virtual void registerModel();
	virtual void closeModel();

protected:

	virtual void runOnImageBatch(const ImageVector& images) = 0;
	virtual size_t getInputFeatureCount() const = 0;
	
public:
	ClassifierEngine(const ClassifierEngine&) = delete;
	ClassifierEngine& operator=(const ClassifierEngine&) = delete;

protected:
	typedef model::ClassificationModel ClassificationModel;

protected:
	ClassificationModel* _model;

};

}

}


