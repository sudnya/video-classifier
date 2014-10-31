/*	\file   UnsupervisedLearnerEngine.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the UnsupervisedLearnerEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierEngine.h>

namespace minerva
{

namespace classifiers
{


/*! \brief Performs unsupervised learning on a given model. */
class UnsupervisedLearnerEngine : public ClassifierEngine
{
public:
	UnsupervisedLearnerEngine();
	virtual ~UnsupervisedLearnerEngine();

public:
	UnsupervisedLearnerEngine(const UnsupervisedLearnerEngine&) = delete;
	UnsupervisedLearnerEngine& operator=(const UnsupervisedLearnerEngine&) = delete;

public:
	void setLayersPerIteration(size_t layers);
	
private:
	virtual void registerModel();
	virtual void closeModel();
	
private:
	virtual void runOnBatch(Matrix&& samples);

private:
	size_t _layersPerIteration;

private:
	typedef std::map<std::string, NeuralNetwork> NetworkMap;

private:
	NetworkMap _augmentorNetworks;

};

}

}




