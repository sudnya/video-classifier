/*	\file   ClassifierEngine.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ClassifierEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/classifiers/interface/Engine.h>

// Standard Library Includes
#include <map>

namespace minerva
{

namespace classifiers
{

class ClassifierEngine : public Engine
{
public:
	ClassifierEngine();
	virtual ~ClassifierEngine();

public:
	void setUseLabeledData(bool useIt);

public:
	virtual void reportStatistics(std::ostream& stream) const;

protected:
	virtual ResultVector runOnBatch(Matrix&& inputs, Matrix&& reference);
	virtual bool requiresLabeledData() const;

private:
	bool _shouldUseLabeledData;

};

}

}




