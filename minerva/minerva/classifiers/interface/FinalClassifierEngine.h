/*	\file   FinalClassifierEngine.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the FinalClassifierEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierEngine.h>

namespace minerva
{

namespace classifiers
{

class FinalClassifierEngine : public ClassifierEngine
{
public:
	FinalClassifierEngine();
	
public:
	virtual void reportStatistics(std::ostream& stream) const;

protected:
	virtual void runOnImageBatch(const ImageVector& images);
	virtual size_t getInputFeatureCount() const;

private:
	class Statistics
	{
	public:
		std::string toString() const;
	
	public:
		// TODO
		
	};

private:
	Statistics _statistics;

};

}

}




