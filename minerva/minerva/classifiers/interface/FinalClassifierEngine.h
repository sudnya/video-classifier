/*	\file   FinalClassifierEngine.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the FinalClassifierEngine class.
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

class FinalClassifierEngine : public Engine
{
public:
	FinalClassifierEngine();
	virtual ~FinalClassifierEngine();

public:
	void useLabeledData(bool useIt);

public:
	virtual void reportStatistics(std::ostream& stream) const;

protected:
	virtual void runOnImageBatch(ImageVector&& images);
	virtual size_t getInputFeatureCount() const;
	
	virtual bool requiresLabeledData() const;

private:

	class LabelStatistic
	{
	public:
		LabelStatistic(const std::string& label, size_t truePositives,
			size_t trueNegatives, size_t falsePositives,
			size_t falseNegatives);

	public:
		std::string label;

		size_t truePositives;
		size_t trueNegatives;
		size_t falsePositives;
		size_t falseNegatives;
	};

	typedef std::map<std::string, LabelStatistic> LabelStatisticMap;

	class Statistics
	{
	public:
		Statistics();

	public:
		std::string toString() const;
	
	public:
		LabelStatisticMap labelStatistics;

		size_t exactMatches;
		size_t totalSamples;
		
	};
	
	typedef video::Image Image;

private:
	void _updateStatistics(const std::string& label, const Image& image);

private:
	Statistics _statistics;
	bool       _shouldUseLabeledData;

};

}

}




