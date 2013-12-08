/*! \file   SampleDatabase.cpp
	\date   Saturday December 6, 2013
	\author Gregory Diamos <solusstutus@gmail.com>
	\brief  The source file for the SampleDatabase class.
*/

// Minerva Includes
#include <minerva/database/interface/SampleDatabase.h>
#include <minerva/database/interface/SampleDatabaseParser.h>
#include <minerva/database/interface/Sample.h>

namespace minerva
{

namespace database
{

SampleDatabase::SampleDatabase(const std::string& path)
: _path(path)
{
	_parse();
}

SampleDatabase::StringVector SampleDatabase::getAllPossibleLabels() const
{
	return StringVector(_labels.begin(), _labels.end());
}

size_t SampleDatabase::getTotalLabelCount() const
{
	return _labels.size();
}

void SampleDatabase::_parse()
{
	SampleDatabaseParser parser(this);

	parser.parse();
}

SampleDatabase::iterator SampleDatabase::begin()
{
	return _samples.begin();
}

SampleDatabase::const_iterator SampleDatabase::begin() const
{
	return _samples.begin();
}

SampleDatabase::iterator SampleDatabase::end()
{
	return _samples.end();
}

SampleDatabase::const_iterator SampleDatabase::end() const
{
	return _samples.end();
}

size_t SampleDatabase::size() const
{
	return _samples.size();
}

bool SampleDatabase::empty() const
{
	return _samples.empty();
}

const std::string& SampleDatabase::path() const
{
	return _path;
}

void SampleDatabase::addSample(const Sample& sample)
{
	if(sample.hasLabel())
	{
		_labels.insert(sample.label());
	}

	return _samples.push_back(sample);
}

void SampleDatabase::addLabel(const std::string& label)
{
	_labels.insert(label);
}

}

}

