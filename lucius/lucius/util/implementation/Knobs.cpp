/*	\file   Knobs.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Knob class.
*/

// Lucius Includes
#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/KnobFile.h>

#include <lucius/util/interface/SystemCompatibility.h>

// Standard Library Includes
#include <stdexcept>
#include <map>

namespace lucius
{

namespace util
{

class KnobDatabaseImplementation
{
public:
	typedef std::map<std::string, std::string> StringMap;

public:
	KnobDatabaseImplementation();

private:
	void loadKnobFiles();

public:
	StringMap knobs;
};

static KnobDatabaseImplementation database;

KnobDatabaseImplementation::KnobDatabaseImplementation()
{
	loadKnobFiles();
}

void KnobDatabaseImplementation::loadKnobFiles()
{
	// Check for an environment variable
	if(isEnvironmentVariableDefined("LUCIUS_KNOB_FILE"))
	{
		KnobFile knobFile(getEnvironmentVariable("LUCIUS_KNOB_FILE"));

		knobFile.loadKnobs();
	}
}

void KnobDatabase::addKnob(const std::string& name, const std::string& value)
{
	database.knobs[name] = value;
}

void KnobDatabase::setKnob(const std::string& name, const std::string& value)
{
	database.knobs[name] = value;
}

bool KnobDatabase::knobExists(const std::string& knobname)
{
	return database.knobs.count(knobname) != 0;
}

std::string KnobDatabase::getKnobValueAsString(const std::string& knobname)
{
	auto knob = database.knobs.find(knobname);

	if(knob == database.knobs.end())
	{
		throw std::runtime_error("Attempted to use uniniatilized knob '" +
			knobname + "'");
	}

	return knob->second;
}

std::string KnobDatabase::getKnobValue(const std::string& knobname,
	const std::string& defaultValue)
{
	if(!knobExists(knobname))
	{
		return defaultValue;
	}

	return getKnobValueAsString(knobname);
}

}

}






