/*	\file   Knobs.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Knob class.
*/

// Lucious Includes
#include <lucious/util/interface/Knobs.h>
#include <lucious/util/interface/KnobFile.h>

#include <lucious/util/interface/SystemCompatibility.h>

// Standard Library Includes
#include <stdexcept>
#include <map>

namespace lucious
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
	if(isEnvironmentVariableDefined("LUCIOUS_KNOB_FILE"))
	{
		KnobFile knobFile(getEnvironmentVariable("LUCIOUS_KNOB_FILE"));

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






