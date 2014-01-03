/*! \file   SampleDatabaseParser.cpp
	\date   Saturday December 6, 2013
	\author Gregory Diamos <solusstutus@gmail.com>
	\brief  The source file for the SampleDatabaseParser class.
*/

// Minerva Includes
#include <minerva/database/interface/SampleDatabaseParser.h>
#include <minerva/database/interface/SampleDatabase.h>
#include <minerva/database/interface/Sample.h>

#include <minerva/video/interface/Video.h>
#include <minerva/video/interface/Image.h>

#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/string.h>

// Standard Library Includes
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace minerva
{

namespace database
{

SampleDatabaseParser::SampleDatabaseParser(SampleDatabase* database)
: _database(database)
{

}

static void parseLabeledPath(SampleDatabase* database, const std::string& line,
	const std::string& directory);
static void parseUnlabeledPath(SampleDatabase* database, const std::string& line,
	const std::string& directory);
static void parseLabelDeclaration(SampleDatabase* database, const std::string& line);

static bool isComment(const std::string& line);
static bool isLabeled(const std::string& line);
static bool isLabelDeclaration(const std::string& line);
static std::string removeWhitespace(const std::string& line);

void SampleDatabaseParser::parse()
{
	std::ifstream file(_database->path().c_str());
	
	if(!file.is_open())
	{
		throw std::runtime_error("Could not open '" +
			_database->path() + "' for reading.");
	}

	auto databaseDirectory = util::getDirectory(_database->path());
	
	while(file.good())
	{
		std::string line;
		
		std::getline(file, line);
		
		line = removeWhitespace(line);
		
		if(line.empty()) continue;
		
		if(isComment(line)) continue;

		if(isLabelDeclaration(line))
		{
			parseLabelDeclaration(_database, line);
			continue;
		}

		if(isLabeled(line))
		{
			parseLabeledPath(_database, line, databaseDirectory);
		}
		else
		{
			parseUnlabeledPath(_database, line, databaseDirectory);
		}
	}

}

static unsigned int parseInteger(const std::string s);

static void parseLabeledPath(SampleDatabase* database, const std::string& line,
	const std::string& databaseDirectory)
{
	auto components = util::split(line, ",");
	
	if(components.size() < 2)
	{
		throw std::runtime_error("Malformed labeled image/video statement '" +
			line + "', should be (path, label) or "
			"(path, label, startFrame, endFrame).");
	}
	
	auto filePath = util::getRelativePath(databaseDirectory,
		removeWhitespace(components[0]));
	
	auto label = removeWhitespace(components[1]);
	
	if(video::Image::isPathAnImage(filePath))
	{
		if(components.size() != 2)
		{
			throw std::runtime_error("Malformed labeled image statement '" +
				line + "', should be (path, label).");
		}

		database->addSample(Sample(filePath, label));
	}
	else if(video::Video::isPathAVideo(filePath))
	{
		if(components.size() != 4)
		{
			throw std::runtime_error("Malformed labeled video statement '" +
				line + "', should be (path, label, startFrame, endFrame).");
		}
		
		unsigned int startFrame = parseInteger(components[2]);
		unsigned int endFrame   = parseInteger(components[3]);
	
		database->addSample(Sample(filePath, label, startFrame, endFrame));
	}
	else
	{
		throw std::runtime_error("Path '" + filePath + " with extension '" +
			util::getExtension(filePath) +
			"' is not an image or video.");
	}
}

static unsigned int parseInteger(const std::string s)
{
	std::stringstream stream;
	
	stream << s;
	
	unsigned int value = 0;
	
	stream >> value;
	
	return value;
}

static void parseUnlabeledPath(SampleDatabase* database,
	const std::string& line, const std::string& databaseDirectory)
{
	auto filePath = util::getRelativePath(databaseDirectory, line);
	
	database->addSample(Sample(filePath));	
}

static void parseLabelDeclaration(SampleDatabase* database,
	const std::string& line)
{
	auto position = line.find(":");
	
	auto components = util::split(line.substr(position + 1), ",");

	for(auto& component : components)
	{
		database->addLabel(removeWhitespace(component));
	}
}

static bool isComment(const std::string& line)
{
	auto comment = removeWhitespace(line);
	
	if(comment.empty())
	{
		return false;
	}
	
	return comment.front() == '#';
}

static bool isLabeled(const std::string& line)
{
	return line.find(",") != std::string::npos;
}

static bool isLabelDeclaration(const std::string& line)
{
	return removeWhitespace(line).find("labels:") == 0;
}

static bool isWhitespace(char c);

static std::string removeWhitespace(const std::string& line)
{
	unsigned int begin = 0;
	
	for(; begin != line.size(); ++begin)
	{
		if(!isWhitespace(line[begin])) break;
	}
	
	unsigned int end = line.size();
	
	for(; end != 0; --end)
	{
		if(!isWhitespace(line[end - 1])) break;
	}	
	
	return line.substr(begin, end - begin);
}

static bool isWhitespace(char c)
{
	return (c == ' ') || (c == '\n') || (c == '\t') || (c == '\r');
}

}

}


