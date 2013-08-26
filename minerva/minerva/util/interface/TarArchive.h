/*	\file   TarArchive.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  An interface to a tar archive.
	
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

namespace minerva
{

namespace util
{

class TarArchiveImplementation;

class TarArchive
{
public:
	typedef std::vector<std::string> StringVector;

public:
	TarArchive(const std::string& path, const std::string& mode = "r:gz");
	~TarArchive();

public:
	TarArchive(const TarArchive&) = delete;
	TarArchive& operator=(const TarArchive&) = delete;

public:
	/*! \brief Get a list of all contained files */
	StringVector list() const;

public:
	/*! \brief Add a file to the archive */
	void addFile(const std::string& name, std::istream& file);
	
	/*! \brief Extract a file from the archive */
	void extractFile(const std::string& name, std::ostream& file);
	
private:
	TarArchiveImplementation* _archive;

};

}

}



