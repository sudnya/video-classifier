/*	\file   TarArchive.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  An interface to a tar archive.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

namespace lucious
{

namespace util
{

class OutputTarArchiveImplementation;

class OutputTarArchive
{
public:
	OutputTarArchive(std::ostream& );
	~OutputTarArchive();

public:
	OutputTarArchive(const OutputTarArchive&) = delete;
	OutputTarArchive& operator=(const OutputTarArchive&) = delete;

public:
	/*! \brief Add a file to the archive */
	void addFile(const std::string& name, std::istream& file);

private:
	OutputTarArchiveImplementation* _archive;

};

class InputTarArchiveImplementation;

class InputTarArchive
{
public:
	typedef std::vector<std::string> StringVector;

public:
	InputTarArchive(std::istream& );
	~InputTarArchive();

public:
	InputTarArchive(const InputTarArchive&) = delete;
	InputTarArchive& operator=(const InputTarArchive&) = delete;

public:
	/*! \brief Get a list of all contained files */
	StringVector list() const;

public:
	/*! \brief Extract a file from the archive */
	void extractFile(const std::string& name, std::ostream& file);

private:
	InputTarArchiveImplementation* _archive;

};

}

}



