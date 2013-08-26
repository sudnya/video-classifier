/*	\file   paths.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  Function headers for common path manipulations
*/

#pragma once

// Standard Library Includes
#include <string>

namespace minerva
{

namespace util
{

char separator();

std::string getExtension(const std::string& path);

std::string getDirectory(const std::string& path);

std::string getRelativePath(const std::string& baseDirectory,
	const std::string& path);

std::string joinPaths(const std::string& left, const std::string& right);

bool isAbsolutePath(const std::string& path);

}

}


