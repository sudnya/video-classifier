/*	\file   paths.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  Function headers for common path manipulations.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

namespace minerva
{

namespace util
{

char separator();

typedef std::vector<std::string> StringVector;

std::string getExtension(const std::string& path);
std::string stripExtension(const std::string& path);
std::string stripTrailingSeparators(const std::string& path);

std::string getDirectory(const std::string& path);
std::string getFile(const std::string& path);

std::string getRelativePath(const std::string& baseDirectory,
	const std::string& path);

std::string joinPaths(const std::string& left, const std::string& right);

bool isAbsolutePath(const std::string& path);

StringVector listDirectoryRecursively(const std::string& path);

bool isFile(const std::string& path);
bool isDirectory(const std::string& path);

void makeDirectory(const std::string& path);

size_t getFileSize(std::istream& stream);
void copyFile(const std::string& outputPath, const std::string& inputPath);

}

}


