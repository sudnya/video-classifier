/*	\file   paths.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  Function implementations of common path manipulations
*/

// Minerva Includes
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/string.h>

namespace minerva
{

namespace util
{

char separator()
{
	return '/';
}

std::string getExtension(const std::string& path)
{
	auto components = split(path, ".");
	
	if(components.size() < 2)
	{
		return "";
	}
	
	return "." + components.back();
}

std::string getDirectory(const std::string& path)
{
	auto position = path.rfind(separator());
	
	if(position == std::string::npos)
	{
		return "";
	}
	
	return path.substr(0, position);
}

std::string getRelativePath(const std::string& baseDirectory,
	const std::string& path)
{
	if(isAbsolutePath(path)) return path;
	
	return joinPaths(baseDirectory, path);
}

std::string joinPaths(const std::string& left, const std::string& right)
{
	return left + separator() + right;
}

bool isAbsolutePath(const std::string& path)
{
	if(path.empty()) return false;
	
	return path.front() == separator();
}


}

}


