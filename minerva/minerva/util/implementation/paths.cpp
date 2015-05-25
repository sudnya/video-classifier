/*	\file   paths.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  Function implementations of common path manipulations
*/

// Minerva Includes
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/string.h>

// Standard Library Includes
#include <stdexcept>

// System Specific Includes
#ifdef __APPLE__
#include <sys/stat.h>
#include <dirent.h>
#endif

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

std::string stripExtension(const std::string& path)
{
	auto position = path.rfind(".");

	if(position == std::string::npos)
	{
		return path;
	}

	return path.substr(0, position);
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

static void listDirectoryRecursively(StringVector& files, const std::string& path)
{
	#ifdef __APPLE__
	DIR* directory = opendir(path.c_str());
	
	if(directory == nullptr)
	{
		throw std::runtime_error("Could not open directory '" + path + "'");
	}
	
	while(true)
	{
		auto entry = readdir(directory);
		
		if(entry == nullptr)
		{
			break;
		}
		
		auto name = std::string(entry->d_name);
	
		// skip the current and previous directory
		if(name == ".." || name == ".")
		{
			continue;
		}

		if(isDirectory(name))
		{
			listDirectoryRecursively(files, joinPaths(path, name));
		}
		else
		{
			files.push_back(name);
		}
	}
	
	closedir(directory);	

	#endif
}

StringVector listDirectoryRecursively(const std::string& path)
{
	StringVector files;
	
	listDirectoryRecursively(files, path);

	return files;
}

bool isFile(const std::string& path)
{
	#ifdef __APPLE__
	struct stat fileStats;
	
	auto result = stat(path.c_str(), &fileStats);
	
	if(result != 0)
	{
		return false;
	}
	
	return S_ISREG(fileStats.st_mode);
	
	#else
	assertM(false, "Not implemented for this platform.");
	#endif
}

bool isDirectory(const std::string& path)
{
	#ifdef __APPLE__
	struct stat fileStats;
	
	auto result = stat(path.c_str(), &fileStats);
	
	if(result != 0)
	{
		return false;
	}
	
	return S_ISDIR(fileStats.st_mode);
	
	#else
	assertM(false, "Not implemented for this platform.");
	#endif
}

}

}


