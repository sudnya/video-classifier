/*	\file   paths.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  Function implementations of common path manipulations
*/

// Minerva Includes
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/string.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <stdexcept>
#include <fstream>

// System Specific Includes
#ifndef _WIN32
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

std::string stripTrailingSeparators(const std::string& originalPath)
{
    auto path = originalPath;

    while(!path.empty() && path.back() == separator())
    {
        path.resize(path.size() - 1);
    }

    return path;
}

std::string getDirectory(const std::string& originalPath)
{
    auto path = stripTrailingSeparators(originalPath);

	auto position = path.rfind(separator());

	if(position == std::string::npos)
	{
		return "";
	}

	return path.substr(0, position);
}

std::string getFile(const std::string& path)
{
	auto position = path.rfind(separator());

	if(position == std::string::npos)
	{
		return "";
	}

	return path.substr(position + 1);
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
	#ifndef _WIN32
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
	#ifdef _WIN32
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
	#ifndef _WIN32
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

void makeDirectory(const std::string& path)
{
    if(isDirectory(path))
    {
        return;
    }

    if(!isDirectory(getDirectory(path)))
    {
        makeDirectory(getDirectory(path));
    }

    #ifndef _WIN32
    int status = mkdir(path.c_str(), 0755);

    if(status != 0)
    {
        throw std::runtime_error("Failed to make directory '" + path + "'.");
    }
    #else
    assertM(false, "Not implemented for this platform.");
    #endif
}

size_t getFileSize(std::istream& stream)
{
    stream.seekg(0, std::ios::end);
    size_t length = stream.tellg();
    stream.seekg(0, std::ios::beg);

    return length;
}

void copyFile(const std::string& outputPath, const std::string& inputPath)
{
    std::ifstream input(inputPath, std::ios::binary);

    if(!input.good())
    {
        throw std::runtime_error("Failed to load input file '" + inputPath + "' for reading.");
    }

    std::ofstream output(outputPath, std::ios::binary);

    if(!output.good())
    {
        throw std::runtime_error("Failed to open output file '" + outputPath + "' for writing.");
    }

    size_t bufferSize    = 8192;
    size_t remainingSize = getFileSize(input);

    char buffer[bufferSize];

    while(input.good() && remainingSize > 0)
    {
        size_t bytes = std::min(bufferSize, remainingSize);

        input.read(buffer, bytes);

        remainingSize -= bytes;

        if(input.fail())
        {
            throw std::runtime_error("Reading input file '" + inputPath + "' failed.");
        }

        output.write(buffer, bytes);

        if(!output.good())
        {
            throw std::runtime_error("Writing output file '" + outputPath + "' failed.");
        }
    }
}

}

}


