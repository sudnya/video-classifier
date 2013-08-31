/*	\file   ClassifierEngine.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ClassifierEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierEngine.h>
#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/video/interface/Image.h>
#include <minerva/video/interface/Video.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <stdexcept>
#include <fstream>

namespace minerva
{

namespace classifiers
{

ClassifierEngine::ClassifierEngine()
: _model(nullptr)
{

}

ClassifierEngine::~ClassifierEngine()
{
	delete _model;
}

void ClassifierEngine::loadModel(const std::string& pathToModelFile)
{
	util::log("ClassifierEngine") << "Loading model file '" << pathToModelFile
		<<  "'...\n";
	
	_model = new ClassificationModel(pathToModelFile);

	util::log("ClassifierEngine") << " model loaded.\n";
}

typedef video::Image	   Image;
typedef video::ImageVector ImageVector;
typedef video::Video	   Video;
typedef video::VideoVector VideoVector;

static void parseImageDatabase(ImageVector& images, VideoVector& video,
	const std::string& path, bool requiresLabeledData);

void ClassifierEngine::runOnPaths(const StringVector& paths)
{
	_model->load();

	registerModel();
		
	if(paths.empty())
	{
		throw std::runtime_error("No input path provided.");
	}
	
	ImageVector images;
	VideoVector videos;
	
	util::log("ClassifierEngine") << "Scanning for input images...\n";
	
	for(auto& path : paths)
	{
		if(Image::isPathAnImage(path))
		{
			util::log("ClassifierEngine") << " found image '" << path << "'\n";
			images.push_back(Image(path));
		}
		else if(Video::isPathAVideo(path))
		{
			util::log("ClassifierEngine") << " found video '" << path << "'\n";
			videos.push_back(Video(path));
		}
		else
		{
			parseImageDatabase(images, videos, path, requiresLabeledData());
		}
	}
	
	unsigned int maxBatchSize = util::KnobDatabase::getKnobValue<unsigned int>(
		"ClassifierEngine::ImageBatchSize", 10);
	
	util::log("ClassifierEngine") << "Running image batches\n";

	// Run images first
	for(unsigned int i = 0; i < images.size(); i += maxBatchSize)
	{
		unsigned int batchSize = std::min(images.size() - i,
			(size_t)maxBatchSize);
	
		ImageVector batch;
		
		for(unsigned int j = 0; j < batchSize; ++j)
		{
			batch.push_back(images[j]);
		}
		
		for(auto& image : batch)
		{
			image.load();
			image = image.sample(getInputFeatureCount());
		}

		util::log("ClassifierEngine") << " running batch with " << batch.size()
			<< " images\n";
		
		runOnImageBatch(batch);
	}
	
    unsigned int maxVideoFrames = util::KnobDatabase::getKnobValue<unsigned int>(
		"ClassifierEngine::MaximumVideoFrames", 500);
	
	// Run videos next
	for(auto& video : videos)
	{
		while(!video.finished())
		{
			auto batch = video.getNextFrames(maxBatchSize);
		
			runOnImageBatch(batch);

            if(maxVideoFrames <= batch.size())
            {
                break;
            }
		
            maxVideoFrames -= batch.size();
        }
	}	
	
	// close
	closeModel();
}

void ClassifierEngine::reportStatistics(std::ostream& stream) const
{
	// intentionally blank
}


void ClassifierEngine::registerModel()
{
	// intentionally blank
}

void ClassifierEngine::closeModel()
{
	// intentionally blank
}
	
bool ClassifierEngine::requiresLabeledData() const
{
	return false;
}

static bool isComment(const std::string& line);
static bool isLabeled(const std::string& line);
static std::string removeWhitespace(const std::string& line);
static void parseSinglePath(ImageVector& images, VideoVector& videos,
	const std::string& line, const std::string& databaseDirectory);
static void parseLabeledPath(ImageVector& images, VideoVector& videos,
	const std::string& line, const std::string& databaseDirectory);
static void consolidateLabels(ImageVector& images, VideoVector& videos);

static void parseImageDatabase(ImageVector& images, VideoVector& videos,
	const std::string& path, bool requiresLabeledData)
{
	util::log("ClassifierEngine") << " scanning image database '"
		<< path << "'\n";
	
	std::ifstream file(path.c_str());
	
	if(!file.is_open())
	{
		throw std::runtime_error("Could not open '" + path + "' for reading.");
	}

	auto databaseDirectory = util::getDirectory(path);
	
	while(file.good())
	{
		std::string line;
		
		std::getline(file, line);
		
		line = removeWhitespace(line);
		
		if(line.empty()) continue;
		
		if(isComment(line)) continue;

		if(isLabeled(line))
		{
			parseLabeledPath(images, videos, line, databaseDirectory);
		}
		else if(!requiresLabeledData)
		{
			parseSinglePath(images, videos, line, databaseDirectory);
		}
		else
		{
			util::log("ClassifierEngine") << "  skipped unlabeled data '"
				<< line	<< "'\n";
		}

		consolidateLabels(images, videos);

	}
}

static void parseSinglePath(ImageVector& images, VideoVector& videos,
	const std::string& line, const std::string& databaseDirectory)
{
	auto filePath = util::getRelativePath(databaseDirectory, line);
	
	if(Image::isPathAnImage(filePath))
	{
		util::log("ClassifierEngine") << "  found image '" << filePath
			<< "'\n";
		
		images.push_back(Image(filePath));
	}
	else if(Video::isPathAVideo(filePath))
	{
		util::log("ClassifierEngine") << "  found video '" << filePath
			<< "'\n";

		videos.push_back(Video(filePath));
	}
	else
	{
		throw std::runtime_error("Path '" + filePath +
			"' is not an image or video.");
	}
}

static unsigned int parseInteger(const std::string s);

static void parseLabeledPath(ImageVector& images, VideoVector& videos,
	const std::string& line, const std::string& databaseDirectory)
{
	auto components = util::split(line, ",");
	
	if(components.size() < 2)
	{
		throw std::runtime_error("Malformed labeled image/video statement '" +
			line + "', should be (path, label) or "
			"(path, label, startFrame, endFrame).");
	}
	
	auto filePath = util::getRelativePath(databaseDirectory, components[0]);
	
	auto label = removeWhitespace(components[1]);
	
	if(Image::isPathAnImage(filePath))
	{
		util::log("ClassifierEngine") << "  found labeled image '" << filePath
			<< "'\n";

		if(components.size() != 2)
		{
			throw std::runtime_error("Malformed labeled image statement '" +
				line + "', should be (path, label).");
		}

		images.push_back(Image(filePath, label));
	}
	else if(Video::isPathAVideo(filePath))
	{
		util::log("ClassifierEngine") << "  found labeled video '" << filePath
			<< "'\n";
		
		if(components.size() != 4)
		{
			throw std::runtime_error("Malformed labeled video statement '" +
				line + "', should be (path, label, startFrame, endFrame).");
		}
		
		unsigned int startFrame = parseInteger(components[2]);
		unsigned int endFrame   = parseInteger(components[3]);
	
		videos.push_back(Video(filePath, label, startFrame, endFrame));
	}
	else
	{
		throw std::runtime_error("Path '" + filePath +
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

static void consolidateLabels(ImageVector& images, VideoVector& videos)
{
	typedef std::map<std::string, Video> VideoMap;
	
	VideoMap pathsToVideos;
	
	for(auto& video : videos)
	{
		auto existingVideo = pathsToVideos.find(video.path());
		
		if(existingVideo == pathsToVideos.end())
		{
			pathsToVideos.insert(std::make_pair(video.path(), video));
		}
		else
		{
			auto labels = video.getLabels();
		
			for(auto& label : labels)
			{
				existingVideo->second.addLabel(label);
			}
		}
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


