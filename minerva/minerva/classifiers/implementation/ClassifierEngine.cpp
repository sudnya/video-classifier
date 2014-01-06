/*	\file   ClassifierEngine.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ClassifierEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierEngine.h>

#include <minerva/model/interface/ClassificationModel.h>

#include <minerva/database/interface/SampleDatabase.h>
#include <minerva/database/interface/Sample.h>

#include <minerva/video/interface/Image.h>
#include <minerva/video/interface/Video.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <stdexcept>
#include <fstream>
#include <random>
#include <cstdlib>
#include <algorithm>

namespace minerva
{

namespace classifiers
{

ClassifierEngine::ClassifierEngine()
: _model(nullptr), _ownModel(true), _maximumSamplesToRun(0),
	_batchSize(0), _areMultipleSamplesAllowed(false), _shouldDisplayImages(false)
{
	_maximumSamplesToRun = util::KnobDatabase::getKnobValue<unsigned int>(
		"ClassifierEngine::MaximumVideoFrames", 20000000);
	_batchSize = util::KnobDatabase::getKnobValue<unsigned int>(
		"ClassifierEngine::ImageBatchSize", 500);
}

ClassifierEngine::~ClassifierEngine()
{
	if(_ownModel)
	{
		delete _model;
	}
}

void ClassifierEngine::setModel(ClassificationModel* model)
{
	if(_ownModel)
	{
		delete _model;
	}

	_ownModel = false;
	_model = model;
}

void ClassifierEngine::loadModel(const std::string& pathToModelFile)
{
	util::log("ClassifierEngine") << "Loading model file '" << pathToModelFile
		<<  "'...\n";
	
	_model = new ClassificationModel(pathToModelFile);
	_ownModel = true;

	util::log("ClassifierEngine") << " model loaded.\n";
}

typedef video::Image	   Image;
typedef video::ImageVector ImageVector;
typedef video::Video	   Video;
typedef video::VideoVector VideoVector;

static void parseImageDatabase(ImageVector& images, VideoVector& video,
	const std::string& path, bool requiresLabeledData);
static void runAllImages(ClassifierEngine* engine, ImageVector& images,
	unsigned int maxBatchSize, unsigned int& maxVideoFrames);
static void runAllVideos(ClassifierEngine* engine, VideoVector& images,
	unsigned int maxBatchSize, unsigned int& maxVideoFrames, bool requiresLabeledData);

void ClassifierEngine::runOnDatabaseFile(const std::string& path)
{
	runOnPaths(StringVector(1, path));
}

void ClassifierEngine::runOnPaths(const StringVector& paths)
{
	if(_ownModel) _model->load();

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
	
	unsigned int maxBatchSize = _batchSize;

	unsigned int maxVideoFrames = _maximumSamplesToRun;
	
	do
	{
		// Run images first
		runAllImages(this, images, maxBatchSize, maxVideoFrames);
		
		// Run videos next
		runAllVideos(this, videos, maxBatchSize, maxVideoFrames,
			requiresLabeledData());
	}
	while(maxVideoFrames > 0 && _areMultipleSamplesAllowed);

	// close
	closeModel();
}

void ClassifierEngine::setMaximumSamplesToRun(unsigned int samples)
{
	_maximumSamplesToRun = samples;
}

void ClassifierEngine::setBatchSize(unsigned int samples)
{
	_batchSize = samples;
}

void ClassifierEngine::setMultipleSamplesAllowed(bool allowed)
{
	_areMultipleSamplesAllowed = allowed;
}

void ClassifierEngine::setDisplayImages(bool shouldDisplay)
{
	_shouldDisplayImages = shouldDisplay;
}

std::string ClassifierEngine::reportStatisticsString() const
{
	std::stringstream stream;

	reportStatistics(stream);
	
	return stream.str();
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

void ClassifierEngine::saveModel()
{
	if(_ownModel) _model->save();
}

static void consolidateLabels(ImageVector& images, VideoVector& videos);

static void parseImageDatabase(ImageVector& images, VideoVector& videos,
	const std::string& path, bool requiresLabeledData)
{
	util::log("ClassifierEngine") << " scanning image database '"
		<< path << "'\n";

	database::SampleDatabase sampleDatabase(path);

	for(auto& sample : sampleDatabase)
	{
		if(!sample.hasLabel() && requiresLabeledData)
		{
			util::log("ClassifierEngine") << "  skipped unlabeled data '"
				<< sample.path() << "'\n";
			continue;
		}

		if(sample.isImageSample())
		{

			if(sample.hasLabel())
			{
				util::log("ClassifierEngine") << "  found labeled image '" << sample.path()
					<< "' with label '" << sample.label() << "'\n";
			}
			else
			{
				util::log("ClassifierEngine") << "  found unlabeled image '" << sample.path()
					<< "'n";
			}
			
			images.push_back(Image(sample.path(), sample.label()));
		}
		else if(sample.isVideoSample())
		{
			if(sample.hasLabel())
			{
				util::log("ClassifierEngine") << "  found labeled video '" << sample.path()
					<< "' with label '" << sample.label() << "'\n";
			}
			else
			{
				util::log("ClassifierEngine") << "  found unlabeled video '" << sample.path()
					<< "'n";
			}

			videos.push_back(Video(sample.path(), sample.label(),
				sample.beginFrame(), sample.endFrame()));
		}
	}
	
	consolidateLabels(images, videos);
}

typedef std::vector<std::pair<unsigned int, unsigned int>> VideoAndFrameVector;

static VideoAndFrameVector pickRandomFrames(VideoVector& videos,
	unsigned int maxVideoFrames, bool requiresLabeledData);

static void runAllVideos(ClassifierEngine* engine, VideoVector& videos,
	unsigned int maxBatchSize, unsigned int& maxVideoFrames,
	bool requiresLabeledData)
{
	util::log("ClassifierEngine") << "Running video batches\n";
	
	auto frames = pickRandomFrames(videos, maxVideoFrames, requiresLabeledData);

	for(unsigned int i = 0; i < frames.size(); i += maxBatchSize)
	{
		unsigned int batchSize = std::min(frames.size() - i, 
			(size_t)maxBatchSize);
			
		ImageVector batch;
		
		for(unsigned int j = 0; j < batchSize; ++j)
		{
			unsigned int video = frames[i + j].first;
			unsigned int frame = frames[i + j].second;

			util::log("ClassifierEngine") << " Getting frame (" << frame
				<< ") from video " << videos[video].path() << "\n"; 
			
			batch.push_back(videos[video].getSpecificFrame(frame));
		}
		
		engine->runOnImageBatch(std::move(batch));

		if(batch.size() < maxVideoFrames)
		{
			maxVideoFrames -= batch.size();
		}
		else
		{
			maxVideoFrames = 0;
			break;
		}
	}
}

static unsigned int mapFrameIndexToLabeledFrameIndex(unsigned int frame, const Video& video)
{
	unsigned int totalLabeledFrames = 0;

	auto labeledFrames = video.getLabels();

	for(auto& label : labeledFrames)
	{
		totalLabeledFrames += label.coveredFrames();
	}

	unsigned int labeledFrame = frame % totalLabeledFrames;
	
	unsigned int framePrefix = 0;

	for(auto& label : labeledFrames)
	{
		if(labeledFrame - framePrefix < label.coveredFrames())
		{
			framePrefix = label.beginFrame + labeledFrame - framePrefix;
			break;
		}

		framePrefix += label.coveredFrames();
	}

	return framePrefix;
}

static VideoAndFrameVector pickRandomFrames(VideoVector& videos,
	unsigned int maxVideoFrames, bool requiresLabeledData)
{
	util::log("ClassifierEngine") << " Picking random " << maxVideoFrames
		<< " frames from videos\n"; 

	if(videos.size() == 0)
	{
		return VideoAndFrameVector();
	}

	std::default_random_engine generator;
	
	bool shouldSeedWithTime = util::KnobDatabase::getKnobValue(
		"ClassifierEngine::SeedWithTime", false);
	if(shouldSeedWithTime)
	{
		generator.seed(std::time(0));
	}

	VideoAndFrameVector positions;
	
	for(unsigned int i = 0; i < maxVideoFrames; ++i)
	{
		unsigned int video = generator() % videos.size();
		unsigned int frames = videos[video].getTotalFrames();
		
		if(frames == 0)
		{
			continue;
		}

		unsigned int frame = generator() % frames;

		if(requiresLabeledData)
		{
			frame = mapFrameIndexToLabeledFrameIndex(frame, videos[video]);
		}

		util::log("ClassifierEngine") << "  Video " << videos[video].path()
			<< " has " << frames << " frames\n"; 
		
		positions.push_back(std::make_pair(video, frame));
	}
		
	return positions;
}

typedef std::vector<unsigned int> IntVector;

IntVector getRandomOrder(unsigned int size)
{
	IntVector order(size);

	for(unsigned int i = 0; i < size; ++i)
	{
		order[i] = i;
	}

	std::default_random_engine generator;
	
	bool shouldSeedWithTime = util::KnobDatabase::getKnobValue(
		"ClassifierEngine::SeedWithTime", true);
	if(shouldSeedWithTime)
	{
		generator.seed(std::time(0));
	}
	
	std::shuffle(order.begin(), order.end(), generator);	

	return order;
}

static void runAllImages(ClassifierEngine* engine, ImageVector& images,
	unsigned int maxBatchSize, unsigned int& maxVideoFrames)
{
	util::log("ClassifierEngine") << "Running image batches\n";

	// shuffle the inputs
	auto randomImageOrder = getRandomOrder(images.size());

	for(unsigned int i = 0; i < images.size(); i += maxBatchSize)
	{
		unsigned int batchSize = std::min(images.size() - i,
			(size_t)maxBatchSize);
	
		ImageVector batch;
		
		for(unsigned int j = 0; j < batchSize; ++j)
		{
			batch.push_back(images[randomImageOrder[i + j]]);
		}
		
		for(auto& image : batch)
		{
			image.load();
		}

		util::log("ClassifierEngine") << " running batch with " << batch.size()
			<< " images\n";
		
		engine->runOnImageBatch(std::move(batch));
		
		if(maxVideoFrames <= batch.size())
		{
			maxVideoFrames = 0;
			break;
		}
	
		maxVideoFrames -= batch.size();
	}
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

	videos.clear();

	for(auto& pathAndVideo : pathsToVideos)
	{
		videos.push_back(pathAndVideo.second);
	}

	// Randomly shuffle videos
	std::shuffle(videos.begin(), videos.end(),
		std::default_random_engine(std::time(0)));
}


}

}


