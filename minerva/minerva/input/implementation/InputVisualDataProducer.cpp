/*	\file   InputVisualDataProducer.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the InputVisualDataProducer class.
*/

#include <minerva/input/interface/InputVisualDataProducer.h>

namespace minerva
{

namespace input
{

InputVisualDataProducer::InputVisualDataProducer(const std::string& imageDatabaseFilename)
: _remainingSamples(0), _colorComponents(3), _initialized(false)
{

}

InputVisualDataProducer::~InputVisualDataProducer()
{

}

typedef video::Image	   Image;
typedef video::ImageVector ImageVector;
typedef video::Video	   Video;
typedef video::VideoVector VideoVector;

static ImageVector getImageBatch(ImageVector& images, VideoVector& video,
	size_t& remainingSamples, size_t batchSize);

InputVisualDataProducer::InputAndReference InputVisualDataProducer::pop()
{
	_initialize();
	
	ImageVector batch = getImageBatch(_images, _videos, _remainingSamples, getBatchSize());
	
	// TODO: specialize this logic
	auto input = batch.convertToStandardizedMatrix(getInputCount(),
		getInputBlockingFactor(), getColorComponents());
	auto reference = batch.getReference();
	
	return std::make_pair(std::move(input), std::move(reference));
}

bool InputVisualDataProducer::empty() const
{
	return _remainingSamples == 0;
}

void InputVisualDataProducer::reset()
{
	_remainingSamples = uniqueSampleCount();
}

static void parseImageDatabase(ImageVector& images, VideoVector& video,
	const std::string& path, bool requiresLabeledData);

void InputVisualDataProducer::_initialize()
{
	if (!_initialized)
	{
		return;
	}
	
	bool shouldSeedWithTime = util::KnobDatabase::getKnobValue(
		"InputVisualDataProducer::SeedWithTime", false);
	
	if(shouldSeedWithTime)
	{
		_generator.seed(std::time(0));
	}
	else
	{
		_generator.seed(127);
	}
	
	parseImageDatabase(_images, _video, getSampleDatabasePath(),
		getRequiresLabelledData());
	
	_initialized = true;
}

static void sliceOutTilesToFitTheNetwork(InputVisualDataProducer* engine, ImageVector& images,
	size_t maxVideoFrames);
	
static void consolidateLabels(ImageVector& images, VideoVector& videos);

static void parseImageDatabase(ImageVector& images, VideoVector& videos,
	const std::string& path, bool requiresLabeledData)
{
	util::log("InputVisualDataProducer") << " scanning image database '"
		<< path << "'\n";

	database::SampleDatabase sampleDatabase(path);

	for(auto& sample : sampleDatabase)
	{
		if(!sample.hasLabel() && requiresLabeledData)
		{
			util::log("InputVisualDataProducer") << "  skipped unlabeled data '"
				<< sample.path() << "'\n";
			continue;
		}

		if(sample.isImageSample())
		{

			if(sample.hasLabel())
			{
				util::log("InputVisualDataProducer") << "  found labeled image '" << sample.path()
					<< "' with label '" << sample.label() << "'\n";
			}
			else
			{
				util::log("InputVisualDataProducer") << "  found unlabeled image '" << sample.path()
					<< "'\n";
			}
			
			images.push_back(Image(sample.path(), sample.label()));
		}
		else if(sample.isVideoSample())
		{
			if(sample.hasLabel())
			{
				util::log("InputVisualDataProducer") << "  found labeled video '" << sample.path()
					<< "' with label '" << sample.label() << "'\n";
			}
			else
			{
				util::log("InputVisualDataProducer") << "  found unlabeled video '" << sample.path()
					<< "'n";
			}

			videos.push_back(Video(sample.path(), sample.label(),
				sample.beginFrame(), sample.endFrame()));
		}
	}
	
	consolidateLabels(images, videos);
}

static void getVideoBatch(ImageVector& batch, VideoVector& videos,
	size_t& remainingSamples, size_t batchSize);
static void getImageBatch(ImageVector& batch, ImageVector& images,
	size_t& remainingSamples, size_t batchSize);

static ImageVector getImageBatch(ImageVector& images, VideoVector& videos,
	size_t& remainingSamples, size_t batchSize)
{
	std::uniform_int_distribution<size_t> distribution(0,batchSize);

	size_t imageBatchSize = distribution(_generator);
	size_t videoBatchSize = batchSize - imageBatchSize;

	ImageVector batch;
	
	getVideoBatch(batch, videos, remainingSamples, videoBatchSize);
	getImageBatch(batch, images, remainingSamples, imageBatchSize);

	return batch;
}

typedef std::vector<std::pair<unsigned int, unsigned int>> VideoAndFrameVector;

static VideoAndFrameVector pickRandomFrames(VideoVector& videos,
	unsigned int maxVideoFrames, bool requiresLabeledData);

static void getVideoBatch(ImageVector& batch, VideoVector& videos,
	size_t& remainingSamples, size_t batchSize)
{
	util::log("InputVisualDataProducer") << "Filling video batch\n";

	size_t frameCount = std::min(batchSize, remainingSamples);
	
	auto frames = pickRandomFrames(videos, frameCount, requiresLabeledData);

	for(auto frame : frames)
	{
		unsigned int video  = frame.first;
		unsigned int offset = frame.second;
		
		util::log("InputVisualDataProducer") << " Getting frame (" << frame
			<< ") from video " << videos[video].path() << "\n"; 
		
		batch.push_back(videos[video].getSpecificFrame(offset));
	}
	
	remainingSamples -= frames.size();
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
	util::log("InputVisualDataProducer") << " Picking random " << maxVideoFrames
		<< " frames from videos\n"; 

	if(videos.size() == 0)
	{
		return VideoAndFrameVector();
	}

	VideoAndFrameVector positions;
	
	for(unsigned int i = 0; i < maxVideoFrames; ++i)
	{
		unsigned int video = _generator() % videos.size();
		unsigned int frames = videos[video].getTotalFrames();
		
		if(frames == 0)
		{
			continue;
		}

		unsigned int frame = _generator() % frames;

		if(requiresLabeledData)
		{
			frame = mapFrameIndexToLabeledFrameIndex(frame, videos[video]);
		}

		util::log("InputVisualDataProducer") << "  Video " << videos[video].path()
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
	
	std::shuffle(order.begin(), order.end(), _generator);	

	return order;
}

static void sliceOutCenterOnly(InputVisualDataProducer* engine, ImageVector& images, size_t maxVideoFrames)
{
	size_t xTile = 0;
	size_t yTile = 0;
	
	util::getNearestToSquareFactors(xTile, yTile,
		engine->getInputFeatureCount() / engine->getColorComponents());

	ImageVector slicedImages;
	
	size_t frames = 0;
	
	for(auto& image : images)
	{
		if(frames >= maxVideoFrames)
		{
			break;
		}

		image.load();
	
		size_t startX = (image.x() - xTile) / 2;
		size_t startY = (image.y() - yTile) / 2;
	
		size_t newX = xTile;
		size_t newY = yTile;
	
		if(newX + startX > image.x())
		{
			newX = image.x() - startX;
		}
		
		if(newY + startY > image.y())
		{
			newY = image.y() - startY;
		}
		
		slicedImages.push_back(image.getTile(startX, startY,
			newX, newY, engine->getColorComponents()));
		
		frames += 1;
	}
	
	images = std::move(slicedImages);

}

static void sliceOutAllTiles(InputVisualDataProducer* engine, ImageVector& images,
	size_t maxVideoFrames)
{	
	size_t xTile = 0;
	size_t yTile = 0;
	
	util::getNearestToSquareFactors(xTile, yTile,
		engine->getInputFeatureCount() / engine->getColorComponents());

	ImageVector slicedImages;
	
	size_t frames = 0;
	
	for(auto& image : images)
	{
		if(frames >= maxVideoFrames)
		{
			break;
		}
			
		image.load();
		
		for(size_t startX = 0; startX < image.x() && frames < maxVideoFrames;
			startX += xTile)
		{
			for(size_t startY = 0; startY < image.y() && frames < maxVideoFrames;
				startY += yTile)
			{
				size_t newX = xTile;
				size_t newY = yTile;
			
				if(newX + startX > image.x())
				{
					newX = image.x() - startX;
				}
				
				if(newY + startY > image.y())
				{
					newY = image.y() - startY;
				}
				
				slicedImages.push_back(image.getTile(startX, startY,
					newX, newY, engine->getColorComponents()));

				++frames;
			}
		}
	}
	
	images = std::move(slicedImages);
}

static void sliceOutTilesToFitTheNetwork(InputVisualDataProducer* engine,
	ImageVector& images, size_t maxVideoFrames)
{
	bool shouldSlice = util::KnobDatabase::getKnobValue(
		"InputVisualDataProducer::SliceInputImagesToFitNetwork", false);

	if(!shouldSlice)
	{
		return;
	}

	bool shouldSliceCenter = util::KnobDatabase::getKnobValue(
		"InputVisualDataProducer::SliceOutCenterTileOnly", true);
	
	if(shouldSliceCenter)
	{
		sliceOutCenterOnly(engine, images, maxVideoFrames);
	}
	else
	{
		sliceOutAllTiles(engine, images, maxVideoFrames);
	}
}


static void getImageBatch(ImageVector& batch, ImageVector& images,
	size_t& remainingSamples, size_t maxBatchSize);
{
	util::log("InputVisualDataProducer") << "Collecting image batche\n";

	// shuffle the inputs
	auto randomImageOrder = getRandomOrder(images.size());

	auto batchSize = std::min(images.size(),
		(size_t)maxBatchSize);

	for(size_t i = 0; i < batchSize; ++i)
	{
		batch.push_back(images[randomImageOrder[i]]);
		batch.back().load();
	}

	maxVideoFrames -= batchSize;
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
}

}

}


