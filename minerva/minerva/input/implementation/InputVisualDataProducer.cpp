/*    \file   InputVisualDataProducer.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the InputVisualDataProducer class.
*/


// Minerva Includes
#include <minerva/input/interface/InputVisualDataProducer.h>

#include <minerva/model/interface/Model.h>

#include <minerva/database/interface/SampleDatabase.h>
#include <minerva/database/interface/Sample.h>

#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/Operation.h>
#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/math.h>

#include <map>

namespace minerva
{

namespace input
{

InputVisualDataProducer::InputVisualDataProducer(const std::string& imageDatabaseFilename)
: _sampleDatabasePath(imageDatabaseFilename), _remainingSamples(0), _initialized(false)
{

}

InputVisualDataProducer::~InputVisualDataProducer()
{

}

typedef video::Image       Image;
typedef video::ImageVector ImageVector;
typedef video::Video       Video;
typedef video::VideoVector VideoVector;

static void parseImageDatabase(ImageVector& images, VideoVector& video,
    const std::string& path, bool requiresLabeledData);

void InputVisualDataProducer::initialize()
{
    if(_initialized)
    {
        return;
    }

    util::log("InputVisualDataProducer") << "Initializing from image database '"
        << _sampleDatabasePath << "'\n";

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

    parseImageDatabase(_images, _videos, _sampleDatabasePath, getRequiresLabeledData());

    reset();

    _initialized = true;
}

static ImageVector getBatch(ImageVector& images, VideoVector& video,
    size_t& remainingSamples, size_t batchSize, std::default_random_engine& generator,
    bool requiresLabeledData);

static void standardizeInput(matrix::Matrix& input, double mean, double standardDeviation)
{
    // subtract mean
    apply(input, input, matrix::Subtract(mean));

    // truncate to three standard deviations
    apply(input, input, matrix::Minimum( 3.0 * standardDeviation));
    apply(input, input, matrix::Maximum(-3.0 * standardDeviation));

    // divide by standard deviation
    apply(input, input, matrix::Divide(standardDeviation));

    util::log("InputVisualDataProducer") << "Standardized input " << input.toString();
}

InputVisualDataProducer::InputAndReferencePair InputVisualDataProducer::pop()
{
    assert(_initialized);

    ImageVector batch = getBatch(_images, _videos, _remainingSamples,
        getBatchSize(), _generator, getRequiresLabeledData());

    auto imageDimension = getInputSize();

    // TODO: specialize this logic
    Matrix input;

    bool shouldCropImages = util::KnobDatabase::getKnobValue(
        "InputVisualDataProducer::CropImages", false);

    if(shouldCropImages)
    {
        input = batch.getRandomCropFeatureMatrix(imageDimension[0], imageDimension[1], imageDimension[2], _generator);
    }
    else
    {
        input = batch.getDownsampledFeatureMatrix(imageDimension[0], imageDimension[1], imageDimension[2]);
    }

    auto reference = batch.getReference(getOutputLabels());

    if(getStandardizeInput())
    {
        standardizeInput(input, _model->getAttribute<double>("InputSampleMean"),
            _model->getAttribute<double>("InputSampleStandardDeviation"));
    }

    util::log("InputVisualDataProducer") << "Loaded batch of '" << batch.size()
        <<  "' image frames, " << _remainingSamples << " remaining in this epoch.\n";

    return InputAndReferencePair(std::move(input), std::move(reference));
}

bool InputVisualDataProducer::empty() const
{
    assert(_initialized);

    return _remainingSamples == 0;
}

void InputVisualDataProducer::reset()
{
    _remainingSamples = getUniqueSampleCount();
}

size_t InputVisualDataProducer::getUniqueSampleCount() const
{
    return std::min(getMaximumSamplesToRun(), _images.size() + _videos.size());
}

static void consolidateLabels(ImageVector& images, VideoVector& videos);

static void parseImageDatabase(ImageVector& images, VideoVector& videos,
    const std::string& path, bool requiresLabeledData)
{
    util::log("InputVisualDataProducer") << " scanning image database '"
        << path << "'\n";

    database::SampleDatabase sampleDatabase(path);

    sampleDatabase.load();

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
    size_t& remainingSamples, size_t batchSize, std::default_random_engine& generator, bool requiresLabeledData);
static void getImageBatch(ImageVector& batch, ImageVector& images,
    size_t& remainingSamples, size_t batchSize, std::default_random_engine& generator);

static ImageVector getBatch(ImageVector& images, VideoVector& videos,
    size_t& remainingSamples, size_t batchSize, std::default_random_engine& generator,
    bool requiresLabeledData)
{
    std::uniform_int_distribution<size_t> distribution(0, batchSize);

    size_t imageBatchSize = distribution(generator);
    size_t videoBatchSize = batchSize - imageBatchSize;

    if(images.empty())
    {
        videoBatchSize = batchSize;
    }

    if(videos.empty())
    {
        imageBatchSize = batchSize;
    }

    ImageVector batch;

    getVideoBatch(batch, videos, remainingSamples, videoBatchSize, generator, requiresLabeledData);
    getImageBatch(batch, images, remainingSamples, imageBatchSize, generator);

    return batch;
}

typedef std::vector<std::pair<unsigned int, unsigned int>> VideoAndFrameVector;

static VideoAndFrameVector pickRandomFrames(VideoVector& videos,
    unsigned int frameCount, bool requiresLabeledData,
    std::default_random_engine& generator);

static void getVideoBatch(ImageVector& batch, VideoVector& videos,
    size_t& remainingSamples, size_t batchSize,
    std::default_random_engine& generator, bool requiresLabeledData)
{
    util::log("InputVisualDataProducer") << "Filling video batch\n";

    size_t frameCount = std::min(batchSize, remainingSamples);

    auto frames = pickRandomFrames(videos, frameCount, requiresLabeledData, generator);

    for(auto frame : frames)
    {
        unsigned int video  = frame.first;
        unsigned int offset = frame.second;

        util::log("InputVisualDataProducer") << " Getting frame (" << offset
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
    unsigned int maxVideoFrames, bool requiresLabeledData,
    std::default_random_engine& generator)
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

        util::log("InputVisualDataProducer") << "  Video " << videos[video].path()
            << " has " << frames << " frames\n";

        positions.push_back(std::make_pair(video, frame));
    }

    return positions;
}

typedef std::vector<unsigned int> IntVector;

IntVector getRandomOrder(unsigned int size, std::default_random_engine& generator)
{
    IntVector order(size);

    for(unsigned int i = 0; i < size; ++i)
    {
        order[i] = i;
    }

    std::shuffle(order.begin(), order.end(), generator);

    return order;
}

static void getImageBatch(ImageVector& batch, ImageVector& images,
    size_t& remainingSamples, size_t maxBatchSize, std::default_random_engine& generator)
{
    util::log("InputVisualDataProducer") << "Collecting image batches\n";

    // shuffle the inputs
    auto randomImageOrder = getRandomOrder(images.size(), generator);

    auto batchSize = std::min(images.size(),
        std::min(remainingSamples, (size_t)maxBatchSize));

    for(size_t i = 0; i < batchSize; ++i)
    {
        batch.push_back(images[randomImageOrder[i]]);
        batch.back().load();
    }

    remainingSamples -= batchSize;
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


