/*  \file   InputTextDataProducer.cpp
    \date   Saturday June 11, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The source file for the InputTextDataProducer class.
*/

#include <lucius/input/interface/InputTextDataProducer.h>

#include <lucius/model/interface/Model.h>
#include <lucius/network/interface/Bundle.h>

#include <lucius/database/interface/SampleDatabase.h>
#include <lucius/database/interface/Sample.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/paths.h>
#include <lucius/util/interface/Knobs.h>

#include <fstream>


namespace lucius
{

namespace input
{

InputTextDataProducer::InputTextDataProducer(const std::string& textDatabaseFilename)
: _sampleDatabasePath(textDatabaseFilename),_sampleDatabaseStream(nullptr), _initialized(false),
  _segmentSize(0), _poppedCount(0), _outputCount(0)
{
}

InputTextDataProducer::InputTextDataProducer(std::istream& textDatabase)
: _sampleDatabaseStream(&textDatabase), _initialized(false), _segmentSize(0),
  _poppedCount(0), _outputCount(0)
{
}

InputTextDataProducer::~InputTextDataProducer()
{

}

size_t InputTextDataProducer::getSegmentSize()
{
    return _segmentSize;
}

void InputTextDataProducer::initialize()
{
    if(_initialized)
    {
        return;
    }

    util::log("InputTextDataProducer") << "Initializing from text database '"
        << _sampleDatabasePath << "'\n";

    createTextDatabase();

    _initialized = true;
}

std::string InputTextDataProducer::getDataFromDescriptor(const FileDescriptor& descriptor)
{
    if(descriptor.getType() == FileDescriptor::FILE_DESCRIPTOR)
    {
        std::ifstream stream(descriptor.getFilename());

        std::vector<int8_t> result(descriptor.getSizeInFile());

        stream.read(reinterpret_cast<char*>(result.data()), descriptor.getSizeInFile());

        return std::string(result.begin(), result.end());
    }
    else
    {
        return descriptor.getLabel();
    }
}

void InputTextDataProducer::getReferenceActivationsForString(const std::string& sample,
    Matrix& referenceActivations, size_t miniBatch)
{
    if(sample.empty())
    {
        return;
    }

    bool ignoreMissingGraphemes = util::KnobDatabase::getKnobValue(
        "InputTextDataProducer::IgnoreMissingGraphemes", false);

    for(size_t sampleIndex = 1; sampleIndex < sample.size(); ++sampleIndex)
    {
        size_t characterPositionInGraphemeSet = getModel()->getOutputCount();

        for(size_t index = 0; index < _model->getOutputCount(); ++index)
        {
            if(sample.substr(sampleIndex, 1) == getModel()->getOutputLabel(index))
            {
                characterPositionInGraphemeSet = index;
                break;
            }
        }

        if(characterPositionInGraphemeSet == _model->getOutputCount())
        {
            if(ignoreMissingGraphemes)
            {
                referenceActivations[{0, miniBatch, sampleIndex - 1}] = 1.0;
                continue;
            }
            throw std::runtime_error("Could not match loaded grapheme '" +
                sample.substr(sampleIndex, 1) + "' against any known grapheme.");
        }

        referenceActivations[{characterPositionInGraphemeSet, miniBatch, sampleIndex - 1}] = 1.0;
    }
}

network::Bundle InputTextDataProducer::pop()
{
    assert(!empty());

    // get minibatch size
    auto miniBatchSize = getBatchSize();

    Matrix inputActivations = matrix::zeros({getModel()->getInputCount(), miniBatchSize, _segmentSize-1},
           matrix::Precision::getDefaultPrecision());
    Matrix referenceActivations = matrix::zeros({getModel()->getInputCount(), miniBatchSize, _segmentSize-1},
           matrix::Precision::getDefaultPrecision());
    // get minibatch number of descriptors (key)
    for (size_t i = 0; i < miniBatchSize; ++i)
    {
        auto data = getDataFromDescriptor(_descriptors[_poppedCount+i]);

        // one hot encoded the samples within descriptors
        convertChunkToOneHot(data, inputActivations, i);
        getReferenceActivationsForString(data, referenceActivations, i);
    }
    // add one hot encoded matrix to bundle
    // add one hot encoded reference matrix (shifted to next char) to bundle
    _poppedCount += miniBatchSize;

    util::log("InputTextDataProducer") << "Loaded batch of '" << miniBatchSize
        <<  "' samples samples (" << inputActivations.size()[2] << " timesteps), "
        << (getUniqueSampleCount() - _poppedCount) << " remaining in this epoch.\n";

    return Bundle(
        std::make_pair("inputActivations", matrix::MatrixVector({inputActivations})),
        std::make_pair("referenceActivations", matrix::MatrixVector({referenceActivations})));
}

bool InputTextDataProducer::empty() const
{
    return _poppedCount >= getUniqueSampleCount();
}

void InputTextDataProducer::reset()
{
    // TODO: std::shuffle(_descriptors.begin(), _descriptors.end(), );
    _poppedCount = 0;
}

size_t InputTextDataProducer::getUniqueSampleCount() const
{
    size_t samples = std::min(getMaximumSamplesToRun(), _descriptors.size());

    return samples - (samples % getBatchSize());
}

void InputTextDataProducer::setSampleLength(size_t length)
{
    _segmentSize = length;
}

void InputTextDataProducer::setModel(model::Model* model)
{
    _model = model;

    _outputCount = getModel()->getOutputCount();

    _outputLabels.clear();

    for(size_t i = 0; i < _outputCount; ++i)
    {
        _outputLabels[getModel()->getOutputLabel(i)] = i;
    }

    _segmentSize = model->getAttribute<size_t>("SegmentSize");
}

void InputTextDataProducer::convertChunkToOneHot(const std::string& data,
    Matrix inputActivations, size_t miniBatch)
{
    if(data.empty())
    {
        return;
    }

    bool ignoreMissingGraphemes = util::KnobDatabase::getKnobValue(
        "InputTextDataProducer::IgnoreMissingGraphemes", false);

    for(size_t charPosInFile = 0; charPosInFile < data.size() - 1; ++charPosInFile)
    {
        char c = data[charPosInFile];

        size_t characterPositionInGraphemeSet = _outputCount;

        if(_outputLabels.count(std::string(1, c)) != 0)
        {
            characterPositionInGraphemeSet = _outputLabels[std::string(1, c)];
        }

        if(characterPositionInGraphemeSet == _outputCount)
        {
            if(ignoreMissingGraphemes)
            {
                inputActivations[{0, miniBatch, charPosInFile}] = 1.0;
                continue;
            }
            throw std::runtime_error("Could not match loaded grapheme '" + std::string(1, c) +
                "' against any known grapheme.");
        }

        inputActivations[{characterPositionInGraphemeSet, miniBatch, charPosInFile}] = 1.0;
    }
}

void InputTextDataProducer::createTextDatabase()
{
    //
    util::log("InputTextDataProducer") << " scanning text database '" << _sampleDatabasePath << "'\n";

    std::unique_ptr<database::SampleDatabase> sampleDatabase;

    if(_sampleDatabaseStream == nullptr)
    {
        sampleDatabase.reset(new database::SampleDatabase(_sampleDatabasePath));
    }
    else
    {
        sampleDatabase.reset(new database::SampleDatabase(*_sampleDatabaseStream));
    }

    sampleDatabase->load();

    for(auto& sample : *sampleDatabase)
    {
        if(sample.isTextSample())
        {
            if(sample.hasLabel())
            {
                util::log("InputTextDataProducer::Detail") << " found labeled text '"
                    << sample.path() << "' with label '" << sample.label() << "'\n";


                size_t totalSize = sample.label().size();

                size_t iterationsInLabel = totalSize / _segmentSize;
                size_t leftOver = totalSize % _segmentSize;

                for(size_t i = 0; i < iterationsInLabel; ++i)
                {
                    _descriptors.push_back(FileDescriptor(sample.label().substr(i*_segmentSize, _segmentSize)));
                }

                if(leftOver > 0)
                {
                    _descriptors.push_back(FileDescriptor(sample.label().substr(iterationsInLabel*_segmentSize)));
                }
            }
            else
            {
                util::log("InputTextDataProducer::Detail") << "  found unlabeled text file '" << sample.path() << "'\n";

                //get file size
                size_t fileSize         = util::getFileSize(sample.path());
                size_t iterationsInFile = fileSize/_segmentSize;
                size_t leftOver         = fileSize%_segmentSize;

                for(size_t i = 0; i < iterationsInFile; ++i)
                {
                    _descriptors.push_back(FileDescriptor(sample.path(), i*_segmentSize, _segmentSize));
                }

                //leftover
                if(leftOver > 0)
                {
                    _descriptors.push_back(FileDescriptor(sample.path(), iterationsInFile*_segmentSize, leftOver));
                }
            }
        }
    }
}

} // input namespace

} // lucius namespace

