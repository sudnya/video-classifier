/*  \file   InputTextDataProducer.cpp
    \date   Saturday June 11, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The source file for the InputTextDataProducer class.
*/

// Lucius Includes
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

// Standard Library Includes
#include <fstream>
#include <algorithm>

namespace lucius
{

namespace input
{

InputTextDataProducer::InputTextDataProducer(const std::string& textDatabaseFilename)
: _sampleDatabasePath(textDatabaseFilename),_sampleDatabaseStream(nullptr), _initialized(false),
  _shiftAmount(0), _poppedCount(0), _outputCount(0),
  _maximumSampleLength(0), _initialSampleLength(0), _sampleLengthStepSize(0),
  _sampleLengthStepPeriod(0)
{
}

InputTextDataProducer::InputTextDataProducer(std::istream& textDatabase)
: _sampleDatabaseStream(&textDatabase), _initialized(false), _shiftAmount(0),
  _poppedCount(0), _outputCount(0),
  _maximumSampleLength(0), _initialSampleLength(0), _sampleLengthStepSize(0),
  _sampleLengthStepPeriod(0)
{
}

InputTextDataProducer::~InputTextDataProducer()
{

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

    bool shouldSeedWithTime = util::KnobDatabase::getKnobValue(
        "InputTextDataProducer::SeedWithTime", false);

    if(shouldSeedWithTime)
    {
        _generator.seed(std::time(0));
    }
    else
    {
        _generator.seed(127);
    }

    _initialized = true;
}

std::string InputTextDataProducer::getDataFromDescriptor(const FileDescriptor& descriptor) const
{
    std::string result;

    if(descriptor.getType() == FileDescriptor::FILE_DESCRIPTOR)
    {
        std::ifstream stream(descriptor.getFilename());

        stream.seekg(descriptor.getOffsetInFile());

        std::vector<int8_t> result(descriptor.getSizeInFile());

        stream.read(reinterpret_cast<char*>(result.data()), descriptor.getSizeInFile());
    }
    else
    {
        result = descriptor.getLabel();
    }

    size_t size = std::min(result.size(), getCurrentSampleLength());

    result.resize(size);

    return result;
}

void InputTextDataProducer::getReferenceActivationsForString(const std::string& sample,
    Matrix& referenceActivations, size_t miniBatch) const
{
    if(sample.empty())
    {
        return;
    }

    bool ignoreMissingGraphemes = util::KnobDatabase::getKnobValue(
        "InputTextDataProducer::IgnoreMissingGraphemes", false);

    for(size_t sampleIndex = getShiftAmount(); sampleIndex < sample.size(); ++sampleIndex)
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
                referenceActivations[{0, miniBatch, sampleIndex - getShiftAmount()}] = 1.0;
                continue;
            }
            throw std::runtime_error("Could not match loaded grapheme '" +
                sample.substr(sampleIndex, 1) + "' against any known grapheme.");
        }

        referenceActivations[
            {characterPositionInGraphemeSet, miniBatch, sampleIndex - getShiftAmount()}] = 1.0;
    }
}

network::Bundle InputTextDataProducer::pop()
{
    assert(!empty());

    // get minibatch size
    auto miniBatchSize = getBatchSize();

    Matrix inputActivations =
        matrix::zeros({getModel()->getInputCount(), miniBatchSize,
                       getCurrentSampleLength() - getShiftAmount()},
        matrix::Precision::getDefaultPrecision());
    Matrix referenceActivations =
        matrix::zeros({getModel()->getInputCount(), miniBatchSize,
                       getCurrentSampleLength() - getShiftAmount()},
        matrix::Precision::getDefaultPrecision());

    // get minibatch number of descriptors (key)
    for (size_t i = 0; i < miniBatchSize; ++i)
    {
        auto data = getDataFromDescriptor(_descriptors[_poppedCount + i]);

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
    std::shuffle(_descriptors.begin(), _descriptors.begin() + getUniqueSampleCount(), _generator);
    _poppedCount = 0;
}

size_t InputTextDataProducer::getUniqueSampleCount() const
{
    // limit to total sample count
    size_t samples = std::min(getMaximumSamplesToRun(), _descriptors.size());

    // clamp to a multiple of the batch size
    return samples - (samples % getBatchSize());
}

void InputTextDataProducer::setMaximumSampleLength(size_t length)
{
    _maximumSampleLength = length;
}

size_t InputTextDataProducer::getMaximumSampleLength() const
{
    return _maximumSampleLength;
}

size_t InputTextDataProducer::getShiftAmount() const
{
    return _shiftAmount;
}

void InputTextDataProducer::setShiftAmount(size_t amount)
{
    _shiftAmount = amount;
}

size_t InputTextDataProducer::getCurrentSampleLength() const
{
    size_t length = getInitialSampleLength() +
        getSampleLengthStepSize() * (_poppedCount / getSampleLengthStepPeriod());

    return std::min(length, getMaximumSampleLength());
}

void InputTextDataProducer::setInitialSampleLength(size_t length)
{
    _initialSampleLength = length;
}

size_t InputTextDataProducer::getInitialSampleLength() const
{
    return _initialSampleLength;
}

void InputTextDataProducer::setSampleLengthStepSize(size_t step)
{
    _sampleLengthStepSize = step;
}

size_t InputTextDataProducer::getSampleLengthStepSize() const
{
    return _sampleLengthStepSize;
}

void InputTextDataProducer::setSampleLengthStepPeriod(size_t period)
{
    _sampleLengthStepPeriod = period;
}

size_t InputTextDataProducer::getSampleLengthStepPeriod() const
{
    return _sampleLengthStepPeriod;
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

    setMaximumSampleLength(model->getAttribute<size_t>("MaximumSampleLength"));
    setShiftAmount(model->getAttribute<size_t>("ShiftAmount"));
    setInitialSampleLength(model->getAttribute<size_t>("InitialSampleLength"));
    setSampleLengthStepSize(model->getAttribute<size_t>("SampleLengthStepSize"));
    setSampleLengthStepPeriod(model->getAttribute<size_t>("SampleLengthStepPeriod"));
}

void InputTextDataProducer::convertChunkToOneHot(const std::string& data,
    Matrix inputActivations, size_t miniBatch) const
{
    if(data.empty())
    {
        return;
    }

    bool ignoreMissingGraphemes = util::KnobDatabase::getKnobValue(
        "InputTextDataProducer::IgnoreMissingGraphemes", false);

    bool reverseSequence = util::KnobDatabase::getKnobValue(
        "InputTextDataProducer::ReverseInputSequence", false);

    size_t sequenceLength = data.size() - getShiftAmount();

    for(size_t charPosInFile = 0; charPosInFile < sequenceLength; ++charPosInFile)
    {
        char c = data[charPosInFile];

        size_t characterPositionInGraphemeSet = _outputCount;

        if(_outputLabels.count(std::string(1, c)) != 0)
        {
            auto position = _outputLabels.find(std::string(1, c));

            characterPositionInGraphemeSet = position->second;
        }

        size_t timestep = reverseSequence ? sequenceLength - charPosInFile - 1 : charPosInFile;

        if(characterPositionInGraphemeSet == _outputCount)
        {
            if(ignoreMissingGraphemes)
            {
                inputActivations[{0, miniBatch, timestep}] = 1.0;
                continue;
            }
            throw std::runtime_error("Could not match loaded grapheme '" + std::string(1, c) +
                "' against any known grapheme.");
        }

        inputActivations[{characterPositionInGraphemeSet, miniBatch, timestep}] = 1.0;
    }
}

void InputTextDataProducer::createTextDatabase()
{
    util::log("InputTextDataProducer") << " scanning text database '"
        << _sampleDatabasePath << "'\n";

    std::unique_ptr<database::SampleDatabase> sampleDatabase;

    if(_sampleDatabaseStream == nullptr)
    {
        sampleDatabase = std::make_unique<database::SampleDatabase>(_sampleDatabasePath);
    }
    else
    {
        sampleDatabase = std::make_unique<database::SampleDatabase>(*_sampleDatabaseStream);
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

                size_t iterationsInLabel = totalSize / getMaximumSampleLength();
                size_t leftOver = totalSize % getMaximumSampleLength();

                for(size_t i = 0; i < iterationsInLabel; ++i)
                {
                    _descriptors.push_back(
                        FileDescriptor(sample.label().substr(i * getMaximumSampleLength(),
                            getMaximumSampleLength())));
                }

                if(leftOver > 0)
                {
                    _descriptors.push_back(FileDescriptor(
                        sample.label().substr(iterationsInLabel * getMaximumSampleLength())));
                }
            }
            else
            {
                util::log("InputTextDataProducer::Detail") << "  found unlabeled text file '"
                    << sample.path() << "'\n";

                //get file size
                size_t fileSize         = util::getFileSize(sample.path());
                size_t iterationsInFile = fileSize / getMaximumSampleLength();
                size_t leftOver         = fileSize % getMaximumSampleLength();

                for(size_t i = 0; i < iterationsInFile; ++i)
                {
                    _descriptors.push_back(FileDescriptor(sample.path(),
                        i * getMaximumSampleLength(), getMaximumSampleLength()));
                }

                //leftover
                if(leftOver > 0)
                {
                    _descriptors.push_back(FileDescriptor(sample.path(),
                        iterationsInFile * getMaximumSampleLength(), leftOver));
                }
            }
        }
    }
}

} // input namespace

} // lucius namespace

