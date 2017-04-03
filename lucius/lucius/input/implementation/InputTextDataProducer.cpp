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
: _sampleDatabasePath(textDatabaseFilename), _sampleDatabaseStream(nullptr), _initialized(false),
  _reverseInputSequence(false), _useStartAndEndTokens(false),
  _shiftAmount(0), _poppedCount(0), _totalPoppedCount(0),
  _outputCount(0),
  _maximumSampleLength(0), _initialSampleLength(0), _sampleLengthStepSize(0),
  _sampleLengthStepPeriod(0)
{
}

InputTextDataProducer::InputTextDataProducer(std::istream& textDatabase)
: _sampleDatabaseStream(&textDatabase), _initialized(false), _reverseInputSequence(false),
  _useStartAndEndTokens(false),
  _shiftAmount(0), _poppedCount(0), _totalPoppedCount(0), _outputCount(0),
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

        result.resize(descriptor.getSizeInFile());

        stream.read(const_cast<char*>(result.c_str()), descriptor.getSizeInFile());
    }
    else
    {
        result = descriptor.getLabel();
    }

    size_t size = std::min(result.size(), getCurrentSampleLength());

    result.resize(size);

    return result;
}

network::Bundle InputTextDataProducer::pop()
{
    assert(!empty());

    // determine the activation sizes
    size_t miniBatchSize = getBatchSize();

    size_t timesteps = getCurrentSampleLength() - getShiftAmount();

    if(getUseStartAndEndTokens())
    {
        timesteps += 2;
    }

    Matrix inputActivations =
        matrix::zeros({getModel()->getInputCount(), miniBatchSize, timesteps},
        matrix::Precision::getDefaultPrecision());
    Matrix referenceActivations =
        matrix::zeros({getModel()->getInputCount(), miniBatchSize, timesteps},
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
    _poppedCount      += miniBatchSize;
    _totalPoppedCount += miniBatchSize;

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

void InputTextDataProducer::setReverseInputSequence(bool reverse)
{
    _reverseInputSequence = reverse;
}

bool InputTextDataProducer::getReverseInputSequence() const
{
    return _reverseInputSequence;
}

void InputTextDataProducer::setUseStartAndEndTokens(bool useStartAndEndTokens)
{
    _useStartAndEndTokens = useStartAndEndTokens;
}

bool InputTextDataProducer::getUseStartAndEndTokens() const
{
    return _useStartAndEndTokens;
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
    size_t maximumStep = 1 + getSampleLengthStepSize() *
        (_totalPoppedCount / getSampleLengthStepPeriod());
    size_t stepIndex = _totalPoppedCount / getBatchSize();

    size_t length = getInitialSampleLength() + stepIndex % maximumStep;

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
    return std::max(static_cast<size_t>(1), _sampleLengthStepPeriod);
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

    if(model->hasAttribute("ReverseInputSequence"))
    {
        setReverseInputSequence(model->getAttribute<bool>("ReverseInputSequence"));
    }
    else
    {
        setReverseInputSequence(util::KnobDatabase::getKnobValue(
            "InputTextDataProducer::ReverseInputSequence", false));
    }

    if(model->hasAttribute("UseStartAndEndTokens"))
    {
        setUseStartAndEndTokens(model->getAttribute<bool>("UseStartAndEndTokens"));
    }
    else
    {
        setUseStartAndEndTokens(util::KnobDatabase::getKnobValue(
            "InputTextDataProducer::UseStartAndEndTokens", false));
    }

    setInitialSampleLength(util::KnobDatabase::getKnobValue(
        "InputTextDataProducer::InitialSampleLength", getMaximumSampleLength()));
    setSampleLengthStepSize(util::KnobDatabase::getKnobValue(
        "InputTextDataProducer::SampleLengthStepSize", 0));
    setSampleLengthStepPeriod(util::KnobDatabase::getKnobValue(
        "InputTextDataProducer::SampleLengthStepPeriod", 1));
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

    size_t sequenceLength = data.size() - getShiftAmount();
    size_t totalSequenceLength = sequenceLength;
    size_t offset = 0;

    if(getUseStartAndEndTokens())
    {
        auto position = _outputLabels.find("START-");

        if(position == _outputLabels.end())
        {
            throw std::runtime_error("InputTextDataProducer is using start and end tokens, "
                "but model does not have a 'START-' token.");
        }

        size_t timestep = getReverseInputSequence() ? totalSequenceLength - 1 : 0;

        inputActivations[{position->second, miniBatch, timestep}] = 1.0;

        offset = 1;
        totalSequenceLength += 2;
    }

    for(size_t charPosInFile = 0; charPosInFile < sequenceLength; ++charPosInFile)
    {
        char c = data[charPosInFile];

        size_t characterPositionInGraphemeSet = _outputCount;

        if(_outputLabels.count(std::string(1, c)) != 0)
        {
            auto position = _outputLabels.find(std::string(1, c));

            characterPositionInGraphemeSet = position->second;
        }

        size_t timestep = getReverseInputSequence() ?
            totalSequenceLength - charPosInFile - offset - 1 : charPosInFile + offset;

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

    if(getUseStartAndEndTokens())
    {
        auto position = _outputLabels.find("-END");

        if(position == _outputLabels.end())
        {
            throw std::runtime_error("InputTextDataProducer is using start and end tokens, "
                "but model does not have an '-END' token.");
        }

        size_t timestep = getReverseInputSequence() ? 0 : totalSequenceLength - 1;

        inputActivations[{position->second, miniBatch, timestep}] = 1.0;
    }
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

    size_t sequenceLength = sample.size() - getShiftAmount();
    size_t totalSequenceLength = sequenceLength;
    size_t offset = 0;

    if(getUseStartAndEndTokens())
    {
        auto position = _outputLabels.find("START-");

        if(position == _outputLabels.end())
        {
            throw std::runtime_error("InputTextDataProducer is using start and end tokens, "
                "but model does not have a 'START-' token.");
        }

        size_t timestep = 0;

        referenceActivations[{position->second, miniBatch, timestep}] = 1.0;

        offset = 1;
        totalSequenceLength += 2;
    }

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

        size_t timestep = sampleIndex + offset - getShiftAmount();

        if(characterPositionInGraphemeSet == _model->getOutputCount())
        {
            if(ignoreMissingGraphemes)
            {
                referenceActivations[{0, miniBatch, timestep}] = 1.0;
                continue;
            }
            throw std::runtime_error("Could not match loaded grapheme '" +
                sample.substr(sampleIndex, 1) + "' against any known grapheme.");
        }

        referenceActivations[
            {characterPositionInGraphemeSet, miniBatch, timestep}] = 1.0;
    }

    if(getUseStartAndEndTokens())
    {
        auto position = _outputLabels.find("-END");

        if(position == _outputLabels.end())
        {
            throw std::runtime_error("InputTextDataProducer is using start and end tokens, "
                "but model does not have an '-END' token.");
        }

        size_t timestep = totalSequenceLength - 1;

        referenceActivations[{position->second, miniBatch, timestep}] = 1.0;
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

