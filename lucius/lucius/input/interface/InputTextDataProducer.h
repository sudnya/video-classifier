/*  \file   InputTextDataProducer.h
    \date   Saturday June 11, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The header file for the InputTextDataProducer class.
*/

#pragma once

#include <lucius/input/interface/InputDataProducer.h>

// Standard Library Includes
#include <istream>
#include <map>
#include <random>

namespace lucius
{

namespace input
{

/*! \brief A class for accessing a stream of text data */
class InputTextDataProducer : public InputDataProducer
{
public:
    InputTextDataProducer(const std::string& textDatabaseFilename);
    InputTextDataProducer(std::istream& textDatabase);
    virtual ~InputTextDataProducer();

public:
    /*! \brief Initialize the state of the producer after all parameters have been set. */
    virtual void initialize();

    /*! \brief Deque a set of samples from the producer.
        Note: the caller must ensure that the producer is not empty.
    */
    virtual Bundle pop();

    /*! \brief Return true if there are no more samples. */
    virtual bool empty() const;

    /*! \brief Reset the producer to its original state, all previously
        popped samples should now be available again. */
    virtual void reset();

    /*! \brief Get the total number of unique samples that can be produced. */
    virtual size_t getUniqueSampleCount() const;

public:
    /*! \brief Set the model. */
    virtual void setModel(model::Model* model);

public:
    /*! \brief Set whether or not the input sequence should be reduced. */
    void setReverseInputSequence(bool sequence);

    /*! \brief Get whether or not the input sequence should be reduced. */
    bool getReverseInputSequence() const;

public:
    /*! \brief Set the size of the input sample */
    void setMaximumSampleLength(size_t length);

    /*! \brief Get the size of an input sample. */
    size_t getMaximumSampleLength() const;

    /*! \brief Set the amount to shift over labels. */
    void setShiftAmount(size_t amount);

    /*! \brief Get the amount to shift over labels. */
    size_t getShiftAmount() const;

public:
    /*! \brief Get current sample length. */
    size_t getCurrentSampleLength() const;

    /*! \brief Set the initial sample length. */
    void setInitialSampleLength(size_t length);

    /*! \brief Get the initial sample length. */
    size_t getInitialSampleLength() const;

    /*! \brief Set the sample length step size. */
    void setSampleLengthStepSize(size_t length);

    /*! \brief Get the sample length step size. */
    size_t getSampleLengthStepSize() const;

    /*! \brief Set the sample length step period in samples. */
    void setSampleLengthStepPeriod(size_t length);

    /*! \brief Get the sample length step period in samples. */
    size_t getSampleLengthStepPeriod() const;

private:
    void createTextDatabase();
    void getReferenceActivationsForString(const std::string& sample,
        Matrix& referenceActivations, size_t miniBatch) const;

private:
    class FileDescriptor
    {
        public:
            enum Type
            {
                FILE_DESCRIPTOR,
                STRING_DESCRIPTOR
            };

        public:
            FileDescriptor(std::string filename, size_t offset, size_t size)
            : _type(FILE_DESCRIPTOR), _filename(filename), _offsetInFile(offset), _sizeInFile(size)
            {
            }

            FileDescriptor(std::string label) : _type(STRING_DESCRIPTOR), _label(label)
            {
            }

        public:
            size_t getOffsetInFile() const
            {
                return _offsetInFile;
            }

            size_t getSizeInFile() const
            {
                return _sizeInFile;
            }

            const std::string& getFilename() const
            {
                return _filename;
            }

        public:
            Type getType() const
            {
                return _type;
            }

        public:
            const std::string& getLabel() const
            {
                return _label;
            }

        private:
            Type _type;

        private:
            std::string _label;

        private:
            std::string _filename;
            size_t _offsetInFile;
            size_t _sizeInFile;
    };

private:
    void convertChunkToOneHot(const std::string& string, Matrix m, size_t miniBatch) const;
    std::string getDataFromDescriptor(const FileDescriptor& descriptor) const;

private:
    std::vector<FileDescriptor> _descriptors;
    std::string _sampleDatabasePath;
    std::istream* _sampleDatabaseStream;
    bool _initialized;
    bool _reverseInputSequence;
    size_t _shiftAmount;
    size_t _poppedCount;
    size_t _outputCount;

    std::map<std::string, size_t> _outputLabels;

private:
    size_t _maximumSampleLength;
    size_t _initialSampleLength;
    size_t _sampleLengthStepSize;
    size_t _sampleLengthStepPeriod;

private:
    std::default_random_engine _generator;

};

}

}

