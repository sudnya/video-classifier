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
    size_t getSegmentSize();

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
    
    /*! \brief Set the size of the input sample */
    virtual void setSampleLength(size_t length);

public:
    virtual void setModel(model::Model* model);

private:
    void createTextDatabase();
    void getReferenceActivationsForString(const std::string& sample, Matrix& referenceActivations, size_t miniBatch);

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
            FileDescriptor(std::string filename, size_t offset, size_t size) : _type(FILE_DESCRIPTOR), _filename(filename), _offsetInFile(offset), _sizeInFile(size)
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
    void convertChunkToOneHot(const std::string& string, Matrix m, size_t miniBatch);
    std::string getDataFromDescriptor(const FileDescriptor& descriptor);

private:
    std::vector<FileDescriptor> _descriptors;
    std::string _sampleDatabasePath;
    std::istream* _sampleDatabaseStream;
    bool _initialized;
    size_t _segmentSize;
    size_t _poppedCount;
    size_t _outputCount;

    std::map<std::string, size_t> _outputLabels;

};

}

}

