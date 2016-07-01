/*  \file   InputTextDataProducer.h
    \date   Saturday June 11, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The header file for the InputTextDataProducer class.
*/

#pragma once

#include <lucius/input/interface/InputDataProducer.h>

// Standard Library Includes
#include <istream>

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
private:
    void createTextDatabase();

private:
    class FileDescriptor
    {
        public:
            FileDescriptor(std::string filename, size_t offset) : _filename(filename), _offsetInFile(offset)
            {
            }

        public:
            size_t getOffsetInFile() const
            {
                return _offsetInFile;
            }

            const std::string& getFilename() const
            {
                return _filename;
            }
        private:
            std::string _filename;
            size_t _offsetInFile;
    };

private:
    void convertChunkToOneHot(const std::string& filename, size_t offsetInFile, Matrix m, size_t miniBatch);

private:
    std::vector<FileDescriptor> _descriptors;
    std::string _sampleDatabasePath;
    std::istream* _sampleDatabase;
    bool _initialized;
    size_t _segmentSize;
    size_t _poppedCount;

};

}

}

