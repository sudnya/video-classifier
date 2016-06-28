/*  \file   InputTextDataProducer.cpp
    \date   Saturday June 11, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The source file for the InputTextDataProducer class.
*/

#include <lucius/input/interface/InputTextDataProducer.h>

#include <lucius/network/interface/Bundle.h>

#include <lucius/database/interface/SampleDatabase.h>
#include <lucius/database/interface/Sample.h>

#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/paths.h>

namespace lucius
{

namespace input
{

InputTextDataProducer::InputTextDataProducer(const std::string& textDatabaseFilename) : _sampleDatabasePath(textDatabaseFilename), _initialized(false)
{
    
}

InputTextDataProducer::InputTextDataProducer(std::istream& textDatabase) : _sampleDatabase(textDatabase), _initialized(false)
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
    util::log("InputTextDataProducer") << "Initializing from text database '" << _sampleDatabasePath << "'\n";
    
    _segmentSize = util::KnobDatabase::getKnobValue("InputTextDataProducer::SegmentSize", 200);

    createTextDatabase();

    _initialized = true;    
}

network::Bundle InputTextDataProducer::pop()
{
    // get minibatch size
    auto miniBatchSize = this->getBatchSize();

    Matrix inputActivations(this->getModel()->getInputCount(), miniBatchSize, _segmentSize);

    // get minibatch number of descriptors (key)
    for (size_t i = 0; i < miniBatchSize; ++i)
    {
        convertChunkToOneHot(_descriptors[_poppedCount+i]);
    }

    // one hot encoded the samples within descriptors
    // add one hot encoded matrix to bundle
    // add one hot encoded reference matrix (shifted to next char) to bundle 
    _poppedCount += miniBatchSize;
    return Bundle();
}

bool InputTextDataProducer::empty() const
{
    return (_descriptor.size() - _poppedCount) == 0;
}

void InputTextDataProducer::reset()
{
    // TODO: std::shuffle(_descriptors.begin(), _descriptors.end(), );
    _poppedCount = 0;
}

size_t InputTextDataProducer::getUniqueSampleCount() const
{
    return _descriptors.size();
}



void InputTextDataProducer::createTextDatabase()
{
    //
    util::log("InputTextDataProducer") << " scanning text database '" << _sampleDatabasePath << "'\n";

    std::unique_ptr<database::SampleDatabase> sampleDatabase;
    
    if(_sampleDatabase == nullptr)
    {
        sampleDatabase.reset(new database::SampleDatabase(_sampleDatabasePath));
    }
    else
    {
        sampleDatabase.reset(new database::SampleDatabase(_sampleDatabase));
    }

    sampleDatabase->load();

    for(auto& sample : *sampleDatabase)
    {
        if(sample.isTextSample())
        {
            if(sample.hasLabel())
            {
                util::log("InputTextDataProducer::Detail") << " found labeled image '" << sample.path() << "' with label '" << sample.label() << "'\n";
            }
            else
            {
                util::log("InputTextDataProducer::Detail") << "  found unlabeled image '" << sample.path() << "'\n";
            }

            //get file size
            size_t fileSize         = util::getFileSize(sample.path());
            size_t iterationsInFile = fileSize/_segmentSize;
            size_t leftOver         = fileSize%_segmentSize;

            for(size_t i = 0; i < iterationsInFile; ++i) 
            {
                _descriptors.push_back(FileDescriptor(sample.path(), i*_segmentSize));
            }
            
            //leftover
            if(leftOver > 0)
            {
                _descriptors.push_back(FileDescriptor(sample.path(), iterationsInFile*_segmentSize));
            }
        }
    }
}

} // input namespace

} // lucius namespace

