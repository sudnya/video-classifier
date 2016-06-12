/*    \file   InputTextDataProducer.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the InputTextDataProducer class.
*/

#include <lucius/input/interface/InputTextDataProducer.h>

#include <lucius/network/interface/Bundle.h>

#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace input
{

InputTextDataProducer::InputTextDataProducer(const std::string& textDatabaseFilename) : _sampleDatabasePath(textDatabaseFilename), _initialized(false)
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
    
    _segmentSize = util::KnobDatabase::getKnobValue( "InputTextDataProducer::SegmentSize", 1000);

    createTextDatabase();

    _initialized = true;    
}

network::Bundle InputTextDataProducer::pop()
{
    assertM(false, "Not implemented.");

    return Bundle();
}

bool InputTextDataProducer::empty() const
{
    return true;
}

void InputTextDataProducer::reset()
{

}

size_t InputTextDataProducer::getUniqueSampleCount() const
{
    return 0;
}


}

void createTextDatabase()
{
    //
    util::log("InputTextDataProducer") << " scanning text database '" << _sampleDatabasePath << "'\n";

    database::SampleDatabase sampleDatabase(_sampleDatabasePath);

    sampleDatabase.load();

    for(auto& sample : sampleDatabase)
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
            size_t fileSize      = util::getFileSize(sample.path());
            int iterationsInFile = fileSize/_segmentSize;
            int leftOver         = fileSize%_segmentSize;

            for (int i = 0; i <= iterationsInFile; ++i) //<= to accomodate the last uneven segment
            {
                _descriptors.add(new FileDescriptor(sample.path, i*_segmentSize));
            }
            //leftover
            //_descriptors.add(new FileDescriptor(sample.path, iterationsInFile*_segmentSize));
        }

    }
}

}


