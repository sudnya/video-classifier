/*    \file   InputDataProducer.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the InputDataProducer class.
*/

// Lucius Includes
#include <lucius/input/interface/InputDataProducerFactory.h>

#include <lucius/input/interface/InputAudioDataProducer.h>
#include <lucius/input/interface/InputTextDataProducer.h>
#include <lucius/input/interface/InputVisualDataProducer.h>

#include <lucius/database/interface/SampleDatabase.h>

namespace lucius
{

namespace input
{

InputDataProducer* InputDataProducerFactory::create(const std::string& producerName, const std::string& databaseName)
{
    if(producerName == "InputAudioDataProducer")
    {
        return new InputAudioDataProducer(databaseName);
    }
    else if(producerName == "InputTextDataProducer")
    {
        return new InputTextDataProducer(databaseName);
    }
    else if(producerName == "InputVisualDataProducer")
    {
        return new InputVisualDataProducer(databaseName);
    }

    return nullptr;
}

InputDataProducer* InputDataProducerFactory::create()
{
    return create("InputVisualDataProducer", "");
}

InputDataProducer* InputDataProducerFactory::createForDatabase(const std::string& databaseName)
{
    database::SampleDatabase database(databaseName);

    database.load();

    if(database.containsAudioSamples())
    {
        return create("InputAudioDataProducer", databaseName);
    }
    else if(database.containsVideoSamples() || database.containsImageSamples())
    {
        return create("InputVisualDataProducer", databaseName);
    }
    else if(database.containsTextSamples())
    {
        return create("InputTextDataProducer", databaseName);
    }

    return nullptr;
}

}

}


