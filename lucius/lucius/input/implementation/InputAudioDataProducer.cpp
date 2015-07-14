/*    \file   InputAudioDataProducer.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the InputAudioDataProducer class.
*/

#include <lucious/input/interface/InputAudioDataProducer.h>

#include <lucious/matrix/interface/Matrix.h>

#include <lucious/util/interface/debug.h>

namespace lucious
{

namespace input
{

InputAudioDataProducer::InputAudioDataProducer(const std::string& imageDatabaseFilename)
{

}

InputAudioDataProducer::~InputAudioDataProducer()
{

}

void InputAudioDataProducer::initialize()
{

}

InputAudioDataProducer::InputAndReferencePair InputAudioDataProducer::pop()
{
    assertM(false, "Not implemented.");

    return InputAndReferencePair();
}

bool InputAudioDataProducer::empty() const
{
    return true;
}

void InputAudioDataProducer::reset()
{

}

size_t InputAudioDataProducer::getUniqueSampleCount() const
{
    return 0;
}


}

}



