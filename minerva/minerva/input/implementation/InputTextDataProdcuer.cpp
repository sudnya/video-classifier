/*    \file   InputTextDataProducer.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the InputTextDataProducer class.
*/

#include <lucious/input/interface/InputTextDataProducer.h>

#include <lucious/matrix/interface/Matrix.h>

#include <lucious/util/interface/debug.h>

namespace lucious
{

namespace input
{

InputTextDataProducer::InputTextDataProducer(const std::string& imageDatabaseFilename)
{

}

InputTextDataProducer::~InputTextDataProducer()
{

}

void InputTextDataProducer::initialize()
{

}

InputTextDataProducer::InputAndReferencePair InputTextDataProducer::pop()
{
    assertM(false, "Not implemented.");

    return InputAndReferencePair();
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

}


