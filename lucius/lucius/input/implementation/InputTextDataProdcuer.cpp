/*    \file   InputTextDataProducer.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the InputTextDataProducer class.
*/

#include <lucius/input/interface/InputTextDataProducer.h>

#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/debug.h>

namespace lucius
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


