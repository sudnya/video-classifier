/*	\file   InputTextDataProducer.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the InputTextDataProducer class.
*/

#include <minerva/input/interface/InputTextDataProducer.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace input
{

InputTextDataProducer::InputTextDataProducer(const std::string& imageDatabaseFilename)
{
	
}

InputTextDataProducer::~InputTextDataProducer()
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


