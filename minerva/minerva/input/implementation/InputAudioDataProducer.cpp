/*	\file   InputAudioDataProducer.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the InputAudioDataProducer class.
*/

#include <minerva/input/interface/InputAudioDataProducer.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace input
{

InputAudioDataProducer::InputAudioDataProducer(const std::string& imageDatabaseFilename)
{
	
}

InputAudioDataProducer::~InputAudioDataProducer()
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



