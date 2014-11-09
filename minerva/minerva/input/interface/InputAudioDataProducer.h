/*	\file   InputAudioDataProducer.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the InputAudoDataProducer class.
*/

#pragma once

#include <minerva/input/interface/InputDataProducer.h>

namespace minerva
{

namespace input
{

/*! \brief A class for accessing a stream of visual data */
class InputAudioDataProducer : public InputDataProducer
{
public:
	InputAudioDataProducer(const std::string& audioDatabaseFilename);
	virtual ~InputAudioDataProducer();

public:
	/*! \brief Deque a set of samples from the producer.

		Note: the caller must ensure that the producer is not empty.

	*/
	virtual InputAndReferencePair pop();
	
	/*! \brief Return true if there are no more samples. */
	virtual bool empty() const;

	/*! \brief Reset the producer to its original state, all previously
		popped samples should now be available again. */
	virtual void reset();
	
	/*! \brief Get the total number of unique samples that can be produced. */
	virtual size_t getUniqueSampleCount() const;

};

}

}

