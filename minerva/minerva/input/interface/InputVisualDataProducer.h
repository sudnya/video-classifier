/*	\file   InputVisualDataProducer.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the InputVisualDataProducer class.
*/

#pragma once

// Minerva Includes
#include <minerva/input/interface/InputDataProducer.h>

#include <minerva/video/interface/ImageVector.h>
#include <minerva/video/interface/Video.h>

namespace minerva
{

namespace input
{

/*! \brief A class for accessing a stream of visual data */
class InputVisualDataProducer : public InputDataProducer
{
public:
	InputVisualDataProducer(const std::string& imageDatabaseFilename);
	virtual ~InputVisualDataProducer();

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

private:
	void _initialize();

private:
	video::VideoVector _videos;
	video::ImageVector _images;

private:
	std::string _sampleDatabasePath;
	
private:
	size_t _remainingSamples;

private:
	size_t _colorComponents;

private:
	bool _initialized;

private:
	std::default_random_engine _generator;

};

}

}

