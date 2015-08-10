/*! \file   AudioVector.h
    \date   Tuesday August 4, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the AudioVector class.
*/

#pragma once

// Lucius Includes
#include <lucius/audio/interface/Audio.h>

// Standard Library Includes
#include <vector>

namespace lucius
{

namespace audio
{

/*! \brief An abstraction for an array of audio samples. */
class AudioVector
{
private:
    typedef std::vector<Audio> BaseAudioVector;

public:
    typedef BaseAudioVector::iterator       iterator;
    typedef BaseAudioVector::const_iterator const_iterator;

public:
    AudioVector();
    ~AudioVector();

public:
	iterator       begin();
	const_iterator begin() const;

	iterator       end();
	const_iterator end() const;

public:
	      Audio& operator[](size_t index);
	const Audio& operator[](size_t index) const;

public:
	      Audio& back();
	const Audio& back() const;

public:
	size_t size()  const;
	bool   empty() const;

public:
	void clear();

public:
	void push_back(const Audio& audio);

public:
    Matrix getFeatureMatrixWithFrameSize(size_t samplesPerFrame) const;
    Matrix getReference(const util::StringVector& labels) const;

private:
    BaseAudioVector _audio;
};


}

}






