/*! \file   AudioVector.h
    \date   Tuesday August 4, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the AudioVector class.
*/

#pragma once

// Lucius Includes
#include <lucius/audio/interface/Audio.h>

#include <lucius/util/interface/string.h>

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{

namespace audio
{

/*! \brief An abstraction for an array of audio samples. */
class AudioVector
{
private:
    typedef std::vector<Audio> BaseAudioVector;
    typedef matrix::Matrix Matrix;

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
    size_t samples() const;
	size_t size()    const;
	bool   empty()   const;

public:
	void clear();

public:
	void push_back(const Audio& audio);

public:
    Matrix getFeatureMatrixForFrameSize(size_t samplesPerFrame) const;
    Matrix getReference(const util::StringVector& labels, size_t samplesPerFrame) const;

private:
    BaseAudioVector _audio;
};


}

}






