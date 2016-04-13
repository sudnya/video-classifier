/*! \file   AudioVector.cpp
    \date   Tuesday August 4, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the AudioVector class.
*/

// Lucius Includes
#include <lucius/audio/interface/AudioVector.h>

#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <limits>

namespace lucius
{

namespace audio
{

AudioVector::AudioVector()
{

}

AudioVector::~AudioVector()
{

}

AudioVector::iterator AudioVector::begin()
{
    return _audio.begin();
}

AudioVector::const_iterator AudioVector::begin() const
{
    return _audio.begin();
}

AudioVector::iterator AudioVector::end()
{
    return _audio.end();
}

AudioVector::const_iterator AudioVector::end() const
{
    return _audio.end();
}

Audio& AudioVector::operator[](size_t index)
{
    return _audio[index];
}

const Audio& AudioVector::operator[](size_t index) const
{
    return _audio[index];
}

Audio& AudioVector::back()
{
    return _audio.back();
}

const Audio& AudioVector::back() const
{
    return _audio.back();
}

size_t AudioVector::samples() const
{
    size_t samples = 0;

    for(auto& sample : *this)
    {
        samples = std::max(samples, sample.size());
    }

    return samples;
}

size_t AudioVector::size() const
{
    return _audio.size();
}

bool AudioVector::empty() const
{
    return _audio.empty();
}

void AudioVector::clear()
{
    _audio.clear();
}

void AudioVector::push_back(const Audio& audio)
{
    _audio.push_back(audio);
}

AudioVector::Matrix AudioVector::getFeatureMatrixForFrameSize(size_t frameSize) const
{
    size_t timesteps = (samples() + frameSize - 1) / frameSize;

    Matrix features({frameSize, size(), timesteps});

    for(size_t timestep = 0; timestep != timesteps; ++timestep)
    {
        for(size_t audioId = 0; audioId != size(); ++audioId)
        {
            auto& audio = (*this)[audioId];

            size_t sample  = 0;
            size_t samples = std::min(audio.size() - timestep * frameSize, frameSize);

            for(; sample != samples; ++sample)
            {
                features(sample, audioId, timestep) =
                    audio.getSample(sample + timestep * frameSize);
            }

            for(; sample < frameSize; ++sample)
            {
                features(sample, audioId, timestep) = 0;
            }
        }
    }

    return features;
}

AudioVector::Matrix AudioVector::getReference(const util::StringVector& labels,
    size_t frameSize) const
{
    size_t timesteps = (samples() + frameSize - 1) / frameSize;

    Matrix reference(matrix::Dimension({labels.size(), size(), timesteps}));

    util::log("AudioVector") << "Generating reference audio:\n";

    for(size_t timestep = 0; timestep != timesteps; ++timestep)
    {
        util::log("AudioVector") << " For timestep " << timestep << "\n";
        size_t sample = timestep * frameSize;

        for(size_t audioId = 0; audioId != size(); ++audioId)
        {
            util::log("AudioVector") << "  For audio sample " << audioId << " with label '"
                << (*this)[audioId].getLabelForSample(sample) << "'\n";

            for(size_t outputNeuron = 0; outputNeuron != labels.size(); ++outputNeuron)
            {
                util::log("AudioVector") << "   For output neuron" << outputNeuron
                    << " with label '" << labels[outputNeuron] << "'\n";

                if((*this)[audioId].getLabelForSample(sample) == labels[outputNeuron])
                {
                    reference(outputNeuron, audioId, timestep) = 1.0;
                }
                else
                {
                    reference(outputNeuron, audioId, timestep) = 0.0;
                }
            }
        }
    }

    util::log("AudioVector") << " Generated matrix: " << reference.toString() << "\n";

    return reference;

}

}

}







