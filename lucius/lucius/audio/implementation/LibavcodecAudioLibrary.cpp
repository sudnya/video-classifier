/*	\file   LibavcodecAudioLibrary.cpp
	\date   Thursday August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LibavcodecAudioLibrary class.
*/

// Lucius Includes
#include <lucius/audio/interface/LibavcodecAudioLibrary.h>

#include <lucius/audio/interface/LibavcodecLibrary.h>

#include <lucius/util/interface/paths.h>
#include <lucius/util/interface/string.h>
#include <lucius/util/interface/Knobs.h>

// Standard Library Includes
#include <fstream>
#include <cassert>

namespace lucius
{

namespace audio
{

LibavcodecAudioLibrary::~LibavcodecAudioLibrary()
{

}

LibavcodecAudioLibrary::HeaderAndData LibavcodecAudioLibrary::loadAudio(std::istream& stream,
    const std::string& format)
{
    LibavcodecLibrary::AVPacket packet;

    LibavcodecLibrary::av_init_packet(&packet);

    auto codecName = util::strip(format, ".");

    auto* codec = LibavcodecLibrary::avcodec_find_decoder_by_name(
        codecName.c_str());

    if(codec == nullptr)
    {
        throw std::runtime_error("Failed to open decoder for " + format);
    }

    LibavcodecLibrary::AVCodecContextRAII context(codec);

    if(context == nullptr)
    {
        throw std::runtime_error("Failed to allocate codec context for " + format);
    }

    auto status = LibavcodecLibrary::avcodec_open2(context, codec, nullptr);

    if(status < 0)
    {
        throw std::runtime_error("Failed to open codec for " + format);
    }

    HeaderAndData headerAndData;

    DataVector buffer(LibavcodecLibrary::AUDIO_INBUF_SIZE +
        LibavcodecLibrary::AV_INPUT_BUFFER_PADDING_SIZE);

    size_t fileSize = util::getFileSize(stream);

    packet.data = buffer.data();
    packet.size = std::min(fileSize, buffer.size());

    stream.read(reinterpret_cast<char*>(packet.data), packet.size);

    LibavcodecLibrary::AVFrameRAII decodedFrame;

    while(packet.size > 0)
    {
        int gotFrame = 0;

        int length = LibavcodecLibrary::avcodec_decode_audio4(context,
             decodedFrame, &gotFrame, &packet);

        if(length < 0)
        {
            throw std::runtime_error("Error while decoding " + format);
        }

        if(gotFrame)
        {
            headerAndData.header.samples += LibavcodecLibrary::getNumberOfSamples(decodedFrame);

            headerAndData.header.bytesPerSample =
                LibavcodecLibrary::getBytesPerSampleForFormat(context);
            headerAndData.header.samplingRate = LibavcodecLibrary::getSamplingRate(context);

            int dataSize = LibavcodecLibrary::av_samples_get_buffer_size(nullptr,
                LibavcodecLibrary::getChannelCount(context),
                LibavcodecLibrary::getNumberOfSamples(decodedFrame),
                LibavcodecLibrary::getSampleFormat(context), 1);
            assert(dataSize >= 0);

            size_t position = headerAndData.data.size();

            headerAndData.data.resize(position + dataSize);

            std::memcpy(reinterpret_cast<uint8_t*>(headerAndData.data.data()) + position,
                LibavcodecLibrary::getData(decodedFrame), dataSize);
        }

        packet.size -= length;
        packet.data += length;

        packet.dts = LibavcodecLibrary::AV_NOPTS_VALUE;
        packet.pts = LibavcodecLibrary::AV_NOPTS_VALUE;

        if(packet.size < LibavcodecLibrary::AUDIO_REFILL_THRESH)
        {
            std::memmove(buffer.data(), packet.data, packet.size);

            packet.data = buffer.data();

            stream.read(reinterpret_cast<char*>(packet.data + packet.size),
                LibavcodecLibrary::AUDIO_INBUF_SIZE - packet.size);

            length = stream.gcount();

            if(length > 0)
            {
                packet.size += length;
            }
        }
    }

    return headerAndData;
}

static int checkSampleFormat(LibavcodecLibrary::AVCodec *codec,
    LibavcodecLibrary::AVSampleFormat format)
{
    auto* p = codec->sample_fmts;

    while(*p != LibavcodecLibrary::AV_SAMPLE_FMT_NONE)
    {
        if(*p == format)
        {
            return 1;
        }

        p++;
    }

    return 0;
}

static int selectChannelLayout(LibavcodecLibrary::AVCodec *codec)
{
    const uint64_t* layout = nullptr;

    uint64_t bestLayout          = 0;
    int      highestChannelCount = 0;

    if(!codec->channel_layouts)
    {
        return LibavcodecLibrary::AV_CH_LAYOUT_MONO;
    }

    layout = codec->channel_layouts;

    while(*layout)
    {
        int channels = LibavcodecLibrary::av_get_channel_layout_nb_channels(*layout);

        if(channels > highestChannelCount)
        {
            bestLayout          = *layout;
            highestChannelCount = channels;
        }

        ++layout;
    }

    return bestLayout;
}

void LibavcodecAudioLibrary::saveAudio(std::ostream& stream, const std::string& format,
    const Header& header, const DataVector& data)
{
    auto codecName = util::strip(format, ".");

    LibavcodecLibrary::AVCodec* codec =
        LibavcodecLibrary::avcodec_find_encoder_by_name(codecName.c_str());

    if(codec == nullptr)
    {
        throw std::runtime_error("Could not find encoder for " + format);
    }

    LibavcodecLibrary::AVCodecContextRAII context(codec);

    if(context == nullptr)
    {
        throw std::runtime_error("Could not create context for " + format);
    }

    LibavcodecLibrary::setBitRate(context, getBitRate());
    LibavcodecLibrary::setSampleFormat(context,
        LibavcodecLibrary::getSampleFormatWithBytes(header.bytesPerSample));

    if(!checkSampleFormat(codec, LibavcodecLibrary::getSampleFormat(context)))
    {
        throw std::runtime_error("Encoder does not support this sample format.");
    }

    LibavcodecLibrary::setSampleRate(context, header.samplingRate);
    LibavcodecLibrary::setChannelLayout(context, selectChannelLayout(codec));
    LibavcodecLibrary::setChannelCount(context, 1);

    auto status = LibavcodecLibrary::avcodec_open2(context, codec, nullptr);

    if(status < 0)
    {
        throw std::runtime_error("Could not open codec.");
    }

    LibavcodecLibrary::AVFrameRAII frame;

    LibavcodecLibrary::setNumberOfSamples(frame, LibavcodecLibrary::getFrameSize(context));
    LibavcodecLibrary::setSampleFormat(frame, LibavcodecLibrary::getSampleFormat(context));
    LibavcodecLibrary::setChannelLayout(frame, LibavcodecLibrary::getChannelLayout(context));

    /* the codec gives us the frame size, in samples,
     * we calculate the size of the samples buffer in bytes */
    int bufferSize = LibavcodecLibrary::av_samples_get_buffer_size(nullptr,
        LibavcodecLibrary::getChannelCount(context), LibavcodecLibrary::getFrameSize(context),
        LibavcodecLibrary::getSampleFormat(context), 0);

    if(bufferSize < 0)
    {
        throw std::runtime_error("Could not get sample buffer size.");
    }

    DataVector buffer(bufferSize);

    /* setup the data pointers in the AVFrame */
    status = LibavcodecLibrary::avcodec_fill_audio_frame(frame,
        LibavcodecLibrary::getChannelCount(context), LibavcodecLibrary::getSampleFormat(context),
        reinterpret_cast<const uint8_t*>(buffer.data()), bufferSize, 0);

    if(status < 0)
    {
        throw std::runtime_error("Could not setup audio frame");
    }

    for(size_t sample = 0; sample < header.samples;
        sample += LibavcodecLibrary::getNumberOfSamples(frame))
    {
        LibavcodecLibrary::AVPacketRAII packet;

        packet->data = nullptr;
        packet->size = 0;

        if(header.samples - sample < LibavcodecLibrary::getNumberOfSamples(frame))
        {
            size_t remainingBytes = (header.samples - sample) * header.bytesPerSample;

            std::memcpy(buffer.data(), reinterpret_cast<const uint8_t*>(data.data()) +
                (sample * header.bytesPerSample),
                remainingBytes);

            std::memset(&buffer[remainingBytes], 0, bufferSize - remainingBytes);
        }
        else
        {
            std::memcpy(buffer.data(), reinterpret_cast<const uint8_t*>(data.data()) +
                (sample * header.bytesPerSample), bufferSize);
        }

        int gotOutput = 0;

        auto status = LibavcodecLibrary::avcodec_encode_audio2(context, packet,
            frame, &gotOutput);

        if(status < 0)
        {
            throw std::runtime_error("Error encoding audio frame.");
        }

        if(gotOutput != 0)
        {
            stream.write(reinterpret_cast<char*>(packet->data), packet->size);
        }
    }

    for(int gotOutput = true; gotOutput != 0; )
    {
        LibavcodecLibrary::AVPacketRAII packet;

        packet->data = nullptr;
        packet->size = 0;

        auto status = LibavcodecLibrary::avcodec_encode_audio2(context, packet,
            nullptr, &gotOutput);

        if(status < 0)
        {
            throw std::runtime_error("Error encoding audio frame.");
        }

        if(gotOutput != 0)
        {
            stream.write(reinterpret_cast<char*>(packet->data), packet->size);
        }
    }
}

LibavcodecAudioLibrary::StringVector LibavcodecAudioLibrary::getSupportedExtensions() const
{
    return StringVector(util::split(".mp4|.mp2|.mp3|.wav|.flac", "|"));
}

size_t LibavcodecAudioLibrary::getBitRate() const
{
    return util::KnobDatabase::getKnobValue("LibavcodecAudioLibrary::Bitrate", 64000);
}

}

}






