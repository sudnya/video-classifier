/*	\file   LibavcodecAudioLibrary.cpp
	\date   Thursday August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LibavcodecAudioLibrary class.
*/

// Lucius Includes
#include <lucius/audio/interface/LibavcodecAudioLibrary.h>

#include <lucius/audio/interface/LibavcodecLibrary.h>

namespace lucius
{

namespace audio
{

LibavcodecAudioLibrary::~LibavcodecAudioLibrary()
{

}

LibavcodecAudioLibrary::HeaderAndData LibavcodecAudioLibrary::loadAudio(const std::string& path)
{
    LibavcodecLibrary::AVPacket packet;

    LibavcodecLibrary::av_init_packet(&packet);

    AVCodec* codec = LibavcodecLibrary::avcodec_find_decoder(getCodecForPath(path));

    if(codec == nullptr)
    {
        throw std::runtime_error("Failed to open decoder for " + path);
    }

    LibavcodecLibrary::AVCodecContextRAII context(codec);

    if(context == nullptr)
    {
        throw std::runtime_error("Failed to allocate codec context for " + path);
    }

    auto status = LibavcodecLibrary::avcodec_open2(context, codec, nullptr);

    if(status < 0)
    {
        throw std::runtime_error("Failed to open codec for " + path);
    }

    HeaderAndData headerAndData;

    std::ifstream file(path);

    if(!file.is_open())
    {
        throw std::runtime_error("Failed to open input file " + path + " for reading.");
    }

    DataVector buffer(LibavcodecLibrary::AUDIO_INBUF_SIZE +
        LibavcodecLibrary::FF_INPUT_BUFFER_PADDING_SIZE);

    size_t fileSize = getFileSize(file);

    packet.data = buffer.data();
    packet.size = std::min(fileSize, buffer.size());

    file.read(packet.data, packet.size());

    LibavcodecLibrary::AVFrameRAII decodedFrame;

    while(packet.size > 0)
    {
        int gotFrame = 0;

        int length = LibavcodecLibrary::avcodec_decode_audio4(context,
             decodedFrame, &gotFrame, &packet);

        if(length < 0)
        {
            throw std::runtime_error("Error while decoding " + path);
        }

        if(gotFrame)
        {
            headerAndData.header.samples += LibavcodecLibrary::getSamples(decodedFrame);

            headerAndData.header.bytesPerSample =
                LibavcodecLibrary::getBytesPerSampleForFormat(context);
            headerAndData.header.samplingRate = LibavcodecLibrary::getSamplingRate(context);

            int dataSize = LibavcodecLibrary::av_samples_get_buffer_size(nullptr,
                LibavcodecLibrary::getChannelCount(context), decodedFrame->nb_samples,
                LibavcodecLibrary::getSampleFormat(context), 1);
            assert(dataSize >= 0);

            size_t position = headerAndData.data.size();

            headerAndData.data.resize(position + dataSize);

            std::memcpy(headerAndData.data.data(), decodedFrame->data[0], dataSize);
        }

        packet.size -= length;
        packet.data += length;

        packet.dts = LibavcodecLibrary::AV_NOPTS_VALUE;
        packet.pts = LibavcodecLibrary::AV_NOPTS_VALUE;

        if(packet.size < LibavcodecLibrary::AUDIO_REFILL_THRESH)
        {
            std::memmove(buffer.data(), packet.data, packet.size);

            packet.data = buffer.data();

            file.read(packet.data + packet.size,
                LibavcodecLibrary::AUDIO_INBUF_SIZE - packet.size);

            length = file.gcount();

            if(length > 0)
            {
                packet.size += length;
            }
        }
    }

    return headerAndData;
}

void LibavcodecAudioLibrary::saveAudio(const std::string& path, const Header& header,
    const DataVector& data)
{
    LibavcodecLibrary::AVCodec* codec =
        LibavcodecLibrary::avcodec_find_encoder(getCodecForPath(path));

    if(codec == nullptr)
    {
        throw std::runtime_error("Could not find encoder for " + path);
    }

    LibavcodecLibrary::AVCodecContextRAII context(codec);

    if(context == nullptr)
    {
        throw std::runtime_error("Could not create context for " + path);
    }

    LibavcodecLibrary::setBitRate(context, getBitrate());
    LibavcodecLibrary::setSampleFormat(context, header.bytesPerSample);

    if(!LibavcodecLibrary::check_sample_fmt(codec, getSampleFormat(context)))
    {
        throw std::runtime_error("Encoder does not support this sample format.");
    }

    LibavcodecLibrary::setSampleRate(codec, header.samplingRate);
    LibavcodecLibrary::setChannelLayout(codec, LibavcodecLibrary::AV_CH_LAYOUT_MONO);
    LibavcodecLibrary::setChannelCount(codec, 1);

    auto status = LibavcodecLibrary::avcodec_open2(context, codec, nullptr);

    if(status < 0)
    {
        throw std::runtime_error("Could not open codec.");
    }

    std::ofstream file(path);

    if(!file.is_open())
    {
        throw std::runtime_error("Could not open " + path + " for writing.");
    }

    LibavcodecLibrary::AVFrameRAII frame;

    frame->nb_samples     = LibavcodecLibrary::getFrameSize(context);
    frame->format         = LibavcodecLibrary::getSampleFomat(context);
    frame->channel_layout = LibavcodecLibrary::getChannelLayout(context);

    /* the codec gives us the frame size, in samples,
     * we calculate the size of the samples buffer in bytes */
    size_t bufferSize = LibavcodecLibrary::av_samples_get_buffer_size(nullptr,
        LibavcodecLibrary::getChannelCount(context), LibavcodecLibrary::getFrameSize(context),
        LibavcodecLibrary::getSampleFomat(context), 0);

    if(bufferSize < 0)
    {
        throw std::runtime_error("Could not get sample buffer size.");
    }

    DataVector buffer(bufferSize);

    /* setup the data pointers in the AVFrame */
    auto status = LibavcodecLibrary::avcodec_fill_audio_frame(frame,
        LibavcodecLibrary::getChannelCount(context), LibavcodecLibrary::getSampleFomat(context),
        reinterpret_cast<const uint8_t*>(buffer.data), bufferSize, 0);

    if(status < 0)
    {
        throw std::runtime_error("Could not setup audio frame");
    }

    for(size_t sample = 0; sample < header.samples; sample += frame->nb_samples)
    {
        LibavcodecLibrary::av_init_packet(&packet);

        packet.data = nullptr;
        packet.size = 0;

        std::memcpy(buffer.data(), data.getDataForTimestep(sample), bufferSize);

        int gotOutput = 0;

        auto status = LibavcodecLibrary::avcodec_encode_audio2(context, &packet,
            frame, &gotOutput);

        if(status < 0)
        {
            throw std::runtime_error("Error encoding audio frame.");
        }

        if(gotOutput != 0)
        {
            file.write(packet.data, packet.size);
            LibavcodecLibrary::av_free_packet(&packet);
        }
    }

    for(int gotOutput = true; gotOutput != 0; )
    {
        auto status = LibavcodecLibrary::avcodec_encode_audio2(context, &packet,
            frame, &gotOutput);

        if(status < 0)
        {
            throw std::runtime_error("Error encoding audio frame.");
        }

        if(gotOutput != 0)
        {
            file.write(packet.data, packet.size);
        }
    }
}

LibavcodecAudioLibrary::StringVector LibavcodecAudioLibrary::getSupportedExtensions() const
{
    return StringVector(util::split(".mp4|.mp2|.mp3|.wav|.flac", "|"));
}

}

}






