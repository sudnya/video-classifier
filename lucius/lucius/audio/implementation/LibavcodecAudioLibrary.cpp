/*    \file   LibavcodecAudioLibrary.cpp
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
#include <iostream>
#include <stdexcept>
#include <memory>
#include <cstring>
#include <list>

namespace lucius
{

namespace audio
{

LibavcodecAudioLibrary::~LibavcodecAudioLibrary()
{

}

static int readFunction(void* opaqueStream, uint8_t* buffer, int bufferSize)
{
    auto& stream = *reinterpret_cast<std::istream*>(opaqueStream);

    stream.read(reinterpret_cast<char*>(buffer), bufferSize);

    return stream.gcount();
}

static int writeFunction(void* opaqueStream, uint8_t* buffer, int bufferSize)
{
    auto& stream = *reinterpret_cast<std::ostream*>(opaqueStream);

    stream.write(reinterpret_cast<char*>(buffer), bufferSize);

    if(!stream.good())
    {
        return -1;
    }

    return bufferSize;
}

static int64_t seekFunction(void* opaqueStream, int64_t offset, int whence)
{
    auto& stream = *reinterpret_cast<std::istream*>(opaqueStream);

    if(whence == LibavcodecLibrary::AVSEEK_SIZE)
    {
        size_t position = stream.tellg();

        stream.seekg(0, std::ios::beg);
        size_t begin = stream.tellg();

        stream.seekg(0, std::ios::end);
        size_t end = stream.tellg();

        stream.seekg(position, std::ios::beg);

        return end - begin;
    }

    stream.seekg(offset, std::ios::beg);

    return stream.tellg();
}

static int64_t seekWriteFunction(void* opaqueStream, int64_t offset, int whence)
{
    auto& stream = *reinterpret_cast<std::ostream*>(opaqueStream);

    if(whence == LibavcodecLibrary::AVSEEK_SIZE)
    {
        size_t position = stream.tellp();

        stream.seekp(0, std::ios::beg);
        size_t begin = stream.tellp();

        stream.seekp(0, std::ios::end);
        size_t end = stream.tellp();

        stream.seekp(position, std::ios::beg);

        return end - begin;
    }

    stream.seekp(offset, std::ios::beg);

    return stream.tellp();
}

class DataBuffer
{
public:
    DataBuffer()
    : _bufferSize(1 << 20)
    {

    }

    void append(const void* data, size_t size)
    {
        _adjustSize(size);

        size_t position = _getCurrentPosition();

        _buffers.back().resize(position + size);

        std::memcpy(&_buffers.back()[position], data, size);
    }

    std::vector<uint8_t> getData() const
    {
        std::vector<uint8_t> contiguousData(totalSize());

        size_t position = 0;

        for(auto& buffer : _buffers)
        {
            std::memcpy(&contiguousData[position], buffer.data(), buffer.size());
            position += buffer.size();
        }

        return contiguousData;
    }

    size_t totalSize() const
    {
        size_t size = 0;

        for(auto& buffer : _buffers)
        {
            size += buffer.size();
        }

        return size;
    }

private:
    void _adjustSize(size_t size)
    {
        if(!_buffers.empty() && (size + _getCurrentPosition() <= _bufferSize))
        {
            return;
        }

        if(size > _bufferSize)
        {
            _bufferSize = size;
        }

        _buffers.push_back(BufferEntry());

        _buffers.back().reserve(_bufferSize);
    }

    size_t _getCurrentPosition() const
    {
        if(_buffers.empty())
        {
            return 0;
        }

        return _buffers.back().size();
    }

private:
    typedef std::vector<uint8_t> BufferEntry;
    typedef std::list<BufferEntry> BufferList;

private:
    BufferList _buffers;
    size_t     _bufferSize;

};

LibavcodecAudioLibrary::HeaderAndData LibavcodecAudioLibrary::loadAudio(std::istream& stream,
    const std::string& format)
{
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

    size_t bufferSize = LibavcodecLibrary::AUDIO_INBUF_SIZE +
        LibavcodecLibrary::AV_INPUT_BUFFER_PADDING_SIZE;

    std::unique_ptr<uint8_t, void(*)(void*)> buffer(reinterpret_cast<uint8_t*>(
        LibavcodecLibrary::av_malloc(bufferSize)),
        &LibavcodecLibrary::av_free);

    if(!buffer)
    {
        throw std::runtime_error("Failed to allocate buffer.");
    }

    std::unique_ptr<LibavcodecLibrary::AVIOContext, void(*)(void*)> avioContext(
        LibavcodecLibrary::avio_alloc_context(
        buffer.get(), bufferSize, 0, reinterpret_cast<void*>(static_cast<std::istream*>(&stream)),
        &readFunction, nullptr, &seekFunction), &LibavcodecLibrary::av_free);

    if(!avioContext)
    {
        throw std::runtime_error("Failed to allocate avio context.");
    }

    std::unique_ptr<LibavcodecLibrary::AVFormatContext,
        void(*)(LibavcodecLibrary::AVFormatContext*)> avFormat(
            LibavcodecLibrary::avformat_alloc_context(),
            &LibavcodecLibrary::avformat_free_context);

    if(!avFormat)
    {
        throw std::runtime_error("Failed to allocate avformat context.");
    }

    avFormat->pb = avioContext.get();

    auto avFormatPtr = avFormat.get();

    auto openStatus = LibavcodecLibrary::avformat_open_input(&avFormatPtr,
        "invalidFilename", nullptr, nullptr);

    if(openStatus < 0)
    {
        // the library frees the format context on error
        avFormat.release();
        buffer.release();

        throw std::runtime_error("Failed to open avformat input with error " +
            LibavcodecLibrary::getErrorCode(openStatus) + ".");
    }

    status = LibavcodecLibrary::avformat_find_stream_info(avFormatPtr, nullptr);

    if(status < 0)
    {
        throw std::runtime_error("Failed to find stream info with error " +
            LibavcodecLibrary::getErrorCode(status) + ".");
    }

    LibavcodecLibrary::av_codec_set_pkt_timebase(context,
        {1, static_cast<int>(LibavcodecLibrary::getSamplingRate(context))});

    LibavcodecLibrary::AVFrameRAII decodedFrame;

    DataBuffer dataBuffer;

    while(true)
    {
        LibavcodecLibrary::AVPacketRAII packet;

        int gotPacket = LibavcodecLibrary::av_read_frame(avFormat.get(), packet);

        if(gotPacket < 0)
        {
            if(headerAndData.header.samples == 0)
            {
                throw std::runtime_error("Failed to get any samples.");
            }
            break;
        }

        int gotFrame = 0;

        int length = LibavcodecLibrary::avcodec_decode_audio4(context,
             decodedFrame, &gotFrame, packet);

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

            int dataSize = 0;

            LibavcodecLibrary::av_samples_get_buffer_size(&dataSize,
                LibavcodecLibrary::getChannelCount(context),
                LibavcodecLibrary::getNumberOfSamples(decodedFrame),
                LibavcodecLibrary::getSampleFormat(context), 1);
            assert(dataSize >= 0);

            dataBuffer.append(LibavcodecLibrary::getData(decodedFrame), dataSize);
        }

        if(length != packet->size)
        {
            throw std::runtime_error("Did not decode the entire packet.");
        }
    }

    LibavcodecLibrary::av_free(avioContext->buffer);
    buffer.release();

    headerAndData.data = dataBuffer.getData();

    return headerAndData;
}

LibavcodecAudioLibrary::Header LibavcodecAudioLibrary::loadAudioHeader(std::istream& stream,
    const std::string& format)
{
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

    size_t bufferSize = LibavcodecLibrary::AUDIO_INBUF_SIZE +
        LibavcodecLibrary::AV_INPUT_BUFFER_PADDING_SIZE;

    std::unique_ptr<uint8_t, void(*)(void*)> buffer(reinterpret_cast<uint8_t*>(
        LibavcodecLibrary::av_malloc(bufferSize)),
        &LibavcodecLibrary::av_free);

    if(!buffer)
    {
        throw std::runtime_error("Failed to allocate buffer.");
    }

    Header header;

    std::unique_ptr<LibavcodecLibrary::AVIOContext, void(*)(void*)> avioContext(
        LibavcodecLibrary::avio_alloc_context(
        buffer.get(), bufferSize, 0, reinterpret_cast<void*>(static_cast<std::istream*>(&stream)),
        &readFunction, nullptr, &seekFunction), &LibavcodecLibrary::av_free);

    if(!avioContext)
    {
        throw std::runtime_error("Failed to allocate avio context.");
    }

    std::unique_ptr<LibavcodecLibrary::AVFormatContext,
        void(*)(LibavcodecLibrary::AVFormatContext*)> avFormat(
            LibavcodecLibrary::avformat_alloc_context(),
            &LibavcodecLibrary::avformat_free_context);

    if(!avFormat)
    {
        throw std::runtime_error("Failed to allocate avformat context.");
    }

    avFormat->pb = avioContext.get();

    auto avFormatPtr = avFormat.get();

    auto openStatus = LibavcodecLibrary::avformat_open_input(&avFormatPtr,
        "invalidFilename", nullptr, nullptr);

    if(openStatus < 0)
    {
        // the library frees the format context on error
        avFormat.release();
        buffer.release();

        throw std::runtime_error("Failed to open avformat input with error " +
            LibavcodecLibrary::getErrorCode(openStatus) + ".");
    }

    status = LibavcodecLibrary::avformat_find_stream_info(avFormatPtr, nullptr);

    if(status < 0)
    {
        throw std::runtime_error("Failed to find stream info with error " +
            LibavcodecLibrary::getErrorCode(status) + ".");
    }

    LibavcodecLibrary::av_codec_set_pkt_timebase(context,
        {1, static_cast<int>(LibavcodecLibrary::getSamplingRate(context))});

    header.samplingRate = LibavcodecLibrary::getSamplingRate(context);
    header.samples = (avFormatPtr->duration / LibavcodecLibrary::AV_TIME_BASE) *
        header.samplingRate;

    return header;
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

    size_t ioBufferSize = LibavcodecLibrary::AUDIO_INBUF_SIZE +
        LibavcodecLibrary::AV_INPUT_BUFFER_PADDING_SIZE;

    std::unique_ptr<uint8_t, void(*)(void*)> ioBuffer(reinterpret_cast<uint8_t*>(
        LibavcodecLibrary::av_malloc(ioBufferSize)),
        &LibavcodecLibrary::av_free);

    if(!ioBuffer)
    {
        throw std::runtime_error("Failed to allocate IO buffer.");
    }

    std::unique_ptr<LibavcodecLibrary::AVIOContext, void(*)(void*)> avioContext(
        LibavcodecLibrary::avio_alloc_context(
        ioBuffer.get(), ioBufferSize, 1,
        reinterpret_cast<void*>(static_cast<std::ostream*>(&stream)),
        nullptr, &writeFunction, &seekWriteFunction), &LibavcodecLibrary::av_free);

    if(!avioContext)
    {
        throw std::runtime_error("Failed to allocate avio context.");
    }

    std::unique_ptr<LibavcodecLibrary::AVFormatContext,
        void(*)(LibavcodecLibrary::AVFormatContext*)> formatContext(
            LibavcodecLibrary::avformat_alloc_context(),
            &LibavcodecLibrary::avformat_free_context);

    if(!formatContext)
    {
        throw std::runtime_error("Failed to allocate avformat context.");
    }

    formatContext->pb = avioContext.get();
    formatContext->oformat = LibavcodecLibrary::av_guess_format(codecName.c_str(),
        format.c_str(), nullptr);

    auto avFormatStream = LibavcodecLibrary::avformat_new_stream(formatContext.get(), codec);

    if(avFormatStream == nullptr)
    {
        throw std::runtime_error("Failed to add stream to format context.");
    }

    auto context = avFormatStream->codec;

    avFormatStream->id = formatContext->nb_streams - 1;
    avFormatStream->time_base = {1, static_cast<int>(header.samplingRate)};

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

    status = LibavcodecLibrary::avformat_write_header(formatContext.get(), nullptr);

    if(status < 0)
    {
        throw std::runtime_error("Failed to write header to stream.");
    }

    /* the codec gives us the frame size, in samples,
     * we calculate the size of the samples buffer in bytes */
    int bufferSize = LibavcodecLibrary::av_samples_get_buffer_size(nullptr,
        1, LibavcodecLibrary::getFrameSize(context),
        LibavcodecLibrary::getSampleFormatWithBytes(header.bytesPerSample), 0);

    DataVector buffer(bufferSize);

    LibavcodecLibrary::AVFrameRAII frame;

    LibavcodecLibrary::setNumberOfSamples(frame, LibavcodecLibrary::getFrameSize(context));
    LibavcodecLibrary::setSampleFormat(frame,
        LibavcodecLibrary::getSampleFormatWithBytes(header.bytesPerSample));
    LibavcodecLibrary::setChannelLayout(frame, selectChannelLayout(codec));

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

        frame->pts = sample;

        LibavcodecLibrary::AVPacketRAII packet;

        packet->data = nullptr;
        packet->size = 0;

        int gotOutput = 0;

        auto status = LibavcodecLibrary::avcodec_encode_audio2(context, packet,
            frame, &gotOutput);

        packet->pts = sample + 1;

        if(status < 0)
        {
            throw std::runtime_error("Error encoding audio frame.");
        }

        if(gotOutput != 0)
        {
            LibavcodecLibrary::av_write_frame(formatContext.get(), packet);
        }
    }

    for(int gotOutput = true; gotOutput != 0; )
    {
        LibavcodecLibrary::AVPacketRAII packet;

        packet->data = nullptr;
        packet->size = 0;

        auto status = LibavcodecLibrary::avcodec_encode_audio2(context, packet,
            nullptr, &gotOutput);

        packet->pts = header.samples + 1;

        if(status < 0)
        {
            throw std::runtime_error("Error encoding audio frame.");
        }

        if(gotOutput != 0)
        {
            LibavcodecLibrary::av_write_frame(formatContext.get(), packet);
        }
    }

    status = LibavcodecLibrary::av_write_trailer(formatContext.get());

    if(status < 0)
    {
        throw std::runtime_error("Failed to write trailer to stream.");
    }
}

LibavcodecAudioLibrary::StringVector LibavcodecAudioLibrary::getSupportedExtensions() const
{
    return StringVector(util::split(".mp4|.mp2|.mp3|.flac", "|"));
}

size_t LibavcodecAudioLibrary::getBitRate() const
{
    return util::KnobDatabase::getKnobValue("LibavcodecAudioLibrary::Bitrate", 64000);
}

}

}






