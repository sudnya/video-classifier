/*	\file   LibavcodecLibrary.cpp
	\date   August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LibavcodecLibrary class.
*/

// Lucius Includes
#include <lucius/audio/interface/LibavcodecLibrary.h>

namespace lucius
{

namespace audio
{

void LibavcodecLibrary::load()
{
	_interface.load();
}

bool LibavcodecLibrary::loaded()
{
	return _interface.loaded();
}

size_t LibavcodecLibrary::getNumberOfSamples(const AVFrame* frame)
{
    int64_t result = 0;

    int status = av_opt_get_int(frame, "nb_samples", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get number of samples.");
    }

    return result;
}

void* LibavcodecLibrary::getData(AVFrame* frame)
{
    uint8_t* result = nullptr;

    int status = av_opt_get(frame, "data", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get frame data.");
    }

    return result;
}

size_t LibavcodecLibrary::getBytesPerSampleForFormat(const AVContext* context)
{
    auto format = getSampleFormat(context);

    switch(format)
    {
        case AV_SAMPLE_FMT_NONE:
        {
            break;
        }
        case AV_SAMPLE_FMT_U8:
        {
            return 1;
        }
        case AV_SAMPLE_FMT_S16:
        {
            return 2;
        }
        case AV_SAMPLE_FMT_S32:
        {
            return 4;
        }
        case AV_SAMPLE_FMT_FLT:
        {
            return 4;
        }
        case AV_SAMPLE_FMT_DBL:
        {
            return 8;
        }
        case AV_SAMPLE_FMT_U8P:
        {
            return 1;
        }
        case AV_SAMPLE_FMT_S16P:
        {
            return 2;
        }
        case AV_SAMPLE_FMT_S32P:
        {
            return 4;
        }
        case AV_SAMPLE_FMT_FLTP:
        {
            return 4;
        }
        case AV_SAMPLE_FMT_DBLP:
        {
            return 8;
        }
        case AV_SAMPLE_FMT_NB:
        {
            break;
        }
        default:
        {
            break;
        }

    }

    assert(false);

    return 0;
}

size_t LibavcodecLibrary::getSamplingRate(const AVContext* context)
{
    int64_t result = 0;

    int status = av_opt_get_int(frame, "sample_rate", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get sampling rate.");
    }

    return result;
}

size_t LibavcodecLibrary::getChannelCount(const AVContext* context)
{
    int64_t result = 0;

    int status = av_opt_get_int(frame, "channels", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get channel count.");
    }

    return result;
}

size_t LibavcodecLibrary::getFrameSize(const AVContext* context)
{
    int64_t result = 0;

    int status = av_opt_get_int(frame, "frame_size", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get frame size.");
    }

    return result;
}

enum LibavcodecLibrary::AVSampleFormat LibavcodecLibrary::getSampleFormat(const AVContext* context)
{
    enum AVSampleFormat result;

    int status = av_opt_get_sample_fmt(context, "sample_fmt", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get sample format.");
    }

    return result;
}

int64_t LibavcodecLibrary::getChannelLayout(const AVContext* context)
{
    int64_t result = 0;

    int status = av_opt_get_channel_layout(context, "channel_layout", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get frame size.");
    }

    return result;
}

void LibavcodecLibrary::setBitRate(AVContext* context, size_t bitRate)
{
    av_opt_set_int(context, "bit_rate", rate, 0);
}

void LibavcodecLibrary::setSampleFormat(AVContext* context, enum AVSampleFormat format)
{
    av_opt_set_sample_fmt(context, "sample_fmt", format, 0);
}

void LibavcodecLibrary::setSampleRate(AVCodec* codec, size_t rate)
{
    av_opt_set_int(context, "sample_rate", rate, 0);
}

void LibavcodecLibrary::setChannelLayout(AVCodec* codec, int64_t layout)
{
    av_opt_set_channel_layout(frame, "channel_layout", layout, 0);
}

void LibavcodecLibrary::setChannelCount(AVCodec* codec, size_t count)
{
    av_opt_set_int(codec, "channels", count, 0);
}

void LibavcodecLibrary::setNumberOfSamples(AVFrame* frame, size_t samples)
{
    av_opt_set_int(codec, "nb_samples", samples, 0);
}

void LibavcodecLibrary::setSampleFormat(AVFrame* frame, enum AVSampleFormat format)
{
    av_opt_set_sample_fmt(frame, "sample_fmt", format, 0);
}

void LibavcodecLibrary::setChannelLayout(AVFrame* frame, int64_t layout)
{
    av_opt_set_channel_layout(frame, "channel_layout", layout, 0);
}

LibavcodecLibrary::AVCodec* LibavcodecLibrary::avcodec_find_decoder_by_name(const char* name)
{
    _check();

    return (*_interface.avcodec_find_decoder_by_name)(name);
}

LibavcodecLibrary::AVCodec* LibavcodecLibrary::avcodec_find_encoder_by_name(const char* name)
{
    _check();

    return (*_interface.avcodec_find_encoder_by_name)(name);
}

void LibavcodecLibrary::av_init_packet(AVPacket* packet)
{
    _check();

    return (*_interface.av_init_packet)(packet);
}

AVCodecContext* LibavcodecLibrary::avcodec_alloc_context3(AVCodec* codec)
{
    _check();

    return (*_interface.avcodec_alloc_context3)(codec);
}

int LibavcodecLibrary::avcodec_open2(AVCodecContext* avctx, const AVCodec* codec,
    AVDictionary** options)
{
    _check();

    return (*_interface.avcodec_open2)(avctx, codec, options);
}

AVFrame* LibavcodecLibrary::av_frame_alloc()
{
    _check();

    return (*_interface.av_frame_alloc)();
}

int LibavcodecLibrary::avcodec_decode_audio4(AVCodecContext* avctx, AVFrame* frame,
    int* got_frame_ptr, const AVPacket* avpkt)
{
    _check();

    return (*_interface.avcodec_decode_audio4)(avctx, frame, got_frame_ptr, avpkt);
}

int LibavcodecLibrary::av_samples_get_buffer_size(int* linesize, int nb_channels, int nb_samples,
    enum AVSampleFormat sample_fmt, int align)
{
    _check();

    return (*_interface.av_samples_get_buffer_size)(linesize, nb_channels, nb_samples,
        sample_fmt, align);
}

int LibavcodecLibrary::check_sample_fmt(AVCodec* codec, enum AVSampleFormat sample_fmt)
{
    _check();

    return (*_interface.check_sample_fmt)(codec, sample_fmt);
}

int LibavcodecLibrary::avcodec_fill_audio_frame(AVFrame* frame, int nb_channels,
    enum AVSampleFormat sample_fmt, const uint8_t* buf, int buf_size, int align)
{
    _check();

    return (*_interface.avcodec_fill_audio_frame)(frame, nb_channels, sample_fmt,
        buf, buf_size, align);
}

int LibavcodecLibrary::avcodec_encode_audio2(AVCodecContext* avctx, AVPacket* avpkt,
    const AVFrame* frame, int* got_packet_ptr)
{
    _check();

    return (*_interface.avcodec_encode_audio2)(avctx, avpkt, frame, got_packet_ptr);
}

void LibavcodecLibrary::av_free(void* pointer)
{
    _check();

    return (*_interface.av_free)(pointer);
}

int LibavcodecLibrary::avcodec_close(AVCodecContext* avctx)
{
    _check();

    return (*_interface.avcodec_close)(avctx);
}

void LibavcodecLibrary::av_frame_free(AVFrame** frame)
{
    _check();

    return (*_interface.av_frame_free)(frame);
}

void LibavcodecLibrary::_check()
{
	load();

	if(!loaded())
	{
		throw std::runtime_error("Tried to call libavcodec function when "
			"the library is not loaded. Loading library failed, consider "
			"installing libavcodec.");
	}
}

LibavcodecLibrary::Interface::Interface()
: _library(nullptr)
{

}

LibavcodecLibrary::Interface::~Interface()
{
    unload();
}

static void checkFunction(void* pointer, const std::string& name)
{
	if(pointer == nullptr)
	{
		throw std::runtime_error("Failed to load function '" + name +
			"' from dynamic library.");
	}
}

void LibavcodecLibrary::Interface::load()
{
	if(loaded()) return;

    #ifdef __APPLE__
    const char* libraryName = "libavcodec.dylib";
    #else
    const char* libraryName = "libavcodec.so";
    #endif

	_library = dlopen(libraryName, RTLD_LAZY);

    util::log("LibavcodecLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
	{
        util::log("LibavcodecLibrary") << " Loading library '" << libraryName << "' failed\n";
		return;
	}

	#define DynLink(function) util::bit_cast(function, dlsym(_library, #function)); \
        checkFunction((void*)function, #function)

	DynLink(avcodec_find_decoder_by_name);
	DynLink(avcodec_find_encoder_by_name);

	DynLink(av_init_packet);
	DynLink(avcodec_alloc_context3);
	DynLink(av_frame_alloc);
	DynLink(avcodec_decode_audio4);
	DynLink(av_samples_get_buffer_size);

	DynLink(check_sample_fmt);
	DynLink(avcodec_fill_audio_frame);
	DynLink(avcodec_encode_audio2);

	DynLink(av_free);
	DynLink(avcodec_close);
	DynLink(av_frame_free);

	#undef DynLink

    util::log("LibavcodecLibrary") << " Loaded library '" << libraryName << "' succeeded\n";
}

bool LibavcodecLibrary::Interface::loaded() const
{
    return _library != nullptr;
}

void LibavcodecLibrary::Interface::unload()
{
    if(!loaded()) return;

    dlclose(_library);
    _library = nullptr;
}

}

}




