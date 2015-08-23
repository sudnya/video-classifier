/*	\file   LibavcodecLibrary.cpp
	\date   August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LibavcodecLibrary class.
*/

// Lucius Includes
#include <lucius/audio/interface/LibavcodecLibrary.h>

#include <lucius/util/interface/Casts.h>

// Standard Library Includes
#include <stdexcept>
#include <cassert>

// System-Specific Includes
#include <dlfcn.h>

namespace lucius
{

namespace audio
{

LibavcodecLibrary::AVCodecContextRAII::AVCodecContextRAII(AVCodec* codec)
: _context(nullptr)
{
    _context = avcodec_alloc_context3(codec);

    if(_context == nullptr)
    {
        throw std::runtime_error("Failed to allocate libavcodec context.");
    }
}

LibavcodecLibrary::AVCodecContextRAII::~AVCodecContextRAII()
{
    avcodec_close(_context);
    av_free(_context);
}

LibavcodecLibrary::AVCodecContextRAII::operator AVCodecContext*()
{
    return _context;
}

LibavcodecLibrary::AVCodecContextRAII::operator const AVCodecContext*() const
{
    return _context;
}

LibavcodecLibrary::LibavcodecLibrary::AVFrameRAII::AVFrameRAII()
: _frame(nullptr)
{
    _frame = av_frame_alloc();

    if(_frame == nullptr)
    {
        throw std::runtime_error("Failed to allocate frame.");
    }
}

LibavcodecLibrary::AVFrameRAII::~AVFrameRAII()
{
    av_frame_free(&_frame);
}

LibavcodecLibrary::AVFrameRAII::operator AVFrame*()
{
    return _frame;
}

LibavcodecLibrary::AVFrameRAII::operator const AVFrame*() const
{
    return _frame;
}

LibavcodecLibrary::AVPacketRAII::AVPacketRAII()
{
    av_init_packet(&_packet);
}

LibavcodecLibrary::AVPacketRAII::~AVPacketRAII()
{
    av_free_packet(&_packet);
}

LibavcodecLibrary::AVPacketRAII::operator AVPacket*()
{
    return &_packet;
}

LibavcodecLibrary::AVPacketRAII::operator const AVPacket*() const
{
    return &_packet;
}

LibavcodecLibrary::AVPacket* LibavcodecLibrary::AVPacketRAII::operator->()
{
    return &_packet;
}

const LibavcodecLibrary::AVPacket* LibavcodecLibrary::AVPacketRAII::operator->() const
{
    return &_packet;
}

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

    int status = av_opt_get_int(const_cast<AVFrame*>(frame), "nb_samples", 0, &result);

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

const LibavcodecLibrary::AVSampleFormat* LibavcodecLibrary::getSampleFormats(const AVCodec* codec)
{
    enum AVSampleFormat* result = nullptr;

    int status = av_opt_get(const_cast<AVCodec*>(codec), "sample_fmts", 0,
        reinterpret_cast<uint8_t**>(&result));

    if(status < 0)
    {
        throw std::runtime_error("Failed to get sample formats.");
    }

    return result;
}

size_t LibavcodecLibrary::getBytesPerSampleForFormat(const AVCodecContext* context)
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

LibavcodecLibrary::AVSampleFormat LibavcodecLibrary::getSampleFormatWithBytes(size_t bytes)
{
    switch(bytes)
    {
    case 1:
    {
        return AV_SAMPLE_FMT_U8;
    }
    case 2:
    {
        return AV_SAMPLE_FMT_S16;
    }
    case 4:
    {
        return AV_SAMPLE_FMT_S32;
    }
    case 8:
    {
        return AV_SAMPLE_FMT_DBL;
    }
    default:
    {
        break;
    }
    }

    assert(false);

    return AV_SAMPLE_FMT_NONE;
}

size_t LibavcodecLibrary::getSamplingRate(const AVCodecContext* context)
{
    int64_t result = 0;

    int status = av_opt_get_int(const_cast<AVCodecContext*>(context), "sample_rate", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get sampling rate.");
    }

    return result;
}

size_t LibavcodecLibrary::getChannelCount(const AVCodecContext* context)
{
    int64_t result = 0;

    int status = av_opt_get_int(const_cast<AVCodecContext*>(context), "channels", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get channel count.");
    }

    return result;
}

size_t LibavcodecLibrary::getFrameSize(const AVCodecContext* context)
{
    int64_t result = 0;

    int status = av_opt_get_int(const_cast<AVCodecContext*>(context), "frame_size", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get frame size.");
    }

    return result;
}

enum LibavcodecLibrary::AVSampleFormat LibavcodecLibrary::getSampleFormat(
    const AVCodecContext* context)
{
    enum AVSampleFormat result;

    int status = av_opt_get_sample_fmt(const_cast<AVCodecContext*>(context),
        "sample_fmt", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get sample format.");
    }

    return result;
}

int64_t LibavcodecLibrary::getChannelLayout(const AVCodecContext* context)
{
    int64_t result = 0;

    int status = av_opt_get_channel_layout(const_cast<AVCodecContext*>(context),
        "channel_layout", 0, &result);

    if(status < 0)
    {
        throw std::runtime_error("Failed to get frame size.");
    }

    return result;
}

void LibavcodecLibrary::setBitRate(AVCodecContext* context, size_t bitRate)
{
    int status = av_opt_set_int(context, "bit_rate", bitRate, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set bit rate.");
    }
}

void LibavcodecLibrary::setSampleFormat(AVCodecContext* context, enum AVSampleFormat format)
{
    int status = av_opt_set_sample_fmt(context, "sample_fmt", format, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set sample format.");
    }
}

void LibavcodecLibrary::setSampleRate(AVCodec* codec, size_t rate)
{
    int status = av_opt_set_int(codec, "sample_rate", rate, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set sample rate.");
    }
}

void LibavcodecLibrary::setChannelLayout(AVCodec* codec, int64_t layout)
{
    int status = av_opt_set_channel_layout(codec, "channel_layout", layout, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set channel layout.");
    }
}

void LibavcodecLibrary::setChannelCount(AVCodec* codec, size_t count)
{
    int status = av_opt_set_int(codec, "channels", count, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set channel count.");
    }
}

void LibavcodecLibrary::setNumberOfSamples(AVFrame* frame, size_t samples)
{
    int status = av_opt_set_int(frame, "nb_samples", samples, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set sample count.");
    }
}

void LibavcodecLibrary::setSampleFormat(AVFrame* frame, enum AVSampleFormat format)
{
    int status = av_opt_set_sample_fmt(frame, "sample_fmt", format, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set sample format.");
    }
}

void LibavcodecLibrary::setChannelLayout(AVFrame* frame, int64_t layout)
{
    int status = av_opt_set_channel_layout(frame, "channel_layout", layout, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set channel layout.");
    }
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

LibavcodecLibrary::AVCodecContext* LibavcodecLibrary::avcodec_alloc_context3(AVCodec* codec)
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

LibavcodecLibrary::AVFrame* LibavcodecLibrary::av_frame_alloc()
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

int LibavcodecLibrary::av_opt_get_int(void* obj, const char* name, int flags, int64_t* out_val)
{
    _check();

    return (*_interface.av_opt_get_int)(obj, name, flags, out_val);
}


int LibavcodecLibrary::av_opt_get(void* obj, const char* name, int flags, uint8_t** out_val)
{
    _check();

    return (*_interface.av_opt_get)(obj, name, flags, out_val);
}

int LibavcodecLibrary::av_opt_get_sample_fmt(void* obj, const char* name, int search_flags,
    enum AVSampleFormat* out_fmt)
{
    _check();

    return (*_interface.av_opt_get_sample_fmt)(obj, name, search_flags, out_fmt);
}

int LibavcodecLibrary::av_opt_get_channel_layout(void* obj, const char* name, int search_flags,
    int64_t* ch_layout)
{
    _check();

    return (*_interface.av_opt_get_channel_layout)(obj, name, search_flags, ch_layout);
}

int LibavcodecLibrary::av_opt_set_int(void* obj, const char* name, int64_t val, int search_flags)
{
    _check();

    return (*_interface.av_opt_set_int)(obj, name, val, search_flags);
}

int LibavcodecLibrary::av_opt_set_sample_fmt(void* obj, const char* name,
    enum AVSampleFormat fmt, int search_flags)
{
    _check();

    return (*_interface.av_opt_set_sample_fmt)(obj, name, fmt, search_flags);
}

int LibavcodecLibrary::av_opt_set_channel_layout(void* obj, const char* name, int64_t ch_layout,
    int search_flags)
{
    _check();

    return (*_interface.av_opt_set_channel_layout)(obj, name, ch_layout, search_flags);
}

void LibavcodecLibrary::av_free(void* pointer)
{
    _check();

    return (*_interface.av_free)(pointer);
}

void LibavcodecLibrary::av_free_packet(AVPacket* packet)
{
    _check();

    return (*_interface.av_free_packet)(packet);
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

	DynLink(avcodec_register_all);

	DynLink(avcodec_find_decoder_by_name);
	DynLink(avcodec_find_encoder_by_name);

	DynLink(av_init_packet);
	DynLink(avcodec_alloc_context3);
	DynLink(av_frame_alloc);
	DynLink(avcodec_decode_audio4);
	DynLink(av_samples_get_buffer_size);

	DynLink(avcodec_fill_audio_frame);
	DynLink(avcodec_encode_audio2);

	DynLink(av_opt_get_int);
	DynLink(av_opt_get);
	DynLink(av_opt_get_sample_fmt);
	DynLink(av_opt_get_channel_layout);

	DynLink(av_opt_set_int);
	DynLink(av_opt_set_sample_fmt);
	DynLink(av_opt_set_channel_layout);

	DynLink(av_free);
	DynLink(av_free_packet);
	DynLink(avcodec_close);
	DynLink(av_frame_free);

	#undef DynLink

    (*avcodec_register_all)();

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

LibavcodecLibrary::Interface LibavcodecLibrary::_interface;

}

}




