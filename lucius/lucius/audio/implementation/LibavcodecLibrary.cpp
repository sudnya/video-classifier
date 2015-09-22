/*    \file   LibavcodecLibrary.cpp
    \date   August 15, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the LibavcodecLibrary class.
*/

// Lucius Includes
#include <lucius/audio/interface/LibavcodecLibrary.h>

#include <lucius/util/interface/Casts.h>
#include <lucius/util/interface/string.h>

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

LibavcodecLibrary::AVCodecContext* LibavcodecLibrary::AVCodecContextRAII::operator->()
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

LibavcodecLibrary::AVFrame* LibavcodecLibrary::AVFrameRAII::operator->()
{
    return _frame;
}

const LibavcodecLibrary::AVFrame* LibavcodecLibrary::AVFrameRAII::operator->() const
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

std::string LibavcodecLibrary::getErrorCode(int32_t _code)
{
    int32_t code = -_code;

    char* codes = reinterpret_cast<char*>(&code);

    std::string result;

    result.push_back(codes[1]);
    result.push_back(codes[2]);
    result.push_back(codes[3]);

    return result;
}

size_t LibavcodecLibrary::getNumberOfSamples(const AVFrame* frame)
{
    return frame->nb_samples;
}

void* LibavcodecLibrary::getData(AVFrame* frame)
{
    return frame->data[0];
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
    return context->sample_rate;
}

size_t LibavcodecLibrary::getChannelCount(const AVCodecContext* context)
{
    return context->channels;
}

size_t LibavcodecLibrary::getFrameSize(const AVCodecContext* context)
{
    return context->frame_size;
}

enum LibavcodecLibrary::AVSampleFormat LibavcodecLibrary::getSampleFormat(
    const AVCodecContext* context)
{
    return context->sample_fmt;
}

int64_t LibavcodecLibrary::getChannelLayout(const AVCodecContext* context)
{
    return context->channel_layout;
}

void LibavcodecLibrary::setBitRate(AVCodecContext* context, size_t bitRate)
{
    int status = av_opt_set_int(context, "maxrate", bitRate, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set bit rate: code '" + getErrorCode(status) + "'");
    }
}

void LibavcodecLibrary::setSampleFormat(AVCodecContext* context, enum AVSampleFormat format)
{
    context->sample_fmt = format;
}

void LibavcodecLibrary::setSampleRate(AVCodecContext* codec, size_t rate)
{
    /*
    int status = av_opt_set_int(codec, "ar", rate, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set sample rate.");
    }
    */
   // codec->time_base   = {0, 0};
    codec->sample_rate = rate;
}

void LibavcodecLibrary::setChannelLayout(AVCodecContext* codec, int64_t layout)
{
    int status = av_opt_set_int(codec, "channel_layout", layout, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set channel layout.");
    }
}

void LibavcodecLibrary::setChannelCount(AVCodecContext* codec, size_t count)
{
    int status = av_opt_set_int(codec, "ac", count, 0);

    if(status < 0)
    {
        throw std::runtime_error("Failed to set channel count: code '" +
            getErrorCode(status) + "'");
    }
}

void LibavcodecLibrary::setNumberOfSamples(AVFrame* frame, size_t samples)
{
    frame->nb_samples = samples;
}

void LibavcodecLibrary::setSampleFormat(AVFrame* frame, enum AVSampleFormat format)
{
    frame->format = format;
}

void LibavcodecLibrary::setChannelLayout(AVFrame* frame, int64_t layout)
{
    av_frame_set_channel_layout(frame, layout);
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

void LibavcodecLibrary::av_codec_set_pkt_timebase(AVCodecContext* avctx, AVRational val)
{
    _check();

    if(_interface.av_codec_set_pkt_timebase == nullptr)
    {
        //avctx->pkt_timebase = val;
        return;
    }

    return (*_interface.av_codec_set_pkt_timebase)(avctx, val);
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

int LibavcodecLibrary::av_opt_set(void* obj, const char* name, const char* val, int search_flags)
{
    _check();

    return (*_interface.av_opt_set)(obj, name, val, search_flags);
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

int LibavcodecLibrary::av_opt_show2(void* obj, void* av_log_obj, int req_flags, int rej_flags)
{
    _check();

    return (*_interface.av_opt_show2)(obj, av_log_obj, req_flags, rej_flags);
}

void LibavcodecLibrary::av_frame_set_channel_layout(AVFrame* frame, int64_t val)
{
    _check();

    if(_interface.av_frame_set_channel_layout == nullptr)
    {
        frame->channel_layout = val;
    }

    return (*_interface.av_frame_set_channel_layout)(frame, val);
}

int LibavcodecLibrary::av_get_channel_layout_nb_channels(uint64_t layout)
{
    _check();

    return (*_interface.av_get_channel_layout_nb_channels)(layout);
}

LibavcodecLibrary::AVIOContext* LibavcodecLibrary::avio_alloc_context(unsigned char* buffer,
    int buffer_size,
    int write_flag,
    void* opaque,
    int(*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int(*write_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t(*seek)(void* opaque, int64_t offset, int whence))
{
    _check();

    return (*_interface.avio_alloc_context)(buffer, buffer_size, write_flag, opaque,
        read_packet, write_packet, seek);
}

LibavcodecLibrary::AVFormatContext* LibavcodecLibrary::avformat_alloc_context()
{
    _check();

    return (*_interface.avformat_alloc_context)();
}

void LibavcodecLibrary::avformat_free_context(AVFormatContext* context)
{
    _check();

    return (*_interface.avformat_free_context)(context);
}

int LibavcodecLibrary::LibavcodecLibrary::avformat_open_input(AVFormatContext** ps,
    const char* filename, AVInputFormat* fmt, AVDictionary** options)
{
    _check();

    int status = (*_interface.avformat_open_input)(ps, filename, fmt, options);

    return status;
}

int LibavcodecLibrary::avformat_find_stream_info(AVFormatContext* ic, AVDictionary** options)
{
    _check();

    int status = (*_interface.avformat_find_stream_info)(ic, options);

    return status;
}

LibavcodecLibrary::AVStream* LibavcodecLibrary::avformat_new_stream(AVFormatContext* s,
    const AVCodec* c)
{
    _check();

    return (*_interface.avformat_new_stream)(s, c);
}

int LibavcodecLibrary::avformat_write_header(AVFormatContext* s, AVDictionary** options)
{
    _check();

    return (*_interface.avformat_write_header)(s, options);
}

int LibavcodecLibrary::av_write_trailer(AVFormatContext* s)
{
    _check();

    return (*_interface.av_write_trailer)(s);
}

int LibavcodecLibrary::av_write_frame(AVFormatContext* s, AVPacket* pkt)
{
    _check();

    return (*_interface.av_write_frame)(s, pkt);
}

int LibavcodecLibrary::av_read_frame(AVFormatContext* s, AVPacket* pkt)
{
    _check();

    return (*_interface.av_read_frame)(s, pkt);
}

LibavcodecLibrary::AVOutputFormat* LibavcodecLibrary::av_guess_format(const char* short_name,
    const char* filename, const char* mime_type)
{
    _check();

    return (*_interface.av_guess_format)(short_name, filename, mime_type);
}

int LibavcodecLibrary::avcodec_copy_context(AVCodecContext* dest, const AVCodecContext* src)
{
    _check();

    return (*_interface.avcodec_copy_context)(dest, src);
}

void* LibavcodecLibrary::av_malloc(size_t size)
{
    _check();

    return (*_interface.av_malloc)(size);
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
    const char* libraryName = "libavformat.dylib";
    #else
    const char* libraryName = "libavformat.so";
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
    DynLink(av_register_all);

    DynLink(avcodec_find_decoder_by_name);
    DynLink(avcodec_find_encoder_by_name);

    DynLink(av_init_packet);
    DynLink(avcodec_alloc_context3);
    DynLink(avcodec_open2);
    DynLink(avcodec_decode_audio4);
    DynLink(av_samples_get_buffer_size);

    DynLink(avcodec_fill_audio_frame);
    DynLink(avcodec_encode_audio2);

    DynLink(av_opt_get_int);
    DynLink(av_opt_get);

    DynLink(av_opt_set);
    DynLink(av_opt_set_int);

    DynLink(av_opt_show2);

    DynLink(av_get_channel_layout_nb_channels);

    DynLink(avformat_alloc_context);
    DynLink(avformat_free_context);

    DynLink(avio_alloc_context);

    DynLink(avformat_open_input);
    DynLink(avformat_find_stream_info);

    DynLink(avformat_new_stream);
    DynLink(avformat_write_header);
    DynLink(av_write_trailer);

    DynLink(av_read_frame);
    DynLink(av_write_frame);

    DynLink(av_guess_format);

    DynLink(avcodec_copy_context);

    DynLink(av_malloc);

    DynLink(av_free);
    DynLink(av_free_packet);
    DynLink(avcodec_close);

    #undef DynLink

    _tryLink(reinterpret_cast<void*&>(av_frame_alloc), {"av_frame_alloc", "avcodec_alloc_frame"});
    _tryLink(reinterpret_cast<void*&>(av_frame_free),  {"av_frame_free",  "av_free"});

    // Optionally supported functions
    util::bit_cast(av_frame_set_channel_layout, dlsym(_library, "av_frame_set_channel_layout"));
    util::bit_cast(av_codec_set_pkt_timebase, dlsym(_library, "av_codec_set_pkt_timebase"));
    util::bit_cast(av_opt_get_sample_fmt, dlsym(_library, "av_opt_get_sample_fmt"));
    util::bit_cast(av_opt_get_channel_layout, dlsym(_library, "av_opt_get_channel_layout"));
    util::bit_cast(av_opt_set_sample_fmt, dlsym(_library, "av_opt_set_sample_fmt"));
    util::bit_cast(av_opt_set_channel_layout, dlsym(_library, "av_opt_set_channel_layout"));

    (*avcodec_register_all)();
    (*av_register_all)();

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

void LibavcodecLibrary::Interface::_tryLink(void*& function, const StringVector& names)
{
    for(auto& name : names)
    {
        util::bit_cast(function, dlsym(_library, name.c_str()));

        if(function != nullptr)
        {
            return;
        }
    }

    throw std::runtime_error("Failed to load function with any of these names {'" +
        util::join(names, ", ") + "'} from dynamic library.");
}

LibavcodecLibrary::Interface LibavcodecLibrary::_interface;

}

}




