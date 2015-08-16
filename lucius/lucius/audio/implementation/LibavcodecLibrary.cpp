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

	DynLink(av_init_packet);
	DynLink(avcodec_alloc_context3);
	DynLink(av_frame_alloc);
	DynLink(avcodec_decode_audio4);
	DynLink(av_samples_get_buffer_size);
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




