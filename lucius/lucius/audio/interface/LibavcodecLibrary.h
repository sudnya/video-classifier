/*	\file   LibavcodecLibrary.h
	\date   August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LibavcodecLibrary class.
*/

#pragma once

// Standard Library Includes
#include <cstddef>

namespace lucius
{

namespace audio
{

class LibavcodecLibrary
{
public:
    class AVCodec;
    class AVCodecContext;

    class AVFrame;
    class AVDictionary;

    class AVBufferRef;
    class AVPacketSideData;

    const int AUDIO_INBUF_SIZE = 20480;
    const int AV_INPUT_BUFFER_PADDING_SIZE = 32;

    typedef struct AVPacket {
        /**
         * A reference to the reference-counted buffer where the packet data is
         * stored.
         * May be NULL, then the packet data is not reference-counted.
         */
        AVBufferRef* buf;
        /**
         * Presentation timestamp in AVStream->time_base units; the time at which
         * the decompressed packet will be presented to the user.
         * Can be AV_NOPTS_VALUE if it is not stored in the file.
         * pts MUST be larger or equal to dts as presentation cannot happen before
         * decompression, unless one wants to view hex dumps. Some formats misuse
         * the terms dts and pts/cts to mean something different. Such timestamps
         * must be converted to true pts/dts before they are stored in AVPacket.
         */
        int64_t pts;
        /**
         * Decompression timestamp in AVStream->time_base units; the time at which
         * the packet is decompressed.
         * Can be AV_NOPTS_VALUE if it is not stored in the file.
         */
        int64_t dts;
        uint8_t* data;
        int size;
        int stream_index;
        /**
         * A combination of AV_PKT_FLAG values
         */
        int flags;
        /**
         * Additional packet data that can be provided by the container.
         * Packet can contain several types of side information.
         */
        AVPacketSideData *side_data;
        int side_data_elems;

        /**
         * Duration of this packet in AVStream->time_base units, 0 if unknown.
         * Equals next_pts - this_pts in presentation order.
         */
        int duration;

        uint64_t who_cares[16];
    } AVPacket;

public:
	static void load();
	static bool loaded();

public:
	static void av_init_packet(AVPacket* packet);
    static AVCodecContext* avcodec_alloc_context3(AVCodec* codec);
    static int avcodec_open2(AVCodecContext* avctx, const AVCodec* codec, AVDictionary** options);
    static AVFrame* av_frame_alloc();
    static int avcodec_decode_audio4(AVCodecContext* avctx, AVFrame* frame,
        int* got_frame_ptr, const AVPacket* avpkt);
    static int av_samples_get_buffer_size(int* linesize, int nb_channels, int nb_samples,
        enum AVSampleFormat sample_fmt, int align);

    static void av_free(void*);
    static int avcodec_close(AVCodecContext* avctx);
    static void av_frame_free(AVFrame** frame);

private:
	static void _check();

private:
	class Interface
	{
	public:
	    void (*av_init_packet)(AVPacket* packet);

        AVCodecContext* (*avcodec_alloc_context3)(AVCodec* codec);
        int (*avcodec_open2)(AVCodecContext* avctx, const AVCodec* codec, AVDictionary** options);
        AVFrame* (*av_frame_alloc)();
        int (*avcodec_decode_audio4)(AVCodecContext* avctx, AVFrame* frame,
            int* got_frame_ptr, const AVPacket* avpkt);
        int (*av_samples_get_buffer_size)(int* linesize, int nb_channels, int nb_samples,
            enum AVSampleFormat sample_fmt, int align);

        void (*av_free)(void*);
        int (*avcodec_close)(AVCodecContext* avctx);
        void (*av_frame_free)(AVFrame** frame);

	public:
		/*! \brief The constructor zeros out all of the pointers */
		Interface();

		/*! \brief The destructor closes dlls */
		~Interface();
		/*! \brief Load the library */
		void load();
		/*! \brief Has the library been loaded? */
		bool loaded() const;
		/*! \brief unloads the library */
		void unload();

	private:

		void* _library;
	};

private:
	static Interface _interface;

};

}

}



