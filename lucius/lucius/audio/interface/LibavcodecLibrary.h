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

    enum AVSampleFormat {
        AV_SAMPLE_FMT_NONE = -1,
        AV_SAMPLE_FMT_U8,          ///< unsigned 8 bits
        AV_SAMPLE_FMT_S16,         ///< signed 16 bits
        AV_SAMPLE_FMT_S32,         ///< signed 32 bits
        AV_SAMPLE_FMT_FLT,         ///< float
        AV_SAMPLE_FMT_DBL,         ///< double

        AV_SAMPLE_FMT_U8P,         ///< unsigned 8 bits, planar
        AV_SAMPLE_FMT_S16P,        ///< signed 16 bits, planar
        AV_SAMPLE_FMT_S32P,        ///< signed 32 bits, planar
        AV_SAMPLE_FMT_FLTP,        ///< float, planar
        AV_SAMPLE_FMT_DBLP,        ///< double, planar

        AV_SAMPLE_FMT_NB           ///< Number of sample formats. DO NOT USE if linking dynamically
    };

    enum AVChannelLayout {
        AV_CH_FRONT_LEFT = 0x00000001,
        AV_CH_FRONT_RIGHT = 0x00000002,
        AV_CH_FRONT_CENTER = 0x00000004,
        AV_CH_LOW_FREQUENCY = 0x00000008,
        AV_CH_BACK_LEFT = 0x00000010,
        AV_CH_BACK_RIGHT = 0x00000020,
        AV_CH_FRONT_LEFT_OF_CENTER = 0x00000040,
        AV_CH_FRONT_RIGHT_OF_CENTER = 0x00000080,
        AV_CH_BACK_CENTER = 0x00000100,
        AV_CH_SIDE_LEFT = 0x00000200,
        AV_CH_SIDE_RIGHT = 0x00000400,
        AV_CH_TOP_CENTER = 0x00000800,
        AV_CH_TOP_FRONT_LEFT = 0x00001000,
        AV_CH_TOP_FRONT_CENTER = 0x00002000,
        AV_CH_TOP_FRONT_RIGHT = 0x00004000,
        AV_CH_TOP_BACK_LEFT = 0x00008000,
        AV_CH_TOP_BACK_CENTER = 0x00010000,
        AV_CH_TOP_BACK_RIGHT = 0x00020000,
        AV_CH_STEREO_LEFT = 0x20000000,
        AV_CH_STEREO_RIGHT = 0x40000000,
        AV_CH_WIDE_LEFT = 0x0000000080000000ULL,
        AV_CH_WIDE_RIGHT = 0x0000000100000000ULL,
        AV_CH_SURROUND_DIRECT_LEFT = 0x0000000200000000ULL,
        AV_CH_SURROUND_DIRECT_RIGHT = 0x0000000400000000ULL,
        AV_CH_LOW_FREQUENCY_2 = 0x0000000800000000ULL,
        AV_CH_LAYOUT_NATIVE = 0x8000000000000000ULL
    };

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

    class AVCodecContextRAII
    {
    public:
        AVCodecContextRAII(AVCodec*);
        ~AVCodecContextRAII();

    public:
        operator AVCodecContext*();
        operator const AVCodecContext*() const;

    public:
        AVCodecContextRAII(const AVCodecContextRAII& ) = delete;
        AVCodecContextRAII& operator=(const AVCodecContextRAII& ) = delete;

    private:
        AVCodecContext* _context;
    };

    class AVFrameRAII
    {
    public:
        AVFrameRAII();
        ~AVFrameRAII();

    public:
        operator AVFrame*();
        operator const AVFrame*() const;

    public:
        AVFrameRAII(const AVFrameRAII& ) = delete;
        AVFrameRAII& operator=(const AVFrameRAII& ) = delete;

    private:
        AVFrame* _frame;
    };

public:
	static void load();
	static bool loaded();

public:
    static size_t getNumberOfSamples(const AVFrame* frame);
    static void* getData(AVFrame* frame);

    static size_t getBytesPerSampleForFormat(const AVContext* context);
    static size_t getSamplingRate(const AVContext* context);
    static size_t getChannelCount(const AVContext* context);
    static size_t getFrameSize(const AVContext* context);
    static enum AVSampleFormat getSampleFormat(const AVContext* context);
    static int64_t getChannelLayout(const AVContext* context);

    static void setBitRate(AVContext* context, size_t bitRate);
    static void setSampleFormat(AVContext* context, enum AVSampleFormat);

    static void setSampleRate(AVCodec* codec, size_t rate);
    static void setChannelLayout(AVCodec* codec, int64_t layout);
    static void setChannelCount(AVCodec* codec, size_t count);

    static void setNumberOfSamples(AVFrame* frame, size_t samples);
    static void setSampleFormat(AVFrame* frame, enum AVSampleFormat);
    static void setChannelLayout(AVFrame* frame, int64_t layout);

public:
    static AVCodec* avcodec_find_decoder_by_name(const char*);
    static AVCodec* avcodec_find_encoder_by_name(const char*);

public:
	static void av_init_packet(AVPacket* packet);
    static AVCodecContext* avcodec_alloc_context3(AVCodec* codec);
    static int avcodec_open2(AVCodecContext* avctx, const AVCodec* codec, AVDictionary** options);
    static AVFrame* av_frame_alloc();
    static int avcodec_decode_audio4(AVCodecContext* avctx, AVFrame* frame,
        int* got_frame_ptr, const AVPacket* avpkt);
    static int av_samples_get_buffer_size(int* linesize, int nb_channels, int nb_samples,
        enum AVSampleFormat sample_fmt, int align);
    static int check_sample_fmt(AVCodec* codec, enum AVSampleFormat sample_fmt);

public:
    static int avcodec_fill_audio_frame(AVFrame* frame, int nb_channels,
        enum AVSampleFormat sample_fmt, const uint8_t* buf, int buf_size, int align);
    static int avcodec_encode_audio2(AVCodecContext* avctx, AVPacket* avpkt,
        const AVFrame* frame, int* got_packet_ptr);

public:
    static int av_opt_get_int(void* obj, const char* name, int search_flags, int64_t* out_val);
    static int av_opt_get(void* obj, const char* name, int search_flags, uint8_t** out_val);
    static int av_opt_get_sample_fmt(void* obj, const char* name, int search_flags,
        enum AVSampleFormat* out_fmt);
    static int av_opt_get_channel_layout(void* obj, const char* name, int search_flags,
        int64_t* ch_layout);

public:
    static int av_opt_set_int(void* obj, const char* name, int64_t val, int search_flags);
    static int av_opt_set_sample_fmt(void *obj, const char *name,
        enum AVSampleFormat fmt, int search_flags);
    static int av_opt_set_channel_layout(void* obj, const char* name, int64_t ch_layout,
        int search_flags);

public:
    static void av_free(void*);
    static int avcodec_close(AVCodecContext* avctx);
    static void av_frame_free(AVFrame** frame);

private:
	static void _check();

private:
	class Interface
	{
    public:
        AVCodec* (*avcodec_find_decoder_by_name)(const char*);
        AVCodec* (*avcodec_find_encoder_by_name)(const char*);

	public:
	    void (*av_init_packet)(AVPacket* packet);

        AVCodecContext* (*avcodec_alloc_context3)(AVCodec* codec);
        int (*avcodec_open2)(AVCodecContext* avctx, const AVCodec* codec, AVDictionary** options);
        AVFrame* (*av_frame_alloc)();
        int (*avcodec_decode_audio4)(AVCodecContext* avctx, AVFrame* frame,
            int* got_frame_ptr, const AVPacket* avpkt);
        int (*av_samples_get_buffer_size)(int* linesize, int nb_channels, int nb_samples,
            enum AVSampleFormat sample
        int (*check_sample_fmt)(AVCodec *codec, enum AVSampleFormat sample_fmt);

    public:
        int (*avcodec_fill_audio_frame)(AVFrame* frame, int nb_channels,
            enum AVSampleFormat sample_fmt, const uint8_t* buf, int buf_size, int align);
        int (*avcodec_encode_audio2)(AVCodecContext* avctx, AVPacket* avpkt,
            const AVFrame* frame, int* got_packet_ptr);

    public:
        int (*av_opt_get_int)(void* obj, const char* name, int search_flags, int64_t* out_val);
        int (*av_opt_get)(void* obj, const char* name, int search_flags, uint8_t** out_val);
        int (*av_opt_get_sample_fmt)(void* obj, const char* name, int search_flags,
            enum AVSampleFormat* out_fmt);
        int (*av_opt_get_channel_layout)(void* obj, const char* name, int search_flags,
            int64_t* ch_layout);

    public:
        int (*av_opt_set_int)(void* obj, const char* name, int64_t val, int search_flags);
        int (*av_opt_set_sample_fmt)(void *obj, const char *name,
            enum AVSampleFormat fmt, int search_flags);
        int (*av_opt_set_channel_layout)(void* obj, const char* name, int64_t ch_layout,
            int search_flags);

    public:
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



