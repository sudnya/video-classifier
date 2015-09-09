/*	\file   LibavcodecLibrary.h
	\date   August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LibavcodecLibrary class.
*/

#pragma once

// Standard Library Includes
#include <cstdint>
#include <cstddef>
#include <string>

namespace lucius
{

namespace audio
{

class LibavcodecLibrary
{
public:
    class AVCodec;
    class AVClass;

    enum AVMediaType
    {
        AVMEDIA_TYPE_UNKNOWN = -1,  ///< Usually treated as AVMEDIA_TYPE_DATA
        AVMEDIA_TYPE_VIDEO,
        AVMEDIA_TYPE_AUDIO,
        AVMEDIA_TYPE_DATA,          ///< Opaque data information usually continuous
        AVMEDIA_TYPE_SUBTITLE,
        AVMEDIA_TYPE_ATTACHMENT,    ///< Opaque data information usually sparse
        AVMEDIA_TYPE_NB
    };

    class AVFrame;
    class AVDictionary;

    class AVBufferRef;
    class AVPacketSideData;

    enum AVSampleFormat
    {
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

    enum AVPixelFormat {
        AV_PIX_FMT_NONE = -1,
        AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_YUYV422,
        AV_PIX_FMT_RGB24,
        AV_PIX_FMT_BGR24,
        AV_PIX_FMT_YUV422P,
        AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUV410P,
        AV_PIX_FMT_YUV411P,
        AV_PIX_FMT_GRAY8,
        AV_PIX_FMT_MONOWHITE,
        AV_PIX_FMT_MONOBLACK,
        AV_PIX_FMT_PAL8,
        AV_PIX_FMT_YUVJ420P,
        AV_PIX_FMT_YUVJ422P,
        AV_PIX_FMT_YUVJ444P
    };

    static const int AV_NUM_DATA_POINTERS = 8;
    static const int AVSEEK_SIZE = 0x10000;

    enum AVColorSpace
    {
        AVCOL_SPC_RGB         = 0,
        AVCOL_SPC_BT709       = 1,
        AVCOL_SPC_UNSPECIFIED = 2,
        AVCOL_SPC_RESERVED    = 3,
        AVCOL_SPC_FCC         = 4,
        AVCOL_SPC_BT470BG     = 5,
        AVCOL_SPC_SMPTE170M   = 6,
        AVCOL_SPC_SMPTE240M   = 7,
        AVCOL_SPC_YCOCG       = 8,
        AVCOL_SPC_BT2020_NCL  = 9,
        AVCOL_SPC_BT2020_CL   = 10,
        AVCOL_SPC_NB
    };

    enum AVColorRange
    {
        AVCOL_RANGE_UNSPECIFIED = 0,
        AVCOL_RANGE_MPEG        = 1,
        AVCOL_RANGE_JPEG        = 2,
        AVCOL_RANGE_NB
    };

    enum AVFieldOrder
    {
        AV_FIELD_UNKNOWN,
        AV_FIELD_PROGRESSIVE,
        AV_FIELD_TT,
        AV_FIELD_BB,
        AV_FIELD_TB,
        AV_FIELD_BT
    };

    enum AVColorPrimaries
    {
        AVCOL_PRI_RESERVED0   = 0,
        AVCOL_PRI_BT709       = 1,
        AVCOL_PRI_UNSPECIFIED = 2,
        AVCOL_PRI_RESERVED    = 3,
        AVCOL_PRI_BT470M      = 4,

        AVCOL_PRI_BT470BG     = 5,
        AVCOL_PRI_SMPTE170M   = 6,
        AVCOL_PRI_SMPTE240M   = 7,
        AVCOL_PRI_FILM        = 8,
        AVCOL_PRI_BT2020      = 9,
        AVCOL_PRI_NB,
    };

    enum AVChromaLocation
    {
        AVCHROMA_LOC_UNSPECIFIED = 0,
        AVCHROMA_LOC_LEFT        = 1,
        AVCHROMA_LOC_CENTER      = 2,
        AVCHROMA_LOC_TOPLEFT     = 3,
        AVCHROMA_LOC_TOP         = 4,
        AVCHROMA_LOC_BOTTOMLEFT  = 5,
        AVCHROMA_LOC_BOTTOM      = 6,
        AVCHROMA_LOC_NB
    };

    enum AVColorTransferCharacteristic
    {
        AVCOL_TRC_RESERVED0    = 0,
        AVCOL_TRC_BT709        = 1,
        AVCOL_TRC_UNSPECIFIED  = 2,
        AVCOL_TRC_RESERVED     = 3,
        AVCOL_TRC_GAMMA22      = 4,
        AVCOL_TRC_GAMMA28      = 5,
        AVCOL_TRC_SMPTE170M    = 6,
        AVCOL_TRC_SMPTE240M    = 7,
        AVCOL_TRC_LINEAR       = 8,
        AVCOL_TRC_LOG          = 9,
        AVCOL_TRC_LOG_SQRT     = 10,
        AVCOL_TRC_IEC61966_2_4 = 11,
        AVCOL_TRC_BT1361_ECG   = 12,
        AVCOL_TRC_IEC61966_2_1 = 13,
        AVCOL_TRC_BT2020_10    = 14,
        AVCOL_TRC_BT2020_12    = 15,
        AVCOL_TRC_NB,
    };

    class AVRational
    {
    public:
        int num;
        int den;
    };

    class AVCodecContext
    {
    public:
        const AVClass* av_class;
        int log_level_offset;
        enum AVMediaType codec_type;
        const AVCodec* codec;
        char codec_name [32];
        int codec_id;
        unsigned int codec_tag;
        unsigned int stream_codec_tag;
        void* priv_data;

        struct AVCodecInternal *internal;
        void *opaque;

        int bit_rate;
        int bit_rate_tolerance;
        int global_quality;
        int compression_level;
        int flags;
        int flags2;

        uint8_t *extradata;
        int extradata_size;

        AVRational time_base;
        int ticks_per_frame;
        int delay;
        int width, height;
        int coded_width, coded_height;
        int gop_size;
        enum AVPixelFormat pix_fmt;
        int me_method;
        void (*draw_horiz_band)(AVCodecContext *s,
                              const AVFrame *src, int offset[AV_NUM_DATA_POINTERS],
                              int y, int type, int height);

        enum AVPixelFormat (*get_format)(AVCodecContext *s, const enum AVPixelFormat * fmt);
        int max_b_frames;
        float b_quant_factor;
        int rc_strategy;
        int b_frame_strategy;
        float b_quant_offset;
        int has_b_frames;
        int mpeg_quant;
        float i_quant_factor;
        float i_quant_offset;
        float lumi_masking;
        float temporal_cplx_masking;
        float spatial_cplx_masking;
        float p_masking;
        float dark_masking;
        int slice_count;
        int prediction_method;
        int *slice_offset;

        AVRational sample_aspect_ratio;

        int me_cmp;
        int me_sub_cmp;
        int mb_cmp;
        int ildct_cmp;
        int dia_size;
        int last_predictor_count;
        int pre_me;
        int me_pre_cmp;
        int pre_dia_size;
        int me_subpel_quality;
        int dtg_active_format;
        int me_range;
        int intra_quant_bias;
        int inter_quant_bias;
        int slice_flags;
        int xvmc_acceleration;
        int mb_decision;
        uint16_t *intra_matrix;
        uint16_t *inter_matrix;
        int scenechange_threshold;
        int noise_reduction;
        int me_threshold;
        int mb_threshold;
        int intra_dc_precision;
        int skip_top;
        int skip_bottom;
        float border_masking;
        int mb_lmin;
        int mb_lmax;
        int me_penalty_compensation;
        int bidir_refine;
        int brd_scale;
        int keyint_min;
        int refs;
        int chromaoffset;
        int scenechange_factor;
        int mv0_threshold;
        int b_sensitivity;
        enum AVColorPrimaries color_primaries;
        enum AVColorTransferCharacteristic color_trc;
        enum AVColorSpace colorspace;
        enum AVColorRange color_range;
        enum AVChromaLocation chroma_sample_location;

        int slices;

        enum AVFieldOrder field_order;

        int sample_rate; ///< samples per second
        int channels;    ///< number of audio channels
        enum AVSampleFormat sample_fmt;  ///< sample format
        int frame_size;
        int frame_number;
        int	block_align;

        int	cutoff;

        int	request_channels;

        uint64_t channel_layout;
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

    static const int AV_CH_LAYOUT_STEREO = (AV_CH_FRONT_LEFT|AV_CH_FRONT_RIGHT);
    static const int AV_CH_LAYOUT_MONO   = (AV_CH_FRONT_CENTER);

    static const int AUDIO_INBUF_SIZE = 20480;
    static const int AV_INPUT_BUFFER_PADDING_SIZE = 32;
    static const int AUDIO_REFILL_THRESH = 4096;
    static const int64_t AV_NOPTS_VALUE = 0x8000000000000000ULL;
    static const int AV_OPT_SEARCH_CHILDREN = 0x0001;

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
        AVPacketSideData* side_data;
        int side_data_elems;

        /**
         * Duration of this packet in AVStream->time_base units, 0 if unknown.
         * Equals next_pts - this_pts in presentation order.
         */
        int duration;

        uint64_t who_cares[16];
    } AVPacket;

    class AVCodec
    {
    public:
        const char* name;
        const char* long_name;
        enum AVMediaType type;
        int id;
        int capabilities;
        const AVRational* supported_framerates;
        const int* pix_fmts;
        const int* supported_samplerates;
        const enum AVSampleFormat* sample_fmts;
        const uint64_t* channel_layouts;
    };

    class AVFrame
    {
    public:
        uint8_t *data[AV_NUM_DATA_POINTERS];
        int linesize[AV_NUM_DATA_POINTERS];
        uint8_t **extended_data;
        int width, height;
        int nb_samples;
        int format;
    };

    class AVInputFormat;
    class AVOutputFormat;

    class AVIOContext
    {
    public:
        const AVClass* av_class;
        unsigned char* buffer;
    };

    class AVFormatContext
    {
    public:
        const AVClass* av_class;
        AVInputFormat* iformat;
        AVOutputFormat* oformat;
        void* priv_data;
        AVIOContext* pb;
    };

    class AVStream
    {
    public:
        int index;
        int id;
        AVCodecContext* codec;
    };

    class AVCodecContextRAII
    {
    public:
        AVCodecContextRAII(AVCodec*);
        ~AVCodecContextRAII();

    public:
        operator AVCodecContext*();
        operator const AVCodecContext*() const;

    public:
        AVCodecContext* operator->();

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

    class AVPacketRAII
    {
    public:
        AVPacketRAII();
        ~AVPacketRAII();

    public:
        operator AVPacket*();
        operator const AVPacket*() const;

    public:
        AVPacket* operator->();
        const AVPacket* operator->() const;

    public:
        AVPacketRAII(const AVPacketRAII& ) = delete;
        AVPacketRAII& operator=(const AVPacketRAII& ) = delete;

    private:
        AVPacket _packet;
    };

public:
	static void load();
	static bool loaded();

public:
    static std::string getErrorCode(int32_t status);

public:
    static size_t getNumberOfSamples(const AVFrame* frame);
    static void* getData(AVFrame* frame);

    static size_t getBytesPerSampleForFormat(const AVCodecContext* context);
    static enum AVSampleFormat getSampleFormatWithBytes(size_t bytes);
    static size_t getSamplingRate(const AVCodecContext* context);
    static size_t getChannelCount(const AVCodecContext* context);
    static size_t getFrameSize(const AVCodecContext* context);
    static enum AVSampleFormat getSampleFormat(const AVCodecContext* context);
    static int64_t getChannelLayout(const AVCodecContext* context);

    static void setBitRate(AVCodecContext* context, size_t bitRate);
    static void setSampleFormat(AVCodecContext* context, enum AVSampleFormat);
    static void setSampleRate(AVCodecContext* context, size_t sampleRate);
    static void setChannelLayout(AVCodecContext* context, int64_t layout);
    static void setChannelCount(AVCodecContext* context, size_t count);

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
    static int av_opt_set(void* obj, const char* name, const char* val, int search_flags);
    static int av_opt_set_int(void* obj, const char* name, int64_t val, int search_flags);
    static int av_opt_set_sample_fmt(void *obj, const char *name,
        enum AVSampleFormat fmt, int search_flags);
    static int av_opt_set_channel_layout(void* obj, const char* name, int64_t ch_layout,
        int search_flags);

public:
    static int av_opt_show2(void* obj, void* av_log_obj, int req_flags, int rej_flags);

public:
    static void av_frame_set_channel_layout(AVFrame *frame, int64_t val);

public:
    static int av_get_channel_layout_nb_channels(uint64_t layout);

public:
    static AVIOContext* avio_alloc_context(unsigned char* buffer,
        int buffer_size,
        int write_flag,
        void* opaque,
        int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
        int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
        int64_t(*seek)(void* opaque, int64_t offset, int whence));

public:
    static AVFormatContext* avformat_alloc_context();
    static void avformat_free_context(AVFormatContext* );

public:
    static int avformat_open_input(AVFormatContext** ps, const char* filename,
        AVInputFormat* fmt, AVDictionary** options);

public:
    static AVStream* avformat_new_stream(AVFormatContext* s, const AVCodec* c);
    static int avformat_write_header(AVFormatContext* s, AVDictionary** options);
    static int av_write_trailer(AVFormatContext* s);

public:
    static int av_write_frame(AVFormatContext* s, AVPacket* pkt);
    static int av_read_frame(AVFormatContext* s, AVPacket* pkt);

public:
    static AVOutputFormat* av_guess_format(const char* short_name, const char* filename,
        const char* mime_type);

public:
    static int avcodec_copy_context(AVCodecContext* dest, const AVCodecContext* src);

public:
    static void* av_malloc(size_t size);

public:
    static void av_free(void*);
    static void av_free_packet(AVPacket* packet);
    static int avcodec_close(AVCodecContext* avctx);
    static void av_frame_free(AVFrame** frame);

private:
	static void _check();

private:
	class Interface
	{
    public:
        void (*avcodec_register_all)();
        void (*av_register_all)();

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
            enum AVSampleFormat sample_fmt, int align);

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
        int (*av_opt_set)(void* obj, const char* name, const char* val, int search_flags);
        int (*av_opt_set_int)(void* obj, const char* name, int64_t val, int search_flags);
        int (*av_opt_set_sample_fmt)(void *obj, const char *name,
            enum AVSampleFormat fmt, int search_flags);
        int (*av_opt_set_channel_layout)(void* obj, const char* name, int64_t ch_layout,
            int search_flags);

    public:
        int (*av_opt_show2)(void* obj, void* av_log_obj, int req_flags, int rej_flags);

    public:
        void (*av_frame_set_channel_layout)(AVFrame *frame, int64_t val);

    public:
        int (*av_get_channel_layout_nb_channels)(uint64_t layout);

    public:
        AVIOContext* (*avio_alloc_context)(unsigned char* buffer,
            int buffer_size,
            int write_flag,
            void* opaque,
            int(*read_packet)(void* opaque, uint8_t* buf, int buf_size),
            int(*write_packet)(void* opaque, uint8_t* buf, int buf_size),
            int64_t(*seek)(void* opaque, int64_t offset, int whence));

    public:
        AVFormatContext* (*avformat_alloc_context)();
        void (*avformat_free_context)(AVFormatContext* );

    public:
        int (*avformat_open_input)(AVFormatContext** ps, const char* filename,
            AVInputFormat* fmt, AVDictionary** options);

    public:
        AVStream* (*avformat_new_stream)(AVFormatContext* s, const AVCodec* c);
        int (*avformat_write_header)(AVFormatContext* s, AVDictionary** options);
        int (*av_write_trailer)(AVFormatContext* s);

    public:
        int (*av_write_frame)(AVFormatContext* s, AVPacket* pkt);
        int (*av_read_frame)(AVFormatContext* s, AVPacket* pkt);

    public:
        AVOutputFormat* (*av_guess_format)(const char* short_name, const char* filename,
            const char* mime_type);

    public:
        int (*avcodec_copy_context)(AVCodecContext* dest, const AVCodecContext* src);

    public:
        void* (*av_malloc)(size_t size);

    public:
        void (*av_free)(void*);
        void (*av_free_packet)(AVPacket* packet);
        int  (*avcodec_close)(AVCodecContext* avctx);
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



