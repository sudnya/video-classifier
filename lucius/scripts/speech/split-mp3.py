
from pydub import AudioSegment

from optparse import OptionParser
import sys

def splitAudio(inputFile, outputPrefix, segmentLength):

    assert outputPrefix != ""
    assert inputFile != ""

    segmentLength *= 1000

    segment = AudioSegment.from_file(inputFile, "mp3")

    length = len(segment)

    segments = (length + segmentLength - 1) / segmentLength

    for i in range(segments):
        begin = i * segmentLength
        end   = min(length, (i+1) * segmentLength)

        segmentSlice = segment[begin:end]

        print "Writing slice " + outputPrefix + str(i) + ".mp3"

        segmentSlice.export(outputPrefix + str(i) + ".mp3", "mp3")


def main():
    parser = OptionParser()

    parser.add_option("-i", "--input-file", default="")
    parser.add_option("-o", "--output-prefix", default="")
    parser.add_option("-l", "--segment-length", default=60)

    (options, arguments) = parser.parse_args()

    splitAudio(options.input_file, options.output_prefix, options.segment_length)



if __name__ == "__main__":
    sys.exit(main())




