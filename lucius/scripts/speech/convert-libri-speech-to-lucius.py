
import sys
import os
import shutil
from argparse import ArgumentParser

def isTranscriptFile(filename):
    head, tail = os.path.splitext(filename)

    if tail != '.txt':
        return False

    head, tail = os.path.splitext(head)

    return tail == '.trans'

def isAudioFile(filename):
    head, tail = os.path.splitext(filename)

    return tail == '.flac'

def hasTranscriptFile(files):
    for filename in files:
        if isTranscriptFile(filename):
            return True

    return False

def getSpeakerId(root):
    base, chapter = os.path.split(root)
    base, speakerId = os.path.split(base)

    return speakerId

def createDirectories(path):
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)

def copyFile(output, input):
    print "Copying '" + input + "' -> '" + output + "'"
    shutil.copyfile(input, output)

def copyFiles(outputPath, speakerId, root, files):
    for filename in files:
        if not isAudioFile(filename):
            continue

        inputFilePath = os.path.join(root, filename)
        outputFilePath = os.path.join(outputPath, 'speech', speakerId, filename)

        createDirectories(outputFilePath)

        copyFile(outputFilePath, inputFilePath)


def clearPath(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.exists(path):
        shutil.rmtree(path)

def getTranscript(files):
    for file in files:
        if isTranscriptFile(file):
            return file

    return None

def getAudioId(filename):
    head, tail = os.path.splitext(filename)

    return head

def getAudioIdFromLabel(label):
    splitLabel = label.split(' ')

    return splitLabel[0]

def getAudioLabel(label):
    splitLabel = label.split(' ')

    return " ".join(splitLabel[1:]).strip('\n')

def getLabel(root, files, filename):
    transcript = getTranscript(files)

    assert transcript != None

    transcriptPath = os.path.join(root, transcript)

    audioId = getAudioId(filename)

    label = None
    with open(transcriptPath, 'r') as transcriptFile:
        for line in transcriptFile:
            labelAudioId = getAudioIdFromLabel(line)

            if audioId == labelAudioId:
                label = getAudioLabel(line)
                break


    assert label != None, "Could not file label for " + audioId

    return label

def updateMetadata(outputPath, speakerId, root, files):
    metadataPath = os.path.join(outputPath, 'database.txt')

    createDirectories(metadataPath)

    with open(metadataPath, 'a') as metadataFile:
        for filename in files:
            if not isAudioFile(filename):
                continue

            label = getLabel(root, files, filename)

            print "Adding label '" + filename + "' -> '" + label + "'"

            filePath = os.path.join(outputPath, 'speech', speakerId, filename)
            metadataFile.write(filePath + ", " + label + "\n")

def convertLibriSpeech(options):
    inputPath  = options["input_path"]
    outputPath = options["output_path"]

    if options["clear_output"]:
        clearPath(outputPath)

    for root, directories, files in os.walk(inputPath):
        if not hasTranscriptFile(files):
            continue

        speakerId = getSpeakerId(root)

        copyFiles(outputPath, speakerId, root, files)
        updateMetadata(outputPath, speakerId, root, files)


def main():
    parser = ArgumentParser()

    parser.add_argument("-i", "--input-path", default="", help="Input path to libri speech dataset.")
    parser.add_argument("-o", "--output-path", default="", help="Output path to lucius dataset.")
    parser.add_argument("-c", "--clear-output", default=False, action='store_true',
        help="Clear the output path before running.")

    arguments = parser.parse_args()

    convertLibriSpeech(vars(arguments))

if __name__ == "__main__":
    sys.exit(main())

