
#!/usr/bin/env python

import urllib2
import urllib
import re
import shutil
import os
from optparse import OptionParser
import socket
import imghdr

def downloadTextFromUrl(url):

    try:
        text = urllib2.urlopen(url).read()
    except:
        return ""

    return text

def getAllTerms():
    termListUrl = 'http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list'

    terms = downloadTextFromUrl(termListUrl).split('\n')

    return terms

def getNameForTerm(term):
    getWordsUrl = 'http://www.image-net.org/api/text/wordnet.synset.getwords?wnid='

    return downloadTextFromUrl(getWordsUrl + term)

def getImageUrlsForTerm(term):
    getImageUrl = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='

    urls = downloadTextFromUrl(getImageUrl + term).split('\n')

    return urls

def getDirectoryForTerm(termName):
    return re.sub('[^\w\-_\. ]', '-', termName.split(',')[0].strip()).replace(' ', '-')

def formDirectoryName(base, termName, imageId):
    return os.path.join(base, getDirectoryForTerm(termName), str(imageId) + ".jpg")

def writeMetadata(path, termName, url, imageId):
    metadataFile = open(path, 'a')

    metadataFile.write(formDirectoryName('.', termName, imageId) + ", " + url.strip() + ", " + getDirectoryForTerm(termName) + "\n")

def createDirectories(path):
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)

def updateMetadata(destinationDirectory, termName, url, imageId):
    metadataPath = os.path.join(destinationDirectory, "metadata.csv")

    createDirectories(metadataPath)

    writeMetadata(metadataPath, termName, url, imageId)

def cleanDestination(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.exists(path):
        shutil.rmtree(path)

def downloadImage(path, url):

    createDirectories(path)

    print "Downloading ", str(url)
    print " to ", path

    if os.path.exists(path):
        print " Skipped"
        return False

    try:
        urllib.urlretrieve(url, path)
        print " Success"
        return True
    except:
        print " Failed"
        cleanDestination(path)
        return False

def main():
    socket.setdefaulttimeout(10.0)

    parser = OptionParser()

    parser.add_option("-o", "--output_path", default="image-net-data/")
    parser.add_option("-I", "--maximum_images", default=1000000000)
    parser.add_option("", "--maximum_images_per_term", default=10)

    (options, arguments) = parser.parse_args()

    destinationDirectory = options.output_path

    #cleanDestination(destinationDirectory)

    allTermIds = getAllTerms()

    print 'Downloaded ' + str(len(allTermIds)) + ' terms'

    imageCount = 0

    for term in allTermIds:
        termName = getNameForTerm(term)

        print ' Downloading all images that match the term: \'' + getDirectoryForTerm(termName) + "\'"

        imageUrls = getImageUrlsForTerm(term)

        remainingImages = options.maximum_images - imageCount

        if len(imageUrls) > remainingImages:
            imageUrls = imageUrls[:remainingImages]

        print '  found ' + str(len(imageUrls)) + " images"

        imageId = 0

        for imageUrl in imageUrls:
            target = formDirectoryName(destinationDirectory, getDirectoryForTerm(termName), imageId)

            if downloadImage(target, imageUrl):
                updateMetadata(destinationDirectory, termName, imageUrl, imageId)

            imageId += 1
            imageCount += 1

            if imageId >= options.maximum_images_per_term:
                break

        if imageCount >= options.maximum_images:
            break


main()


