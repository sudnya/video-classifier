
#!/usr/bin/env python

import urllib2
import urllib
import re
import shutil
import os
from optparse import OptionParser
import socket
import imghdr

import urlparse

def isUrl(url):
    return urlparse.urlparse(url).scheme != ""

def downloadTextFromUrl(url):

    try:
        text = urllib2.urlopen(url).read()
    except:
        return ""

    return text

def getAllTerms():
    termListUrl = 'http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list'

    terms = [term for term in downloadTextFromUrl(termListUrl).split('\n') if term != ""]

    return terms

def getIsARelationships():
    isAListUrl = 'http://www.image-net.org/archive/wordnet.is_a.txt'

    relationships = downloadTextFromUrl(isAListUrl).split('\n')

    parentChildPairs = []

    for relationship in relationships:
        parentAndChild = relationship.split(" ")

        if len(parentAndChild) != 2:
            continue

        parentChildPairs.append((parentAndChild[0].strip(), parentAndChild[1].strip()))

    return parentChildPairs

def getNameForTerm(term):
    getWordsUrl = 'http://www.image-net.org/api/text/wordnet.synset.getwords?wnid='

    return downloadTextFromUrl(getWordsUrl + term)

def getImageUrlsForTerm(term):
    getImageUrl = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='

    possibleUrls = downloadTextFromUrl(getImageUrl + term).split('\n')

    urls = []

    for url in possibleUrls:
        if isUrl(url):
            urls.append(url)

    return urls

def isValidTerm(term):
    return len(getImageUrlsForTerm(term)) > 0

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

def isValidImage(path):
    if imghdr.what(path) == None:
        return False

    if os.path.getsize(path) < 3000:
        return False

    return True

def downloadImage(path, url):

    createDirectories(path)

    print "Downloading ", str(url)
    print " to ", path

    if os.path.exists(path):
        if isValidImage(path):
            print " Skipped"
            return False

    try:
        urllib.urlretrieve(url, path)

        if not isValidImage(path):
            print " Download succeeded, but file is not an image"
            cleanDestination(path)
            return False

        print " Success"
        return True
    except:
        print " Failed"
        cleanDestination(path)
        return False

def downloadImagesForTerms(selectedTermIds, options):

    destinationDirectory = options.output_path
    imageCount = 0

    for term in selectedTermIds:
        print 'For term: \'' + term + "\'"
        termName = getNameForTerm(term)

        print ' Downloading all images that match the term: \'' + getDirectoryForTerm(termName) + "\'"

        imageUrls = getImageUrlsForTerm(term)

        remainingImages = int(options.maximum_images) - imageCount

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

                if imageId >= int(options.maximum_images_per_term):
                    break

        if imageCount >= int(options.maximum_images):
            break

def fillInChildCounts(term, classes):
    if classes[term][2] != -1:
        return classes[term][2]

    if len(classes[term][1]) == 0:
        classes[term] = (classes[term][0], classes[term][1], 1)
        return 1

    count = 0

    for child in classes[term][1]:
        count += fillInChildCounts(child, classes)

    classes[term] = (classes[term][0], classes[term][1], count)

    return count

def selectTerms(selectedCount):
    allTerms = getAllTerms()
    isARelationships = getIsARelationships()

    classes = {}

    for term in allTerms:
        if not term in classes:
            classes[term] = (set([]), set([]), -1)

    for parent, child in isARelationships:

        if not child in classes:
            classes[child] = (set([]), set([]), -1)

        if not parent in classes:
            classes[parent] = (set([]), set([]), -1)

        classes[child][0].add(parent)
        classes[parent][1].add(child)

    for parent in classes.keys():
        fillInChildCounts(parent, classes)

    childCountsAndTerms = []

    for term, stats in classes.iteritems():
        childCountsAndTerms.append((stats[2], term))

    sortedTerms = [term for childCount, term in sorted(childCountsAndTerms, reverse=True)]

    selected = set([])
    frontier = []

    selectedCount = min(selectedCount, len(allTerms))

    for rootTerm in sortedTerms:
        if len(selected) >= selectedCount:
            break

        if not isValidTerm(rootTerm):
            continue

        selected.add(rootTerm)

        for parent in classes[rootTerm][0]:
            selected.discard(parent)

    return selected

def main():
    socket.setdefaulttimeout(10.0)

    parser = OptionParser()

    parser.add_option("-o", "--output_path", default="image-net-data/")
    parser.add_option("-I", "--maximum_images", default=1000000000)
    parser.add_option("-t", "--terms", default=10)
    parser.add_option("-c", "--clean", default=False, action="store_true")
    parser.add_option("-i", "--maximum_images_per_term", default=100)

    (options, arguments) = parser.parse_args()

    destinationDirectory = options.output_path

    if options.clean:
        cleanDestination(destinationDirectory)

    selectedTermIds = selectTerms(int(options.terms))

    print 'Selected', str(len(selectedTermIds)), 'terms, each with', options.maximum_images_per_term, 'images'

    downloadImagesForTerms(selectedTermIds, options)


main()


