
#!/usr/bin/env python

import urllib2
import urllib
import re
import shutil
import os
from optparse import OptionParser
import socket
import imghdr
import Queue
from threading import Thread

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

def getImageUrlsForTermAndSubclassesRecurse(urls, term, classes, remainingImages):
    urls.extend(getImageUrlsForTerm(term))

    for child in classes[term][1]:
        if len(urls) >= remainingImages:
            break
        getImageUrlsForTermAndSubclassesRecurse(urls, child, classes, remainingImages)

def getImageUrlsForTermAndSubclasses(term, classes, remainingImages):
    urls = []

    getImageUrlsForTermAndSubclassesRecurse(urls, term, classes, remainingImages)

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

class IgnorePasswordURLOpener(urllib.FancyURLopener):
    def __init__():
        super(IgnorePasswordURLOpener, self).__init__()

    def prompt_user_passwd(self, host, realm):
        return ("username", "password")


def downloadImage(path, url):

    createDirectories(path)

    print "Downloading ", str(url)
    print " to ", path

    if os.path.exists(path):
        if isValidImage(path):
            print " Skipped"
            return False

    try:
        IgnorePasswordURLOpener urlopener;

        urlopener.retrieve(url, path)

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

class Downloader(Thread):
    def __init__(self, file_url, save_path):
        super(Downloader, self).__init__()

        self.file_url = file_url
        self.save_path = save_path
        self.result = False

    def run(self):
        self.result = downloadImage(self.save_path, self.file_url)

def updateSampleDatabase(destinationDirectory, term):
    path = os.path.join(destinationDirectory, "database.txt")

    createDirectories(path)

    databaseFile = open(path, 'a')

    databaseFile.write(getDirectoryForTerm(term) + ", " + getDirectoryForTerm(term) + "\n")


def downloadImagesForTerms(selectedTermIds, classes, options):

    destinationDirectory = options.output_path
    imageCount = 0

    for term in selectedTermIds:

        updateSampleDatabase(desintationDirectory, term)

        print 'For term: \'' + term + "\'"
        termName = getNameForTerm(term)

        print ' Downloading all images that match the term: \'' + getDirectoryForTerm(termName) + "\'"

        remainingImages = int(options.maximum_images) - imageCount

        imageUrls = getImageUrlsForTermAndSubclasses(term, classes, remainingImages)

        if len(imageUrls) > remainingImages:
            imageUrls = imageUrls[:remainingImages]

        print '  found ' + str(len(imageUrls)) + " images"

        imageId = 0
        threadCount = 100
        timeout = 3.0

        for index in range(0, len(imageUrls), threadCount):

            threads = []

            for threadId in range(threadCount):
                i = index + threadId

                if i >= len(imageUrls):
                    continue

                imageUrl = imageUrls[i]

                target = formDirectoryName(destinationDirectory, getDirectoryForTerm(termName), i) + '.tmp'

                threads.append(Downloader(imageUrl, target))
                threads[-1].start()

            for thread in threads:
                thread.join(timeout)

                if not thread.result:
                    continue

                target = formDirectoryName(destinationDirectory, getDirectoryForTerm(termName), imageId)

                os.rename(thread.save_path, target)

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
    print 'Downloading all terms'
    allTerms = getAllTerms()
    print ' got ' + str(len(allTerms))

    print 'Downloading all relationships'
    isARelationships = getIsARelationships()
    print ' got ' + str(len(isARelationships))

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

    return selected, classes

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

    selectedTermIds, classes = selectTerms(int(options.terms))

    print 'Selected', str(len(selectedTermIds)), 'terms, each with', options.maximum_images_per_term, 'images'

    downloadImagesForTerms(selectedTermIds, classes, options)


main()


