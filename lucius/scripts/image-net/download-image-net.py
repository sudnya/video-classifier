
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
import random

import urlparse

def isUrl(url):
    return urlparse.urlparse(url).scheme != ""

def downloadTextFromUrl(url):

    try:
        text = urllib2.urlopen(url).read()
    except Exception as exception:
	print exception
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
            urls.append(url.strip())

    return urls

def getApproximateSize(term):
    getImageUrl = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='

    try:
        page = urllib2.urlopen(getImageUrl + term)

        if 'Content-Length' in page.info():
            return int(page.info()['Content-Length'])
    except:
        pass

    return 0



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

def getDirectoryForTerm(termName):
    return re.sub('[^\w\-_\. ]', '-', termName.split(',')[0].strip()).replace(' ', '-')

def formImagePath(base, termName, imageId):
    return os.path.join(base, getDirectoryForTerm(termName), str(imageId) + ".jpg")

def formDirectoryName(base, termName, imageId):
    path = formImagePath(base, termName, imageId)

    while os.path.exists(path):
        imageId += 1
        path = formImagePath(base, termName, imageId)

    return path, imageId

def writeMetadata(path, termName, url, imageId):
    metadataFile = open(path, 'a')

    metadataFile.write(formImagePath('.', termName, imageId) + ", " + url.strip() + ", " +
        getDirectoryForTerm(termName) + "\n")

def createDirectories(path):
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)

def updateMetadata(destinationDirectory, termName, url, imageId):
    metadataPath = os.path.join(destinationDirectory, "metadata.csv")

    createDirectories(metadataPath)

    writeMetadata(metadataPath, termName, url, imageId)

def updateFailedList(destinationDirectory, url):
    failedListPath = os.path.join(destinationDirectory, "failed.csv")

    createDirectories(failedListPath)

    failedList = open(failedListPath, 'a')

    failedList.write(url + '\n')

    print 'updating failed list with ' + url


def cleanDestination(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.exists(path):
        shutil.rmtree(path)

def isValidImage(path):
    if imghdr.what(path) == None:
        return False

    return True

class IgnorePasswordURLOpener(urllib.FancyURLopener):
    def __init__(self):
        urllib.FancyURLopener.__init__(self)

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
        urlopener = IgnorePasswordURLOpener()

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

class UrlRequest:
    def __init__(self, destinationDirectory, term, url, i):
        self.index = i
        self.save_path, self.index = formDirectoryName(destinationDirectory, getDirectoryForTerm(term), i)
        self.file_url = url
        self.result = False

class Downloader(Thread):
    def __init__(self, queue):
        super(Downloader, self).__init__()

        self.queue = queue

    def run(self):
        while True:
            try:
                request = self.queue.get_nowait()

            except Queue.Empty:
                break

            try:
                request.result = downloadImage(request.save_path, request.file_url)
            except:
                pass

            self.queue.task_done()

def updateSampleDatabase(destinationDirectory, term):
    path = os.path.join(destinationDirectory, "database.txt")

    createDirectories(path)

    databaseFile = open(path, 'a')

    databaseFile.write(getDirectoryForTerm(term) + ", " + getDirectoryForTerm(term) + "\n")

def getAlreadyDownloadedFilesFromMetadata(destinationDirectory):
    existing = set()

    metadataPath = os.path.join(destinationDirectory, "metadata.csv")

    if not os.path.exists(metadataPath):
        return existing

    with open(metadataPath, 'r') as metadata:
        for line in metadata:
            elements = [element.strip() for element in line.split(',')]

            if len(elements) != 3:
                continue

            path, url, category = elements[0], elements[1], elements[2]

            existing.add(url)

    return existing

def getAlreadyFailedFiles(destinationDirectory):
    existing = set()

    failedListPath = os.path.join(destinationDirectory, "failed.csv")

    if not os.path.exists(failedListPath):
        return existing

    with open(failedListPath, 'r') as failedList:
        for line in failedList:
            url = line.strip()

            if len(url) == 0:
                continue

            existing.add(url)

    return existing


def downloadImagesForTerms(selectedTermIds, classes, options):

    destinationDirectory = options.output_path
    imageCount = 0

    downloadedAlready = getAlreadyDownloadedFilesFromMetadata(destinationDirectory)
    failedAlready = getAlreadyFailedFiles(destinationDirectory)

    print 'Already downloaded ' + str(len(downloadedAlready))
    print 'Already failed ' + str(len(failedAlready))

    random.seed()
    random.shuffle(selectedTermIds)

    for term in selectedTermIds:

        updateSampleDatabase(destinationDirectory, term)

        print 'For term: \'' + term + "\'"
        termName = getNameForTerm(term)

        if len(termName) == 0:
            print ' Skipping invalid term'
            continue

        print ' Downloading all images that match the term: \'' + getDirectoryForTerm(termName) + "\'"

        remainingImages = int(options.maximum_images) - imageCount

        possibleUrls = getImageUrlsForTermAndSubclasses(term, classes, remainingImages)

        urls = []

        for url in possibleUrls:
            if url in downloadedAlready:
               continue

            if url in failedAlready and not options.retry_failed:
                continue

            urls.append(url)
            downloadedAlready.add(url)

        imageUrls = []
        index = 0

        for url in urls:
            request = UrlRequest(destinationDirectory, termName, url, index)
            index = request.index + 1
            imageUrls.append(request)

        if len(imageUrls) > remainingImages:
            imageUrls = imageUrls[:remainingImages]

        print '  found ' + str(len(imageUrls)) + " images (skipped " + \
            str(len(possibleUrls) - len(urls)) + " already downloaded or failed)"

        threadCount = 3

        queue = Queue.Queue()

        for request in imageUrls:
            queue.put(request)

        for threadId in range(threadCount):
            thread = Downloader(queue)

            thread.setDaemon(True)
            thread.start()

        queue.join()

        urllib.urlcleanup()

        successes = 0

        for request in imageUrls:
            if not request.result:
                updateFailedList(destinationDirectory, request.file_url)
                continue

            successes += 1

            if successes >= int(options.maximum_images_per_term):
                cleanDestination(request.save_path)
                continue

            updateMetadata(destinationDirectory, termName, request.file_url, request.index)

            imageCount += 1

        if imageCount >= int(options.maximum_images):
            break

def fillInChildCounts(term, classes):
    if classes[term][2] != -1:
        return classes[term][2]

    if len(classes[term][1]) == 0:
        classes[term] = (classes[term][0], classes[term][1], 1)
        return getApproximateSize(term)

    count = 0

    for child in classes[term][1]:
        count += fillInChildCounts(child, classes)

    classes[term] = (classes[term][0], classes[term][1], count)

    return count

def selectAllTerms():
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

    return allTerms, classes

def selectTerms(selectedCount):
    allTerms, classes = selectAllTerms()

    for parent in classes.keys():
        fillInChildImageCounts(parent, classes)

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

        print 'Selected ', rootTerm, ' with ', classes[rootTerm][2], ' total images'
        selected.add(rootTerm)

        for parent in classes[rootTerm][0]:
            if parent in selected:
                print ' discarded ', parent
                selected.discard(parent)

    return selected, classes

def main():
    parser = OptionParser()

    parser.add_option("-o", "--output_path", default="image-net-data/")
    parser.add_option("-I", "--maximum_images", default=1000000000)
    parser.add_option("-t", "--terms", default=1000000)
    parser.add_option("-c", "--clean", default=False, action="store_true")
    parser.add_option("-a", "--all_terms", default=False, action="store_true")
    parser.add_option("-i", "--maximum_images_per_term", default=10000000)
    parser.add_option("-r", "--retry_failed", default=False, action="store_true")

    (options, arguments) = parser.parse_args()

    destinationDirectory = options.output_path

    if options.clean:
        cleanDestination(destinationDirectory)

    if options.all_terms:
        selectedTermIds, classes = selectAllTerms()
    else:
        selectedTermIds, classes = selectTerms(int(options.terms))

    print 'Selected', str(len(selectedTermIds)), 'terms, each with', \
        options.maximum_images_per_term, 'images'

    socket.setdefaulttimeout(10.0)

    downloadImagesForTerms(selectedTermIds, classes, options)


main()


