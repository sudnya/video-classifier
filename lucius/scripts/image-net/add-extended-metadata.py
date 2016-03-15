#!/usr/bin/env python

import urllib2
import urllib
import re
import shutil
import os
from optparse import OptionParser
import socket

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

def loadCachedTerms():
    if not os.path.exists('synset_list.txt'):
        return None

    terms = []

    with open('synset_list.txt', 'r') as synsets:
        for line in synsets:
            term = line.strip()

            if len(term) > 0:
                terms.append(term)

    return terms

def cacheTerms(terms):
    with open('synset_list.txt', 'w') as synsets:
        for term in terms:
            synsets.write(term + '\n')

def getAllTerms():
    cachedTerms = loadCachedTerms()

    if cachedTerms != None:
        return cachedTerms

    termListUrl = 'http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list'

    terms = [term for term in downloadTextFromUrl(termListUrl).split('\n') if term != ""]

    cacheTerms(terms)

    return terms

def loadCachedRelationships():
    if not os.path.exists('synset_relationships.txt'):
        return None

    parentChildPairs = []

    with open('synset_relationships.txt', 'r') as synsets:
        for line in synsets:
            parentAndChild = line.split(",")

            if len(parentAndChild) != 2:
                continue

            parentChildPairs.append((parentAndChild[0].strip(), parentAndChild[1].strip()))

    return parentChildPairs

def cacheRelationships(relationships):
    with open('synset_relationships.txt', 'w') as relationshipFile:
        for parent, child in relationships:
            relationshipFile.write(parent + ', ' + child + '\n')

def getIsARelationships():
    cachedRelationships = loadCachedRelationships()

    if cachedRelationships != None:
        return cachedRelationships

    isAListUrl = 'http://www.image-net.org/archive/wordnet.is_a.txt'

    relationships = downloadTextFromUrl(isAListUrl).split('\n')

    parentChildPairs = []

    for relationship in relationships:
        parentAndChild = relationship.split(" ")

        if len(parentAndChild) != 2:
            continue

        parentChildPairs.append((parentAndChild[0].strip(), parentAndChild[1].strip()))

    cacheRelationships(parentChildPairs)

    return parentChildPairs

def getDirectoryForTerm(termName):
    return re.sub('[^\w\-_\. ]', '-', termName.split(',')[0].strip()).replace(' ', '-')

def getNameForTerm(term):
    getWordsUrl = 'http://www.image-net.org/api/text/wordnet.synset.getwords?wnid='

    name = getDirectoryForTerm(downloadTextFromUrl(getWordsUrl + term))

    if len(name) == 0:
        return term

    return name

def getImageUrlsForTerm(term):
    getImageUrl = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='

    possibleUrls = downloadTextFromUrl(getImageUrl + term).split('\n')

    urls = []

    for url in possibleUrls:
        if isUrl(url):
            urls.append(url.strip())

    return urls

def loadMetadata(input_path):
    metaDataPath = input_path

    metadataValues = []

    with open(metaDataPath, 'r') as metadata:
        for line in metadata:
            elements = [element.strip() for element in line.split(',')]

            if len(elements) != 3:
                continue

            path, url, category = elements[0], elements[1], elements[2]

            metadataValues.append((path, url, category))

    return metadataValues

class ClassEntry:
    def __init__(self, i, synid, name):
        self.synid    = synid
        self.name     = name
        self.children = set()
        self.parents  = set()

        if i % 100 == 0:
            print i, self.getName()

    def addChild(self, child):
        self.children.add(child)
        child.parents.add(self)

    def getName(self):
        return self.name

    def getAllParents(self):
        if len(self.parents) > 0:
            parent = list(self.parents)[0]
            return parent.getAllParents() + [parent]
        else:
            return []

def loadCachedTermsAndNames():
    if not os.path.exists('synset_names.txt'):
        return None

    termsAndNames = []

    with open('synset_names.txt', 'r') as synsets:
        for line in synsets:
            termAndName = line.split(",")

            if len(termAndName) != 2:
                continue

            term = termAndName[0].strip()
            name = termAndName[1].strip()

            termsAndNames.append((term, name))

    return termsAndNames

def cacheTermsAndNames(termsAndNames):
    with open('synset_names.txt', 'w') as synsets:
        for term, name in termsAndNames:
            synsets.write(term + ', ' + name + '\n')

def addNamesToTerms(terms):
    termsAndNames = loadCachedTermsAndNames()

    if termsAndNames != None:
        return termsAndNames

    termsAndNames = [(term, getNameForTerm(term)) for term in terms]

    cacheTermsAndNames(termsAndNames)

    return termsAndNames

def getAllClasses():
    print 'Loading terms'
    terms         = getAllTerms()
    termsAndNames = addNamesToTerms(terms)

    classes = {synid : ClassEntry(i, synid, name) for i, (synid, name) in enumerate(termsAndNames)}

    print 'Loading relationships'
    relationships = getIsARelationships()

    for parent, child in relationships:
        if not parent in classes:
            continue
        if not child in classes:
            continue

        parentClass = classes[parent]
        childClass  = classes[child]

        print ' linking ', parentClass.getName(), ' <- ', childClass.getName()

        parentClass.addChild(childClass)

    for synid, entry in classes.iteritems():
        print ' <- '.join([parent.getName() for parent in
            entry.getAllParents()] + [entry.getName()])

    return [entry for synid, entry in classes.iteritems()]

def addExtendedMetadata(metadata, classes):
    classesByName = { entry.getName() : entry for entry in classes }

    extendedMetadata = []

    for path, url, category in metadata:

        if category in classesByName:
            entry   = classesByName[category]
            parents = entry.getAllParents()

            line = [path, url]
            extendedMetadata.append(line + [parent.getName() for parent in parents] + [category])

        else:
            extendedMetadata.append([path, url, category])


    return extendedMetadata

def writeExtendedMetadata(outputPath, extendedMetadata):
    with open(outputPath, 'w') as extendedMetadataFile:
        for line in extendedMetadata:
            extendedMetadataFile.write(', '.join(line) + '\n')


def addExtendedMetadataTerms(options):
    inputPath  = options.input_path
    outputPath = options.output_path

    print 'Loading metadata'
    metadata = loadMetadata(inputPath)

    classes = getAllClasses()

    extendedMetadata = addExtendedMetadata(metadata, classes)

    writeExtendedMetadata(outputPath, extendedMetadata)

def main():
    parser = OptionParser()

    parser.add_option("-i", "--input_path",  default="image-net-data/metadata.csv")
    parser.add_option("-o", "--output_path", default="image-net-data/extended-metadata.csv")

    (options, arguments) = parser.parse_args()

    addExtendedMetadataTerms(options)

main()

