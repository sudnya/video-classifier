###############################################################################
#
# \file    get-popular-tags.py
# \author  Sudnya Diamos <mailsudnya@gmail.com>
# \date    Thursday January 5, 2017
# \brief   A python script to get the most popular tags from labels of free 
#          sound dataset and use those labels for the appropriate files
#
###############################################################################

import os
import argparse
import logging
import json
import re
import operator


logger = logging.getLogger('get-popular-tags')

def getTags(tagList):
    retVal = set()
    retVal = re.split(' |\"|\n', tagList)
    return retVal

def reTag(inputDb, outputDb, countN, blacklist):
    allTags = {}
    outF = open(outputDb, 'w')
    
    if os.path.isfile(inputDb):
        inF = open(inputDb, 'r')
        for line in inF:
            #logger.debug(line)
            t = re.split(",", line)
            fileName = t[0]
            temp = t[-1]
            tags = getTags(temp)
            
            for t in tags:
                if t in blacklist:
                    continue

                if not t in allTags:
                    allTags[t] = []
                allTags[t].append(fileName)

    else:
         logger.error ("Specified input path does not exist on current machine " + str(inputDb))
         raise
    
    logger.info ("Downloaded file " + str(inputDb) + " locally at " + str(outputDb))
    
    alreadyWritten = set()
    topN = {}
    allTags.pop('')
    counter = 0
    logger.info("Consider top " + str(countN) + " most frequent tags")
    for k in sorted(allTags, key=lambda k: len(allTags[k]), reverse=True):
        logger.debug("Adding top tag: " + k + " count " + str(len(allTags[k])))
        if counter < countN:
            topN[k] = allTags[k]
            counter += 1
        else:
            break

    for k in sorted(topN, key=lambda k: len(topN[k]), reverse=False):
        print "Adding top tag: " + k + " count " + str(len(topN[k]))
        val= topN[k]
        for v in val:
            if v not in alreadyWritten:
                writeStr = v.rstrip('\n') + " , \"" + k + "\"\n"
                logger.debug("Writing to file: " + writeStr)
                outF.write(writeStr)
                alreadyWritten.add(v)

#    for key, val in topN.iteritems():
#        print "Adding top tag: " + key + " count " + str(len(topN[key]))
#        for v in val:
#            if v not in alreadyWritten:
#                writeStr = v.rstrip('\n') + " , \"" + key + "\"\n"
#                logger.debug("Writing to file: " + writeStr)
#                outF.write(writeStr)
#                alreadyWritten.add(v)

    inF.close()
    outF.close()


def main():
    parser = argparse.ArgumentParser(description="Data set re-labling tool")
    parser.add_argument("-v", "--verbose",        default = False, action = "store_true")
    parser.add_argument("-i", "--input_file",        default = "~/temp/database.txt")
    parser.add_argument("-o", "--output_file",        default = "/tmp/retagged.txt")
    parser.add_argument("-n", "--topN",        default = "500")
    
    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']
    inputFile   = arguments['input_file']
    outputFile  = arguments['output_file']
    topN        = int(arguments['topN'])
    
    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        blacklist = set()
        blacklist.add("field-recording")
        blacklist.add("noise")
        blacklist.add("bass")
        blacklist.add("loop")
        blacklist.add("sound")
        blacklist.add("horror")
        blacklist.add("beat")
        blacklist.add("effect")
        blacklist.add("percussion")
        blacklist.add("ambience")
        blacklist.add("atmosphere")
        blacklist.add("hit")
        blacklist.add("processed")
        blacklist.add("ambient")
        blacklist.add("game")
        blacklist.add("foley")
        blacklist.add("scary")
        blacklist.add("dark")
        blacklist.add("sci-fi")
        blacklist.add("fx")
        blacklist.add("car") #TODO: would have been nice, but 5/6 samples I tested were terrible
        blacklist.add("movie")
        blacklist.add("cinematic")
        blacklist.add("vocal") #2/4 were perfect samples of people talking, rest all strange :(
        blacklist.add("click")
        blacklist.add("wood")
        blacklist.add("film")
        blacklist.add("stereo")
        blacklist.add("environmental-sounds-research")    
        blacklist.add("machine")
        blacklist.add("electronic")
        blacklist.add("reverb")
        blacklist.add("human")
    
        reTag(inputFile, outputFile, topN, blacklist)

    except ValueError as e:
        logger.error ("Invalid Arguments: " + str(e))
        logger.error (parser.print_help())

    except Exception as e:
        logger.error ("Configuration Failed: " + str(e))
    

if __name__ == '__main__':
    main()
