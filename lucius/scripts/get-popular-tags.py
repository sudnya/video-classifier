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

def reTag(inputDb, outputDb, countN):
    allTags = {}
    outF = open(outputDb, 'w')
    
    if os.path.isfile(inputDb):
        inF = open(inputDb, 'r')
        for line in inF:
            t = re.split(",", line)
            fileName = t[0]
            temp = t[-1]
            tags = getTags(temp)
            for t in tags:
                if not t in allTags:
                    allTags[t] = []
                allTags[t].append(fileName)

    else:
         logger.error ("Specified input path does not exist on current machine " + str(inputDb))
         raise
    
    logger.info ("Downloaded file " + str(inputDb) + " locally at " + str(outputDb))
    
    topN = {}
    allTags.pop('')
    counter = 0
    logger.info("Consider top " + countN + " most frequent tags")
    for k in sorted(allTags, key=lambda k: len(allTags[k]), reverse=True):
        logger.debug("Adding top tag: " + k)
        if counter < countN:
            topN[k] = allTags[k]
            counter += 1
        else:
            break
   
    for key, val in topN.iteritems():
        for v in val:
            writeStr = v.rstrip('\n') + " , \"" + key + "\"\n"
            logger.debug("Writing to file: " + writeStr)
            outF.write(writeStr)

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
    topN        = arguments['topN']
    
    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        reTag(inputFile, outputFile, topN)

    except ValueError as e:
        logger.error ("Invalid Arguments: " + str(e))
        logger.error (parser.print_help())

    except Exception as e:
        logger.error ("Configuration Failed: " + str(e))
    

if __name__ == '__main__':
    main()
