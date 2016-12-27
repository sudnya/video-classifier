###############################################################################
#
# \file    getVocabulary.py
# \author  Sudnya Diamos <mailsudnya@gmail.com>
# \date    Monday September 5, 2016
# \brief   A python script to count the unique characters from a given text
#          dataset
#
###############################################################################

import os
import argparse
import logging
import json
import codecs

logger = logging.getLogger('getVocabulary')


def getVocab(dirPath):
    retVal = set()
    if os.path.isdir(dirPath):
        for filename in os.listdir(dirPath):
            fName = os.path.join(dirPath, filename)
            logger.info("Reading file: " + fName)
            with codecs.open(fName, encoding='utf-8') as f:
                for line in f:
                    for c in line:
                        retVal.add(c.encode('ascii','ignore'))
    else:
        logger.error ("Specified input path directory does not exist on current machine " + str(dirPath))

    return retVal

def main():
    parser = argparse.ArgumentParser(description="Vocabulary counting for text dataset tool")
    parser.add_argument("-i", "--input",   default = "/Users/dagny/Downloads/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true")
    
    parsedArguments = parser.parse_args()
    arguments       = vars(parsedArguments)
    isVerbose       = arguments['verbose']
    fileDir         = arguments['input']
    
    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        vocabSet = getVocab(fileDir)
        logger.info("vocabulary: " + str(vocabSet))

    except ValueError as e:
        logger.error ("Invalid Arguments: " + str(e))
        logger.error (parser.print_help())

    except Exception as e:
        logger.error ("Configuration Failed: " + str(e))
    

if __name__ == '__main__':
    main()
