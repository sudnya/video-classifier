
import requests

from argparse import ArgumentParser

import os
import logging
import random

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def touchFile(path, times=None):
    with open(path, 'a'):
        os.utime(path, times)

def sanitize(path):
    return path.replace(",", "")

def getExtension(path):
    filename, extension = os.path.splitext(path)

    return extension

class Sound:
    def __init__(self, url, user="unkown",
            title = "unknown", date = "unknown", description = "",
            tags = "", filename = "unknown"):
        self.url = url.encode('ascii', errors='ignore')
        self.user = user.encode('ascii', errors='ignore')
        self.title = title.encode('ascii', errors='ignore')
        self.date = date.encode('ascii', errors='ignore')
        self.description = description.encode('ascii', errors='ignore')
        self.tags = tags.encode('ascii', errors='ignore')
        self.filename = sanitize(filename.encode('ascii', errors='ignore'))

    def getMetadataLine(self):
        return str(self.url + ", " +
                   self.user + ", " +
                   self.title + ", " +
                   self.date + ", " +
                   self.description + ", " +
                   self.tags + ", " +
                   self.filename + "\n")

    def getDatabaseLine(self):
        return self.getRelativePath() + ", \"" + self.getLabel() + "\"\n"

    def getLabel(self):
        return str(" ".join(self.tags.split(":")))

    def getRelativePath(self):
        return os.path.join(self.getRelativeDirectory(), self.filename)

    def getLeadingTag(self):
        if len(self.tags) == 0:
            return "unknown"

        return self.tags.split(":")[0]

    def getRelativeDirectory(self):
        return self.getLeadingTag()


class FileLimitReached:
    def __init__(self):
        pass


class Downloader:
    def __init__(self, options):
        self.clientId = options["client_id"]
        self.apiKey = options["api_key"]
        self.authorizationCode = options["authorization_code"]
        self.fileLimit = int(options["file_limit"])
        self.outputPath = options["output_path"]
        self.timeout = float(options['timeout'])
        self.accessToken = options["access_token"]

        self.downloadedFiles = 0

        self.createLogger(options["verbose"])

        self.getAccessToken()

    def getAccessToken(self):

        if len(self.accessToken) != 0:
            self.logger.info("Using access token '" + self.accessToken + "'")
            return

        self.logger.info("Getting access token...")

        params = {"client_id" : self.clientId,
                  "client_secret" : self.apiKey,
                  "grant_type" : "authorization_code",
                  "code" : self.authorizationCode}

        request = requests.post("https://www.freesound.org/apiv2/oauth2/access_token/",
            params=params, timeout=self.timeout)

        request.raise_for_status()

        if not 'access_token' in request.json():
            raise KeyError('Could not find field "access_token" in access request response')

        self.accessToken = request.json()['access_token']

        print "Got access token '" + self.accessToken + "'"

    def createLogger(self, is_verbose):
        self.logger = logging.getLogger("Downloader")

        # create console handler and set level to debug
        ch = logging.StreamHandler()

        if is_verbose:
            ch.setLevel(logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.WARNING)
            self.logger.setLevel(logging.WARNING)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

    def run(self):
        self.createOutput()
        self.loadExistingFileMetadata()

        self.downloadFiles()

    def createOutput(self):

        self.logger.info("Creating output directory structure at \'" + self.outputPath + "\'")
        mkdir(self.outputPath)
        self.metadataPath = os.path.join(self.outputPath, "metadata.txt")

        touchFile(self.metadataPath)

        self.databasePath = os.path.join(self.outputPath, "database.txt")
        touchFile(self.databasePath)

    def loadExistingFileMetadata(self):
        metadataPath = os.path.join(self.outputPath, "metadata.txt")

        self.metadata = {}

        with open(metadataPath, 'r') as metadataFile:
            for line in metadataFile:
                metadata = self.createMetadataFromLine(line)

                if metadata == None:
                    continue

                self.metadata[metadata.url] = metadata

        self.downloadedFiles = len(self.metadata)

    def createMetadataFromLine(self, line):
        splitLine = [entry.strip() for entry in line.split(",")]

        if len(splitLine) == 0:
            return None

        metadata = Sound(splitLine[0])

        if len(splitLine) >= 2:
            metadata.user = splitLine[1]

        if len(splitLine) >= 3:
            metadata.title = splitLine[2]

        if len(splitLine) >= 4:
            metadata.date = splitLine[3]

        if len(splitLine) >= 5:
            metadata.description = splitLine[4]

        if len(splitLine) >= 6:
            metadata.tags = splitLine[5]

        if len(splitLine) >= 7:
            metadata.filename = splitLine[6]

        return metadata

    def downloadFiles(self):
        self.filesDownloadedSoFar = 0

        pageCount = self.getSearchPageCount()

        self.logger.debug("Got sound page count of " + str(pageCount))

        pageCount = min(pageCount, self.fileLimit / 75)

        self.logger.debug("Using page count of " + str(pageCount))

        try:
            pageRange = range(pageCount)
            random.shuffle(pageRange)
            for pageNumber in pageRange:
                sounds = []

                try:
                    sounds = self.getSoundsOnPage(pageNumber)
                except Exception as e:
                    self.logger.warn("Failed to get sounds on page " + str(pageNumber))
                    self.logger.warn(" with error " + str(e))
                    pass

                self.logger.debug("Got sounds on page " + str(pageNumber))

                for sound in sounds:
                    if self.downloadedFiles >= self.fileLimit:
                        self.logger.debug("  Skipping track, file limit reached")
                        raise FileLimitReached()

                    if self.isSoundAlreadyDownloaded(sound):
                        self.logger.debug("  Skipping track, already downloaded")
                        continue

                    self.downloadSound(sound)

                    if sound.data != None:
                        self.logger.debug("  Downloaded sound")
                        self.downloadedFiles += 1
                        self.saveSound(sound)
                    else:
                        self.logger.warn("  Download failed!")

        except FileLimitReached as e:
            pass

    def getSearchPageCount(self):
        result = self.getSearchPageResult(0)

        if not 'count' in result:
            raise KeyError("Field 'count' not found in search request response.")

        return result['count']

    def getSearchPageResult(self, page):
        request = requests.get("https://www.freesound.org/apiv2/search/text/?token=" +
            self.apiKey + "&query=&page=" + str(page) + "sort=downloads_desc&page_size=150" +
            "&fields=id,username,url,name,created,description,tags,download",
            timeout=self.timeout)

        request.raise_for_status()

        return request.json()

    def isSoundAlreadyDownloaded(self, sound):
        return sound.url in self.metadata

    def getSoundsOnPage(self, pageNumber):
        pageResult = self.getSearchPageResult(pageNumber)

        if not 'results' in pageResult:
            raise KeyError("Missing field 'results' in search page response.")

        sounds = []

        for sound in pageResult['results']:

            url = sound['download']
            user = sound['username']
            title = sound['name']
            date = sound['created']
            description = sound['description']
            tag = ':'.join(sound['tags'])
            filename = str(sound['id']) + getExtension(title)

            newSound = Sound(url, user, title, date, description, tag, filename)

            self.logger.info("  Get got '" + repr(newSound.getMetadataLine()) + "'")

            sounds.append(newSound)

        return sounds

    def downloadSound(self, sound):
        url = sound.url

        sound.data = None

        try:
            sound.data = self.downloadData(url)
        except Exception as e:
            self.logger.warning("Downloading sound from URL '" + url + "' failed with '" +
                str(e) + "'.")


    def getFilename(self, path):
        return os.path.split(path)[1]

    def downloadData(self, url):
        self.logger.debug("   Downloading sound data from url \'" + url + "\'")

        request = requests.get(url, headers={'Authorization' : "Bearer " + self.accessToken},
            timeout=self.timeout)

        request.raise_for_status()

        return request.content

    def saveSound(self, sound):
        with open(self.metadataPath, 'a') as metadataFile:
            metadataFile.write(sound.getMetadataLine())

        with open(self.databasePath, 'a') as databaseFile:
            databaseFile.write(sound.getDatabaseLine())

        directory = os.path.join(self.outputPath, sound.getRelativeDirectory())
        path = os.path.join(self.outputPath, sound.getRelativePath())

        mkdir(directory)

        self.logger.info('Writing data to \'' + path + '\'')
        with open(path, 'wb') as dataFile:
            dataFile.write(sound.data)



# MAIN
def main():
    parser = ArgumentParser(description="A program to download audio from freesound.org")

    parser.add_argument("-c", "--client-id", default = "ODMmGFX8FAtrlT5bSTRZ",
        help="Free sound client id.")
    parser.add_argument("-k", "--api-key", default = "mkKYBUgk6Gqxm7nMWV6dge4IgOwSbq6prIFKL7ef",
        help="Free sound api key.")
    parser.add_argument("-a", "--authorization-code", default="", help="Free sound OAuth2 "
        "authorization code used for generating an access token.")
    parser.add_argument("-T", "--access-token", default="", help="Free sound OAuth2 "
        "access token if it has already been obtained.")
    parser.add_argument("-t", "--timeout", default = 10.0, help="Timeout for requests.")
    parser.add_argument("-l", "--file-limit", default = 1,
        help="A limit on the maximum number of files to download.")
    parser.add_argument("-o", "--output-path", default = "free-sounds",
        help = "The path to a directory to store downloaded audio and metadata.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Print out verbose logging info.")

    arguments = parser.parse_args()

    downloader = Downloader(vars(arguments))

    downloader.run()

main()


