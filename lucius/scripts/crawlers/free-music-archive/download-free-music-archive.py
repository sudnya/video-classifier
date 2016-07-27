
import requests

from argparse import ArgumentParser

import os
import logging

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def touchFile(path, times=None):
    with open(path, 'a'):
        os.utime(path, times)

class Track:
    def __init__(self, url, curator="unkown", album = "unknown",
            title = "unknown", artist = "unknown", date = "unknown", genre = "",
            filename = "unknown"):
        self.url = url
        self.curator = curator
        self.album = album
        self.title = title
        self.artist = artist
        self.date = date
        self.genre = genre
        self.filename = filename

    def getMetadataLine(self):
        return str(self.url + ", " +
                   self.curator + ", " +
                   self.album + ", " +
                   self.title + ", " +
                   self.artist + ", " +
                   self.date + ", " +
                   self.genre + ", " +
                   self.filename + "\n")

    def getDatabaseLine(self):
        return str(self.getRelativePath() + ", \"" +
                   self.getLabel() + "\"\n")

    def getLabel(self):
        return "\"music by the artist " + self.artist + " in the genres " + " and ".join(
            self.genre.split(":")) + "\""

    def getRelativePath(self):
        return os.path.join(self.getRelativeDirectory(), self.filename)

    def getRelativeDirectory(self):
        return os.path.join(self.curator, self.artist, self.album)


class FileLimitReached:
    def __init__(self):
        pass


class Downloader:
    def __init__(self, options):
        self.apiKey = options["api_key"]
        self.fileLimit = int(options["file_limit"])
        self.outputPath = options["output_path"]

        self.downloadedFiles = 0

        self.createLogger(options["verbose"])

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

    def createMetadataFromLine(self, line):
        splitLine = [entry.strip() for entry in line.split(",")]

        if len(splitLine) == 0:
            return None

        metadata = Track(splitLine[0])

        if len(splitLine) >= 2:
            metadata.curator = splitLine[1]

        if len(splitLine) >= 3:
            metadata.album = splitLine[2]

        if len(splitLine) >= 4:
            metadata.title = splitLine[3]

        if len(splitLine) >= 5:
            metadata.artist = splitLine[4]

        if len(splitLine) >= 6:
            metadata.date = splitLine[5]

        if len(splitLine) >= 7:
            metadata.genre = splitLine[6]

        if len(splitLine) >= 8:
            metadata.filename = splitLine[7]

        return metadata


    def downloadFiles(self):
        self.filesDownloadedSoFar = 0

        pages = self.getCuratorPageCount()

        self.logger.debug("Got curator page count of " + str(pages))

        try:
            for pageNumber in range(pages):
                curators = self.getCuratorsOnPage(pageNumber)

                self.logger.debug("Got curator page " + str(pageNumber) + " with " + str(curators))

                for curator in curators:
                    self.downloadFilesFromCurator(curator)

        except FileLimitReached as e:
            pass

    def getCuratorPageCount(self):
        request = requests.get("https://freemusicarchive.org/api/get/curators.json?api_key=" +
            self.apiKey)

        request.raise_for_status()

        pages = 0

        if 'total_pages' in request.json():
            pages = int(request.json()['total_pages'])

        return pages

    def getCuratorsOnPage(self, page):
        request = requests.get("https://freemusicarchive.org/api/get/curators.json?api_key=" +
            self.apiKey + "&page=" + str(page))

        request.raise_for_status()

        if not 'dataset' in request.json():
            return []

        curators = []

        for curator in request.json()['dataset']:
            curators.append(curator['curator_handle'])

        return curators

    def downloadFilesFromCurator(self, curator):
        self.logger.debug(" for curator " + curator)

        pages = self.getAlbumPageCount(curator)

        self.logger.debug(" Got album page count of " + str(pages))

        for pageNumber in range(pages):
            albums = self.getAlbumsOnPage(curator, pageNumber)

            self.logger.debug(" Got album page " + str(pageNumber) + " with " + str(albums))

            for album in albums:
                self.downloadFilesFromAlbum(curator, album)

    def getAlbumPageCount(self, curator):
        request = requests.get("https://freemusicarchive.org/api/get/albums.json?api_key=" +
            self.apiKey + "&curator_handle=" + curator)

        request.raise_for_status()

        pages = 0

        if 'total_pages' in request.json():
            pages = int(request.json()['total_pages'])

        return pages

    def getAlbumsOnPage(self, curator, page):
        request = requests.get("https://freemusicarchive.org/api/get/albums.json?api_key=" +
            self.apiKey + "&page=" + str(page) + "&curator_handle=" + curator)

        request.raise_for_status()

        if not 'dataset' in request.json():
            return []

        albums = []

        for album in request.json()['dataset']:
            albums.append(album['album_handle'])

        return albums

    def downloadFilesFromAlbum(self, curator, album):
        self.logger.debug("  for album " + album)

        pages = self.getTrackPageCount(curator, album)

        self.logger.debug("  Got track page count of " + str(pages))

        for pageNumber in range(pages):
            tracks = self.getTracksOnPage(curator, album, pageNumber)

            self.logger.debug("  Got track page " + str(pageNumber) + " with " +
                str([track.title for track in tracks]))

            for track in tracks:
                if self.downloadedFiles >= self.fileLimit:
                    self.logger.debug("  Skipping track, file limit reached")
                    raise FileLimitReached()

                if self.isTrackAlreadyDownloaded(track):
                    self.logger.debug("  Skipping track, already downloaded")
                    self.downloadedFiles += 1
                    continue

                self.downloadTrack(track)

                if track.data != None:
                    self.logger.debug("  Downloaded track")
                    self.downloadedFiles += 1
                    self.saveTrack(track)
                else:
                    self.logger.warn("  Download failed!")

    def isTrackAlreadyDownloaded(self, track):
        return track.url in self.metadata

    def getTrackPageCount(self, curator, album):
        request = requests.get("https://freemusicarchive.org/api/get/tracks.json?api_key=" +
            self.apiKey + "&curator_handle=" + curator + "&album_handle=" + album)

        request.raise_for_status()

        pages = 0

        if 'total_pages' in request.json():
            pages = int(request.json()['total_pages'])

        return pages

    def getTracksOnPage(self, curator, album, page):

        request = requests.get("https://freemusicarchive.org/api/get/tracks.json?api_key=" +
            self.apiKey + "&page=" + str(page) + "&curator_handle=" + curator +
            "&album_handle=" + album)

        request.raise_for_status()

        if not 'dataset' in request.json():
            return []

        tracks = []

        for track in request.json()['dataset']:
            url = 'https://freemusicarchive.org/file/' + track['track_file']
            title = track['track_title']
            artist = track['artist_name']
            date = track['track_date_created']
            genres = ":".join([genre['genre_title'] for genre in track['track_genres']])
            filename = self.getFilename(track['track_file'])

            tracks.append(Track(url, curator, album, title, artist, date, genres, filename))

        return tracks

    def getFilename(self, path):
        return os.path.split(path)[1]

    def downloadTrack(self, track):
        url = track.url

        track.data = None

        try:
            track.data = self.downloadData(url)
        except Exception as e:
            self.logger.warning("Downloading track from URL '" + url + "' failed with '" +
                str(e) + "'.")
            pass

    def downloadData(self, url):
        self.logger.debug("   Downloading track from url \'" + url + "\'")
        request = requests.get(url)

        request.raise_for_status()

        return request.content

    def saveTrack(self, track):
        with open(self.metadataPath, 'a') as metadataFile:
            metadataFile.write(track.getMetadataLine())

        with open(self.databasePath, 'a') as databaseFile:
            databaseFile.write(track.getDatabaseLine())

        directory = os.path.join(self.outputPath, track.getRelativeDirectory())
        path = os.path.join(self.outputPath, track.getRelativePath())

        mkdir(directory)

        self.logger.info('Writing data to \'' + path + '\'')
        with open(path, 'wb') as dataFile:
            dataFile.write(track.data)



# MAIN
def main():
    parser = ArgumentParser(description="A program to download audio from the free music archive")

    parser.add_argument("-k", "--api-key", default = "IEXBC4ZZ7EA0KR4R",
        help="Free music archive api key.")
    parser.add_argument("-l", "--file-limit", default = 1,
        help="A limit on the maximum number of files to download.")
    parser.add_argument("-o", "--output-path", default = "free-music-archive",
        help = "The path to a directory to store downloaded audio and metadata.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Print out verbose logging info.")

    arguments = parser.parse_args()

    try:
        downloader = Downloader(vars(arguments))

        downloader.run()

    except ValueError as e:
        print "Bad Inputs: " + str(e) + "\n\n"
        print parser.print_help()

main()

