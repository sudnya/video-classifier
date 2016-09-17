
from flask import Flask, request
from flask_restful import Resource, Api
import json
import StringIO
from pydub import AudioSegment
import base64

from lucius import Model

class DescribeAudio(Resource):
    def post(self):
        rawData = request.data

        response = None

        try:
            response = self.handleRequest(rawData)
        except Exception as e:
            response = self.errorHandler(e, rawData)

        return response

    def handleRequest(self, data):

        dataDictionary = json.loads(data)

        if not 'type' in dataDictionary:
            raise ValueError('Missing \'type\' field in request json payload.')

        if not 'key' in dataDictionary:
            raise ValueError('Missing \'key\' field in request json payload.')

        if not 'data' in dataDictionary:
            raise ValueError('Missing \'data\' field in request json payload.')

        dataType = dataDictionary['type']
        key = dataDictionary['key']

        if not self.validateKey(key):
            raise ValueError('Invalid api key \'' + key + '\'.')

        if not self.validateDataType(dataType):
            raise ValueError('Invalid data type \'' + dataType + '\'.')

        audioData = dataDictionary['data']

        fileString = self.getStringAsFile(audioData)

        audioSegment = AudioSegment.from_file(fileString, format=dataType)

        description = self.describeAudio(audioSegment)

        return self.formResponse(description, dataType)

    def validateKey(self, key):
        # TODO
        return True

    def errorHandler(self, exception, data):
        response = {}

        response['error'] = str(exception)
        response['request'] = data[0:60]

        return response

    def validateDataType(self, dataType):
        supportedFormats = self.getSupportedTypes()

        return dataType in supportedFormats

    def getSupportedTypes(self):
        return set(['wav', 'mp3', 'raw', 'flac'])

    def getStringAsFile(self, string):
        return StringIO.StringIO(base64.b64decode(string))

    def getWavData(self, audioSegment):
        stringFile = StringIO.StringIO()

        audioSegment.export(stringFile, 'wav')

        return stringFile

    def describeAudio(self, audioSegment):
        audioData = self.getWavData(audioSegment)

        return model.infer(audioData, 'wav')

    def formResponse(self, description, dataType):
        response = {}

        response['description'] = description
        response['type'] = dataType

        return response

class ArtifactsApplication:
    def __init__(self):
        self.application = Flask(__name__)
        self.api = Api(application)
        self.defaultModelPath = "model.tar.gz"


    def run(self, debug=False):
        with Model(defaultModelPath) as model:
            describer = DescribeAudio()
            describer.setModel(model)

            self.api.add_resource(describer, "/audio/describe")

            self.application.run(debug)

if __name__ == '__main__':
    ArtifactsApplication().run(debug=True)
else:
    ArtifactsApplication.run()



