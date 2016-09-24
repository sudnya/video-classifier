
import requests
from argparse import ArgumentParser
import base64
import json

class Tester:
    def __init__(self, arguments):
        self.url  = arguments['url']
        self.key  = arguments['key']
        self.path = arguments['path']

    def run(self):
        payload = {}

        payload['key']  = self.key
        payload['type'] = 'mp3'
        payload['data'] = base64.b64encode(self.loadData())

        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json'
        }

        request = requests.post(self.url, data=json.dumps(payload), headers=headers)
        print request.request.headers

        request.raise_for_status()

        print request.json()

    def loadData(self):
        with open(self.path, 'rb') as audioFile:
            return audioFile.read()

def main():
    parser = ArgumentParser(description="A program to test the audio description API.")

    parser.add_argument("-u", "--url", default = "http://127.0.0.1:5000/audio/describe",
        help="URL to test.")
    parser.add_argument("-i", "--path", default = "/Users/gregorydiamos/temp/rickroll.mp3",
        help="Path to the audio file to transmit")
    parser.add_argument("-k", "--key", default = "1",
        help="API key to test.")

    arguments = parser.parse_args()

    tester = Tester(vars(arguments))

    tester.run()


if __name__ == '__main__':
    main()


