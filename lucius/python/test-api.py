
import lucius
from argparse import ArgumentParser

class Tester:
    def __init__(self, arguments):
        self.modelPath     = arguments["model_path"]
        self.inputDataPath = arguments["input_data_path"]
        self.inputType     = arguments["input_type"]
        self.verbose       = arguments["verbose"]

    def run(self):
        try:
            with lucius.Model(self.modelPath, self.verbose) as model:
                with open(self.inputDataPath, 'rb') as inputFile:
                    label = model.infer(inputFile.read(), self.inputType)

        except Exception as e:
            print "Test Failed: exception: " + str(e)
            return

        print "Test Passed: label is '" + str(label) + "'"

def main():
    parser = ArgumentParser(description="A program to test the lucius python API.")

    parser.add_argument("-m", "--model-path", default = "", help="Model to test.")
    parser.add_argument("-i", "--input-data-path", default = "",
        help="Input data file to run inference on.")
    parser.add_argument("-t", "--input-type", default = ".wav",
        help="Input data file type.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help="Print out debugging information.")

    arguments = parser.parse_args()

    tester = Tester(vars(arguments))

    tester.run()

if __name__ == '__main__':
    main()







