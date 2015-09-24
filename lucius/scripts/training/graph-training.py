
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot

class ExperimentData:
    def __init__(self):
        self.trainingError   = None
        self.validationError = None

def parseCost(line):
    position = line.find("running cost sum")
    if position == -1:
        return None

    remainder = line[position + len("running cost sum"):]

    if len(remainder) == 0:
        return None

    words = remainder.split()

    resultString = words[0].strip(",")

    try:
        result = float(resultString)
    except ValueError:
        return None

    return result


def loadTrainingErrorFromLogFile(path):
    data = []
    with open(path, 'r') as log:
        for line in log:
            cost = parseCost(line)

            if cost == None:
                continue

            data.append(cost)

    return data

def loadValidationError(path):
    return None

def loadLogFile(path):
    logPath        = os.path.join(path, 'log')
    validationPath = os.path.join(path, 'validation-error.csv')

    if not os.path.exists(logPath):
        raise ValueError("The path" + logPath + " does not exist.")

    experimentData = ExperimentData()

    experimentData.trainingError = loadTrainingErrorFromLogFile(logPath)

    if os.path.exists(validationPath):
        experimentData.validationError = loadValidationError(validationPath)

    return experimentData

class Visualizer:
    def __init__(self, arguments):
        self.inputs = arguments["input_file"]
        self.output = arguments["output_file"]

    def run(self):
        if len(self.inputs) == 0:
            raise ValueError("No input files specified.")

        experiments = self.loadExperiments()
        plots = self.plotExperiments(experiments)
        self.savePlots(plots)

    def loadExperiments(self):
        return [loadLogFile(filename) for filename in self.inputs]

    def plotExperiments(self, experiments):
        figure, axes = pyplot.subplots()

        for experiment in experiments:
            axes.plot(range(len(experiment.trainingError)), experiment.trainingError)

        return (figure, axes)

    def savePlots(self, plots):
        figure, axes = plots

        pyplot.figure(figure.number)

        output = self.output

        if len(output) == 0:
            output = os.path.join(self.inputs[0], "training-error.png")

        print "saving training error at " + output

        pyplot.savefig(output)

# MAIN
def main():

    parser = ArgumentParser(description="")

    parser.add_argument("-i", "--input-file", default = [], action="append",
        help = "An input training experiment directory with log files.")
    parser.add_argument("-o", "--output-file", default = "/mnt/www/files/training-error.png",
        help = "The output file path for the figure (.png, .pdf, etc).")

    arguments = parser.parse_args()

    try:
        viz = Visualizer(vars(arguments))

        viz.run()

    except ValueError as e:
        print "Bad Inputs: " + str(e) + "\n\n"
        print parser.print_help()
    except SystemError as e:
        print >> sys.stderr, "Visualization Failed: \n\n" + str(e)


# Main Guard
if __name__ == "__main__":
    main()




