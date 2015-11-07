
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot

class ExperimentData:
    def __init__(self, name):
        self.name            = name
        self.trainingError   = None
        self.validationError = None

    def resize(self, maximumIterations):
        if len(self.trainingError) > maximumIterations:
            self.trainingError = self.trainingError[0:maximumIterations]

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

def getExperimentName(path):
    head, tail = os.path.split(path)

    if len(tail) == 0:
        return head

    return tail

def loadLogFile(path):
    logPath        = os.path.join(path, 'log')
    validationPath = os.path.join(path, 'validation-error.csv')
    name           = getExperimentName(path)

    if not os.path.exists(logPath):
        raise ValueError("The path" + logPath + " does not exist.")

    experimentData = ExperimentData(name)

    experimentData.trainingError = loadTrainingErrorFromLogFile(logPath)

    if os.path.exists(validationPath):
        experimentData.validationError = loadValidationError(validationPath)

    return experimentData

def formatNameForLabel(name):
    characters = 95
    limit = min(len(name), characters)

    result = name[0:limit]

    for i in range(characters, len(name), characters):
        limit = min(len(name), i+characters)
        result += "\n" + name[i:limit]

    return result

def isExperimentPath(path):
    logPath = os.path.join(path, 'log')

    if not os.path.exists(logPath):
        return False

    return True

def discoverInputs(inputs):
    if len(inputs) != 1:
        return inputs

    path = inputs[0]

    if isExperimentPath(path):
        return inputs

    # try one level
    contents = os.listdir(path)

    results = []

    for possibleExperiment in contents:
        experimentPath = os.path.join(path, possibleExperiment)
        if isExperimentPath(experimentPath):
            results.append(experimentPath)

    return results

class Visualizer:
    def __init__(self, arguments):
        self.inputs = arguments["input_file"]
        self.output = arguments["output_file"]
        self.maximumIterations = int(arguments["maximum_iterations"])

    def run(self):
        if len(self.inputs) == 0:
            raise ValueError("No input files specified.")

        experiments = self.loadExperiments()

        self.formatExperiments(experiments)

        plots = self.plotExperiments(experiments)
        self.savePlots(plots)

    def formatExperiments(self, experiments):
        if self.maximumIterations != 0:
            for experiment in experiments:
                experiment.resize(self.maximumIterations)

    def loadExperiments(self):
        self.inputs = discoverInputs(self.inputs)
        return [loadLogFile(filename) for filename in self.inputs]

    def plotExperiments(self, experiments):
        figure, axes = pyplot.subplots()

        for experiment in experiments:
            experimentLabel = formatNameForLabel(experiment.name)
            axes.plot(range(len(experiment.trainingError)), experiment.trainingError,
                label=experimentLabel)

        percent = min(0.07 * len(experiments), .7)

        box = axes.get_position()
        axes.set_position([box.x0, box.y0 + box.height * percent,
                 box.width, box.height * (1.0 - percent)])

        axes.legend(bbox_to_anchor=(0.0, -.5*percent, 1, 0), loc='upper center',
            ncol=1, mode="expand", borderaxespad=0., fontsize='x-small')

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
    parser.add_argument("-m", "--maximum-iterations", default = 0,
        help = "The maximum number of iterations to draw.")

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




