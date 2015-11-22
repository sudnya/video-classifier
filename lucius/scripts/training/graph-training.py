
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot

def lower_bound(array, value):
    index = 0

    for i in range(len(array)):
        if array[i] > value:
            break

        index = i

        if array[i] == value:
            break

    return index


class ExperimentData:
    def __init__(self, name):
        self.name                 = name
        self.trainingError        = None
        self.trainingIterations   = None
        self.validationError      = None
        self.epochs               = None

    def resize(self, maximumIterations):
        maximumIterations = lower_bound(self.trainingIterations, maximumIterations)

        #print maximumIterations, len(self.trainingIterations), self.trainingIterations[0:10]

        if len(self.trainingError) > maximumIterations:
            self.trainingError = self.trainingError[0:maximumIterations]
        if len(self.trainingIterations) > maximumIterations:
            self.trainingIterations = self.trainingIterations[0:maximumIterations]

    def setTrainingError(self, data):
        self.trainingError, self.trainingIterations = data

        self.normalizeTrainingError()

    def setValidationError(self, data):
        self.validationError = data

        self.normalizeValidationError()

    def hasValidationError(self):
        if self.epochs == None or self.validationError == None:
            return False


        if len(self.epochs) == 0 or len(self.validationError) == 0:
            return False

        return True

    def normalizeTrainingError(self):

        if len(self.trainingError) > len(self.trainingIterations):
            self.trainingError = self.trainingError[0:len(self.trainingIterations)]

        if len(self.trainingIterations) > len(self.trainingError):
            self.trainingIterations = self.trainingIterations[0:len(self.trainingError)]

        if len(self.trainingIterations) == 0:
            return

        epochLength = self.trainingIterations[0]

        previousSampleCount = self.trainingIterations[0]
        increment = 0
        maxvalue = 0
        self.epochs = []

        self.trainingIterations[0] = 0

        for i in range(1, len(self.trainingIterations)):
            sampleCount = self.trainingIterations[i]

            if sampleCount > previousSampleCount:
                increment += epochLength
                epochLength = sampleCount
                self.epochs.append(increment)

            previousSampleCount = sampleCount

            samplesSoFar = epochLength - sampleCount
            self.trainingIterations[i] = samplesSoFar + increment

    def normalizeValidationError(self):
        maxEpochLength = 0
        previousIterations = 0

        for iterations in self.epochs:
            maxEpochLength = max(maxEpochLength, iterations - previousIterations)
            previousIterations = iterations

        epochs = []
        previousIterations = 0

        for iterations in self.epochs:
            epochLength = iterations - previousIterations

            if epochLength > (maxEpochLength / 2):
                epochs.append(iterations)

            previousIterations = iterations

        self.epochs = epochs

        if len(self.validationError) > len(self.epochs):
            self.validationError = self.validationError[0:len(self.epochs)]

        if len(self.epochs) > len(self.validationError):
            self.epochs = self.epochs[0:len(self.validationError)]


class ExperimentGroup:
    def __init__(self, name):
        self.name        = name
        self.experiments = []

    def addExperiment(self, experiment):
        self.experiments.append(experiment)

    def size(self):
        return len(self.experiments)

    def getExperiments(self):
        return self.experiments

    def empty(self):
        return self.size() == 0

    def resize(self, size):
        for experiment in self.getExperiments():
            experiment.resize(size)

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

def parseIteration(line):
    position = line.find(" image frames, ")
    if position == -1:
        return None

    remainder = line[position + len(" image frames, "):]

    if len(remainder) == 0:
        return None

    words = remainder.split()

    resultString = words[0].strip()

    try:
        result = int(resultString)
    except ValueError:
        return None

    return result


def loadTrainingErrorFromLogFile(path):
    data       = []
    iterations = []
    with open(path, 'r') as log:
        for line in log:
            cost      = parseCost(line)
            iteration = parseIteration(line)

            if cost != None:
                data.append(cost)

            if iteration != None:
                if len(data) == 0:
                    if len(iterations) == 0:
                        iterations.append(iteration)
                    iterations[0] = iteration
                else:
                    iterations.append(iteration)

    #print path, data[0:10], iterations[0:10]

    return data, iterations

def loadValidationError(path):
    error = []

    with open(path, 'r') as errorFile:
        for line in errorFile:
            elements = line.split(',')

            for element in elements:
                try:
                    result = float(element.strip())
                except ValueError:
                    continue

                error.append(result)

    return error

def getExperimentName(path):
    head, tail = os.path.split(path)

    if len(tail) == 0:
        return head

    return tail

def getGroupName(path):
    name = getExperimentName(path)

    if name == '.':
        name = 'training_error'

    return name

def loadExperiment(path):
    logPath        = os.path.join(path, 'log')
    validationPath = os.path.join(path, 'validation-error.csv')
    name           = getExperimentName(path)

    if not os.path.exists(logPath):
        raise ValueError("The path" + logPath + " does not exist.")

    experimentData = ExperimentData(name)

    experimentData.setTrainingError(loadTrainingErrorFromLogFile(logPath))

    if os.path.exists(validationPath):
        experimentData.setValidationError(loadValidationError(validationPath))

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

def discoverGroups(inputs):
    if len(inputs) > 1:
        groups = []
        for input in inputs:
            groups += discoverGroups([input])
        return groups

    path = inputs[0]

    if isExperimentPath(path):
        return [ExperimentGroup(path)]

    groups = []

    # find groups
    for root, directories, files in os.walk(path):
        name = getGroupName(root)
        group = ExperimentGroup(name)

        for directory in directories:
            directoryPath = os.path.join(root, directory)

            if isExperimentPath(directoryPath):
                group.addExperiment(loadExperiment(directoryPath))

        if not group.empty():
            groups.append(group)

    return groups

def getYLimit(experiments):
    limit = 0

    for experiment in experiments:
        if len(experiment.trainingError) > 0:
            limit = max(limit, max(experiment.trainingError))

    return [0, limit]

class Visualizer:
    def __init__(self, arguments):
        self.inputs = arguments["input_file"]
        self.output = arguments["output_file"]

        if not os.path.exists(self.output):
            self.output = self.inputs[0]

        self.maximumIterations = int(arguments["maximum_iterations"])
        self.autoScale = arguments["scale"]

    def run(self):
        if len(self.inputs) == 0:
            raise ValueError("No input files specified.")

        experimentGroups = self.loadExperiments()

        self.formatExperiments(experimentGroups)

        plots = self.plotExperiments(experimentGroups)
        self.savePlots(plots)

    def formatExperiments(self, experiments):
        if self.maximumIterations != 0:
            for experiment in experiments:
                experiment.resize(self.maximumIterations)

    def loadExperiments(self):
        return discoverGroups(self.inputs)

    def plotExperiments(self, experimentGroups):
        plots = []

        for group in experimentGroups:

            figure, axes = pyplot.subplots()

            for experiment in group.getExperiments():
                experimentLabel = formatNameForLabel(experiment.name)
                axes.plot(experiment.trainingIterations, experiment.trainingError,
                    label=experimentLabel)
                if experiment.hasValidationError():
                    #print experiment.epochs, experiment.validationError
                    axes.plot(experiment.epochs, experiment.validationError, linestyle='--', marker='o',
                        label=(experimentLabel + '-val'))

            percent = max(0.1, min(0.07 * group.size(), .7))

            box = axes.get_position()
            axes.set_position([box.x0, box.y0 + box.height * percent,
                     box.width, box.height * (1.0 - percent)])

            if not self.autoScale:
                axes.set_ylim(getYLimit(group.getExperiments()))

            axes.legend(bbox_to_anchor=(0.0, -.5*percent, 1, 0), loc='upper center',
                ncol=1, mode="expand", borderaxespad=0., fontsize='x-small')

            axes.minorticks_on()

            axes.yaxis.grid(b=True, which='major', linestyle='-')
            axes.yaxis.grid(b=True, which='minor', linestyle='--')

            plots.append((group, figure, axes))

        return plots

    def savePlots(self, plots):
        for group, figure, axes in plots:

            pyplot.figure(figure.number)

            output = self.output

            path = os.path.join(output, group.name)

            print "saving training error at " + path

            pyplot.savefig(path)

# MAIN
def main():

    parser = ArgumentParser(description="")

    parser.add_argument("-i", "--input-file", default = [], action="append",
        help = "An input training experiment directory with log files.")
    parser.add_argument("-o", "--output-file", default = "/mnt/www/files/experiments/",
        help = "The output file path for the figure (.png, .pdf, etc).")
    parser.add_argument("-m", "--maximum-iterations", default = 0,
        help = "The maximum number of iterations to draw.")
    parser.add_argument("--scale", default = False, action="store_true",
        help = "Choose the y scale automatically.")

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




