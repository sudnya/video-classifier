
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot

class ExperimentData:
    def __init__(self, name):
        self.name                 = name
        self.trainingError        = None
        self.trainingIterations   = None
        self.validationError      = None
        self.validationIterations = None

    def resize(self, maximumIterations):
        if len(self.trainingError) > maximumIterations:
            self.trainingError = self.trainingError[0:maximumIterations]

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

    def setTrainingErrror(self, data):
        self.trainingErrror, self.trainingIterations = data

        if len(self.trainingError) > len(self.trainingIterations):
            self.trainingError = self.trainingError[0:trainingIterations]



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

            if iterations != None:
                iterations.append(iteration)

    return data, iterations

def loadValidationError(path):
    return None

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
        limit = max(limit, max(experiment.trainingError))

    return [0, limit]

class Visualizer:
    def __init__(self, arguments):
        self.inputs = arguments["input_file"]
        self.output = arguments["output_file"]
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
                axes.plot(experiment.trainingErrorIterations, experiment.trainingError,
                    label=experimentLabel)

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




