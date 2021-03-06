#! /usr/local/bin/python

import os
import math
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

def isfinite(x):
    return (not math.isinf(x)) and (not math.isnan(x))

def sanitizeFigureName(name):
    return name.replace('.', '_')

class ExperimentData:
    def __init__(self, name):
        self.name                 = name
        self.trainingError        = None
        self.trainingIterations   = None
        self.validationError      = None
        self.validationIterations = None

    def resize(self, maximumIterations):
        maximumIterations = lower_bound(self.trainingIterations, maximumIterations)

        if len(self.trainingError) > maximumIterations:
            self.trainingError = self.trainingError[0:maximumIterations]
        if len(self.trainingIterations) > maximumIterations:
            self.trainingIterations = self.trainingIterations[0:maximumIterations]

    def setTrainingError(self, data):
        self.trainingError, self.trainingIterations = data

        self.normalizeTrainingError()

    def setValidationError(self, data):
        self.validationError, self.validationIterations = data

        self.normalizeValidationError()

    def hasValidationError(self):
        if self.validationError == None:
            return False

        if len(self.validationError) == 0:
            return False

        return True

    def removeInvalidTrainingData(self):
        self.trainingError = [ x for x in self.trainingError if isfinite(x) ]

    def removeInvalidValidationData(self):
        self.validationError = [ x for x in self.validationError if isfinite(x) ]

    def normalizeTrainingError(self):
        self.removeInvalidTrainingData()

        if len(self.trainingError) > len(self.trainingIterations):
            self.trainingError = self.trainingError[0:len(self.trainingIterations)]

        if len(self.trainingIterations) > len(self.trainingError):
            self.trainingIterations = self.trainingIterations[0:len(self.trainingError)]

        averageError = []

        average = 0.0
        samples = 1

        for sample in self.trainingError:
            windowSamples = min(100, samples)
            ratio = 1.0 / windowSamples
            average = average * (1.0 - ratio) + ratio * sample
            averageError.append(average)
            samples += 1

        self.trainingError = averageError

    def normalizeValidationError(self):

        self.removeInvalidValidationData()

        if len(self.validationError) > len(self.validationIterations):
            self.validationError = self.validationError[0:len(self.validationIterations)]

        if len(self.validationIterations) > len(self.validationError):
            self.validationIterations = self.validationIterations[0:len(self.validationError)]


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

    def hasAnyValidationError(self):
        for experiment in self.getExperiments():
            if experiment.hasValidationError():
                return True

        return False

    def getSimpleName(self):
        base, extension = os.path.split(self.name)

        if len(extension) == 0:
            return base
        else:
            return extension

def loadTrainingErrorFromLogFile(path):
    data       = []
    iterations = []
    with open(path, 'r') as log:
        for line in log:
            elements = line.split(',')

            if len(elements) != 2:
                continue

            try:
                cost = float(elements[0].strip())
            except ValueError:
                continue

            try:
                iteration = int(elements[1].strip())
            except ValueError:
                continue


            data.append(cost)
            iterations.append(iteration)

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

    return error, [i for i in range(len(error))]

def getContainingDirectory(path):

    head = path
    tail = ""

    while len(tail) == 0:
        if len(head) == 0:
            return '.'

        head, tail = os.path.split(head)

    return tail

def getExperimentName(path):
    return getContainingDirectory(path)


def getGroupName(path):
    name = getExperimentName(path)

    if name == '.':
        name = 'experiment'

    return name

def loadExperiment(path):
    print 'loading experiment from "' + path + '"'

    trainingPath   = os.path.join(path, 'training-error.csv')
    validationPath = os.path.join(path, 'validation-error.csv')
    name           = getExperimentName(path)

    if not os.path.exists(trainingPath):
        raise ValueError("The path" + trainingPath + " does not exist.")

    experimentData = ExperimentData(name)

    experimentData.setTrainingError(loadTrainingErrorFromLogFile(trainingPath))

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
    logPath = os.path.join(path, 'training-error.csv')

    if not os.path.exists(logPath):
        return False

    return True

def discoverGroups(inputs, combineInputs):
    if len(inputs) > 1:
        groups = []
        for input in inputs:
            groups += discoverGroups([input], False)
        if combineInputs:
            for i in range(1,len(groups)):
                for experiment in groups[i].getExperiments():
                    groups[0].addExperiment(experiment)

            return [groups[0]]

        return groups

    path = inputs[0]

    if isExperimentPath(path):
        group = ExperimentGroup(path)
        group.addExperiment(loadExperiment(path))
        return [group]

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

def parseTuple(text, elements):
    if len(text) == 0:
        return None

    split = text.split(",")

    if len(split) != elements:
        raise ValueError("Expecting a " + str(elements) + " tuple, but got " + text)

    return [float(i) for i in split]

class Visualizer:
    def __init__(self, arguments):
        self.inputs = arguments["input_file"]
        self.combineInputs = arguments["combine_inputs"]
        self.output = arguments["output_file"]
        self.xscale = parseTuple(arguments["x_scale"], 2)
        self.yscale = parseTuple(arguments["y_scale"], 2)

        if not os.path.exists(self.output):
            self.output = self.inputs[0]

        self.maximumIterations = int(arguments["maximum_iterations"])
        self.autoScale = arguments["scale"]
        self.logy = arguments["log_y"]
        self.logx = arguments["log_x"]

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
        return discoverGroups(self.inputs, self.combineInputs)

    def plotExperiments(self, experimentGroups):
        plots = []

        for group in experimentGroups:
            plots.append(self.plotTrainingError(group))
            plots.append(self.plotValidationError(group))

        return plots

    def plotTrainingError(self, group):
        if group.empty():
            print "Group " + group.name + " is empty"
            return (None, None, None, None)

        return self.plotGroupData(group.getSimpleName() + '-training', group, True)

    def plotValidationError(self, group):
        if not group.hasAnyValidationError():
            return (None, None, None, None)

        return self.plotGroupData(group.getSimpleName() + '-validation', group, False)

    def plotGroupData(self, name, group, useTraining):
        figure, axes = pyplot.subplots()

        for experiment in group.getExperiments():
            experimentLabel = formatNameForLabel(experiment.name)

            if useTraining:
                if self.logx and self.logy:
                    axes.loglog(experiment.trainingIterations, experiment.trainingError,
                        label=experimentLabel)
                else:
                    axes.plot(experiment.trainingIterations, experiment.trainingError,
                        label=experimentLabel)

            elif experiment.hasValidationError():
                axes.plot(experiment.validationIterations, experiment.validationError,
                    linestyle='--', marker='o', label=(experimentLabel + '-val'))

        percent = max(0.1, min(0.07 * group.size(), .7))

        box = axes.get_position()
        axes.set_position([box.x0, box.y0 + box.height * percent,
                 box.width, box.height * (1.0 - percent)])

        if self.logy:
            axes.set_yscale("log")

        if not self.autoScale:
            if self.yscale != None:
                axes.set_ylim(self.yscale)
            else:
                axes.set_ylim(getYLimit(group.getExperiments()))

            if self.xscale != None:
                axes.set_xlim(self.xscale)

        axes.legend(bbox_to_anchor=(0.0, -.5*percent, 1, 0), loc='upper center',
            ncol=1, mode="expand", borderaxespad=0., fontsize='x-small')

        axes.minorticks_on()

        axes.yaxis.grid(b=True, which='major', linestyle='-')
        axes.yaxis.grid(b=True, which='minor', linestyle='--')

        return (name, group, figure, axes)


    def savePlots(self, plots):
        if plots == None:
            return

        for name, group, figure, axes in plots:

            if name == None:
                continue

            pyplot.figure(figure.number)

            output = self.output

            path = os.path.join(output, sanitizeFigureName(name))

            print "saving plot " + name + " at " + path + '.png'

            pyplot.savefig(path)

            pyplot.close(figure.number)

# MAIN
def main():

    parser = ArgumentParser(description="")

    parser.add_argument("-i", "--input-file", default = [], action="append",
        help = "An input training experiment directory with log files.")
    parser.add_argument("-o", "--output-file", default = "/mnt/www/files/experiments/",
        help = "The output file path for the figure (.png, .pdf, etc).")
    parser.add_argument("-m", "--maximum-iterations", default = 0,
        help = "The maximum number of iterations to draw.")
    parser.add_argument("-x", "--x-scale", default = "",
        help = "The scale for the x axis.")
    parser.add_argument("-y", "--y-scale", default = "",
        help = "The scale for the y axis.")
    parser.add_argument("-c", "--combine-inputs", default = False, action="store_true",
        help = "Combine multiple input paths into a single graph.")
    parser.add_argument("--scale", default = False, action="store_true",
        help = "Choose the y scale automatically.")
    parser.add_argument("--log-y", default = False, action="store_true",
        help = "Use a log scale for y.")
    parser.add_argument("--log-x", default = False, action="store_true",
        help = "Use a log scale for x.")

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




