##############################################################################
## File:   visualize-features-with-pca.py
## Author: Gregory Diamos <gregory.diamos@gmail.com>
## Date:   Sunday January 19, 2014
## Brief:  A simple script for visualizing the principle components of
##         features extracted by minerva.
###############################################################################


# Import Modules
from sklearn.decomposition import PCA

from argparse import ArgumentParser

from matplotlib import pyplot

from numpy import array

import csv

# MAIN
def main():
	parser = ArgumentParser(description="Principle Component Visualization of Minerva Features")
	
	parser.add_argument("-i", "--input-file", default = "",
		help = "The input file path (.csv).")
	parser.add_argument("-o", "--output-file", default = "",
		help = "The output file path (.png, .pdf, etc).")
		
	arguments = parser.parse_args()
	
	try:
		viz = Visualizer(vars(arguments))
		
		viz.run()
	
	except ValueError as e:
		print "Bad Inputs: " + str(e) + "\n\n"
		print parser.print_help()
	except SystemError as e:
		print >> sys.stderr, "Visualization Failed: \n\n" + str(e)

# Visualizer
class Visualizer:
	def __init__(self, options):
		self.inputFilename  = options["input_file" ]
		self.outputFilename = options["output_file"]

	def run(self):
		pca  = PCA(n_components = 2)
		labels, data = self.extractData()
		
		pca.fit(data)
		
		transformedData = pca.transform(data)
		
		self.visualize(labels, transformedData)	
		
	def extractData(self):
		inputFile = open(self.inputFilename, 'r')
		
		reader = csv.reader(inputFile)
		
		labels = []
		data   = []
		
		for row in reader:
			labels.append(row[0])
			data.append([float(x) for x in row[1:]])

		return labels, data
	
	def visualize(self, labels, data):
		
		print data.shape
		
		groups = self.groupByLabel(labels, data)
	
		for group in reversed(groups):
			(label, groupData) = group

			pyplot.plot(groupData[:,0], groupData[:,1], 'o', label=label)

		pyplot.legend()

		if self.outputFilename == "":
			pyplot.show()
		else:
			pyplot.savefig(self.outputFilename)

	def groupByLabel(self, labels, data):
		groups = {}
		
		for label, row in zip(labels, data):
			if not label in groups:
				groups[label] = []
			
			groups[label].append(list(row))
		
		flatGroups = []

		for label, data in groups.iteritems():
			flatGroups.append((label, array(data)))

		return flatGroups


# Main Guard
if __name__ == "__main__":
    main()

