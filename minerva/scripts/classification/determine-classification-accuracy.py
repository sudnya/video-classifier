##############################################################################
## File:   determine-classification-accuracy.py
## Author: Gregory Diamos <gregory.diamos@gmail.com>
## Date:   Sunday January 19, 2014
## Brief:  A simple script for determining how good a straightforward classifier
##         could perform on a labeled training set.  The point of the script
##         is the judge the quality of a given set of input features.
###############################################################################

# Import Modules
import csv

from argparse import ArgumentParser
from sklearn  import svm
from sklearn  import cross_validation

# MAIN
def main():
	parser = ArgumentParser(
		description="Accuracy Determination of Minerva Features")
	
	parser.add_argument("-i", "--input-file", default = "",
		help = "The input file path (.csv).")
		
	arguments = parser.parse_args()
	
	try:
		classifier = Classifier(vars(arguments))
		
		classifier.run()
	
	except ValueError as e:
		print "Bad Inputs: " + str(e) + "\n\n"
		print parser.print_help()
	except SystemError as e:
		print >> sys.stderr, "Visualization Failed: \n\n" + str(e)

## Determines classification accuracy over a training set
class Classifier:
	def __init__(self, options):
		self.inputFilename  = options["input_file" ]
		self.dataSplits     = 10
		
	def run(self):
		labels, data = self.extractData()
		
		classifierEngine = self.getEngine()
		
		scores = cross_validation.cross_val_score(
			classifierEngine, data, self.convertLabels(labels),
			cv=self.dataSplits)
		
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
		
	def getEngine(self):
		return svm.SVC(kernel='rbf')
	
	def convertLabels(self, labels):
		labelMap = {}
		
		newLabels = []
		
		for label in labels:
			if not label in labelMap:
				labelMap[label] = float(len(labelMap))
	
			newLabels.append(labelMap[label])
	
		return newLabels
	
	def extractData(self):
		inputFile = open(self.inputFilename, 'r')
		
		reader = csv.reader(inputFile)
		
		labels = []
		data   = []
		
		for row in reader:
			labels.append(row[0])
			data.append([float(x) for x in row[1:]])
		
		return labels, data


# Main Guard
if __name__ == "__main__":
    main()
    
    
