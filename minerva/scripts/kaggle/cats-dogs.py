# Author: Sudnya Padalikar
# Date  : Dec 30 2013
# Brief : A python script to call minerva for cats vs. dogs dataset

#!/usr/bin/python
import argparse
import os
import itertools
import subprocess

unsupervised_learning_reduction_factor = 100
#unsupervised_learning_db = 'unsupervised_learning_db.txt'
#training_db = 'training_db.txt'
#cross_val_db = 'cross_val_db.txt'

unsupervised_learning_db = '../../examples/faces-training-database.txt'
training_db = '../../examples/faces-training-database.txt'
cross_val_db = '../../examples/faces-test-database.txt'

def populate_db(path, freq):
    training_file = open(training_db, 'w')
    unsupervised_learning_file = open(unsupervised_learning_db, 'w')
    cv_file = open(cross_val_db, 'w')
    
    training_count = 0
    unsupervised_count = 0
    for filename in os.listdir(path):
        striped_name = filename.split('.')[0]
        if (striped_name == 'cat'):
            label = 0
        else:
            label = 1
        
        if training_count > freq:
            cv_file.write(path + '/' + filename + ' , ' + str(label) + '\n')
            training_count = 0
        else:
            training_file.write(path + '/' + filename + ' , ' + str(label) + '\n')
            if unsupervised_count > unsupervised_learning_reduction_factor:
                unsupervised_learning_file.write(path + '/' + filename + ' , ' + str(label) + '\n')
                unsupervised_count = 0
            unsupervised_count += 1
        
        training_count += 1

    training_file.close()
    cv_file.close()



def generate_list(start, total_entries=1, increment=1, is_factor=False):
    mylist = []
    prev = 0
    for i in range(0, total_entries):
        # we want to have multiples eg: 4, 8, 16, 32
        if is_factor:
            if len(mylist) == 0:
                mylist.append(start)
            else:
                mylist.append(float(prev)*increment)
        # we want to have additional increments
        else:
            mylist.append(float(start + i*increment))
        prev = mylist[-1]
    #print mylist
    return mylist



def call_minerva(p, options, count):

    knobs = [str(key) + "=" + str(value) for key, value in zip(options.keys(), p)]

    cmd = 'minerva-classifier --options \"' + ",".join(knobs) + "\""

    logs = ' -L ClassifierEngine,UnsupervisedLearnerEngine,LearnerEngine,FinalClassifierEngine'

    model = 'models/' + str(count) + '.tgz'
    redirect_log = ' | tee ' + str(count) + '.log'
    # create model
    create = cmd + ' -n -m ' + model + redirect_log + '.create'
    process = subprocess.Popen(create, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print "Create : " , create
    (stdOutData, stdErrData) = process.communicate()
    
    # unsupervised learning
    unsupervised = cmd + logs + ' -i ' + training_db + ' -l -m ' + model #+ redirect_log + '.unsupervised'
    process = subprocess.Popen(unsupervised, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print "Unsupervised : ", unsupervised
    (stdOutData, stdErrData) = process.communicate()
 
    # supervised learning
    supervised = cmd + logs + ' -i ' + training_db + ' -t -m ' + model #+ redirect_log + '.supervised'
    process = subprocess.Popen(supervised, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print "Supervised : ", supervised
    (stdOutData, stdErrData) = process.communicate()

    # test (we use cross val set here)
    test = cmd + logs + ' -i ' + cross_val_db + ' -c -m ' + model #+ redirect_log + '.test'
    process = subprocess.Popen(test, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print "Classifiy : ", test
    (stdOutData, stdErrData) = process.communicate()

    

def generate_search_space():
    options = {
            "ClassificationModelBuilder::ResolutionX" : generate_list(64, 3, 2, True), 
            "ClassificationModelBuilder::ResolutionY" : generate_list(64, 3, 2, True),
            "ClassificationModelBuilder::ResolutionColorComponents" : generate_list(3),
            "Classifier::DoThresholdTest" : [int(i) for i in generate_list(1)],
            "Classifier::NeuralNetwork::Outputs" : generate_list(1),
            "Classifier::NeuralNetwork::OutputLabel0" : [int(i) for i in generate_list(1)],
            "ClassifierEngine::ImageBatchSize" : generate_list(4, 6, 2, True),
            "NeuralNetwork::Lambda" : generate_list(0.0, 5, 0.2), 
            "NeuralNetwork::Sparsity" : generate_list(0.02, 3, 0.2),
            "NeuralNetwork::SparsityWeight" : generate_list(0.0, 5, 0.2)
            }
    product = [x for x in apply(itertools.product, options.values())]
    counter = 0
    for p in product:
        call_minerva(p, options, counter)
        counter += 1
        if counter > 1:
            break



def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
#    parser.add_argument('-training',    help="path to directory containing training data", type=str, default='/Users/sudnya/Documents/checkout/git/binary-image-classifier/data/train')
    parser.add_argument('-cv',          help="the frequency to populate cross validation for every training sample (1 cv for every 4 training samples -> 20%)", type=int, default=5)
    parser.add_argument('-x',           help="x resolution", type=int, default=48)
    parser.add_argument('-y',           help="y resolution", type=int, default=48)
    parser.add_argument('-trainingsize',help="number of training files to use", type=int, default=5)
    args = parser.parse_args()
#    populate_db(args.training, args.cv)
    generate_search_space()


if __name__ == '__main__':
    main()


