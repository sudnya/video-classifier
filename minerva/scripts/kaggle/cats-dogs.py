# Author: Sudnya Padalikar
# Date  : Dec 30 2013
# Brief : A python script to call minerva for cats vs. dogs dataset

#!/usr/bin/python
import argparse
import os
import itertools

def populate_db(path, freq):
    training_file = open('training_db.txt', 'w')
    cv_file = open('cross_val_db.txt', 'w')
    
    training_count = 0
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

    model = 'models/' + str(count) + '.tgz'
    # create model
    create = cmd + ' -n ' + model
    # unsupervised learning
    unsup = cmd + ' -l -m ' + model 
    # supervised learning
    supervised = cmd + ' -t -m ' + model
    # test (we use cross val set here)
    test = cmd + ' -c -m ' + model
    
    print "Create : " , create
    print "Unsupervised : ", unsup
    print "Supervised : ", supervised
    print "Classifiy : ", test

    #print [dict(zip(options.keys(), p))] , "\n"

def generate_search_space():
    options = {
            "ClassificationModelBuilder::ResolutionX" : generate_list(64, 3, 2, True), 
            "ClassificationModelBuilder::ResolutionY" : generate_list(64, 3, 2, True),
            "ClassificationModelBuilder::ResolutionColorComponents" : generate_list(3),
            "Classifier::NeuralNetwork::Outputs" : generate_list(1),
            "Classifier::NeuralNetwork::Outputs0" : generate_list(1),
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



def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
    parser.add_argument('-training',    help="path to directory containing training data", type=str, default='/Users/sudnya/Documents/checkout/git/binary-image-classifier/data/train')
    parser.add_argument('-cv',          help="the frequency to populate cross validation for every training sample (1 cv for every 4 training samples -> 20%)", type=int, default=5)
    parser.add_argument('-x',           help="x resolution", type=int, default=48)
    parser.add_argument('-y',           help="y resolution", type=int, default=48)
    parser.add_argument('-trainingsize',help="number of training files to use", type=int, default=5)
    args = parser.parse_args()
    populate_db(args.training, args.cv)
    generate_search_space()


if __name__ == '__main__':
    main()


