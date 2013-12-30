# Author: Sudnya Padalikar
# Date  : Dec 30 2013
# Brief : A python script to call minerva for cats vs. dogs dataset

#!/usr/bin/python
import argparse
import os

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

def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
    parser.add_argument('-training',    help="path to directory containing training data", type=str, default='/Users/sudnya/Documents/checkout/git/binary-image-classifier/data/train')
    parser.add_argument('-cv',          help="the frequency to populate cross validation for every training sample (1 cv for every 4 training samples -> 20%)", type=int, default=5)
    parser.add_argument('-x',           help="x resolution", type=int, default=48)
    parser.add_argument('-y',           help="y resolution", type=int, default=48)
    parser.add_argument('-trainingsize',help="number of training files to use", type=int, default=5)
    args = parser.parse_args()
    populate_db(args.training, args.cv)

if __name__ == '__main__':
    main()


