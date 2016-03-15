#! /usr/local/bin/python


"""Generic linux daemon base class for python 3.x."""

import sys, os, time, atexit, signal

import subprocess
import json
from argparse import ArgumentParser
from luciusgraphing import Visualizer

class daemon:
    """A generic daemon class.

    Usage: subclass the daemon class and override the run() method."""

    def __init__(self, pidfile): self.pidfile = pidfile

    def daemonize(self):
        """Deamonize class. UNIX double fork mechanism."""

        try:
            pid = os.fork()
            if pid > 0:
                # exit first parent
                sys.exit(0)
        except OSError as err:
            sys.stderr.write('fork #1 failed: {0}\n'.format(err))
            sys.exit(1)

        # decouple from parent environment
        os.chdir('/')
        os.setsid()
        os.umask(0)

        # do second fork
        try:
            pid = os.fork()
            if pid > 0:

                # exit from second parent
                sys.exit(0)
        except OSError as err:
            sys.stderr.write('fork #2 failed: {0}\n'.format(err))
            sys.exit(1)

        # redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        si = open(os.devnull, 'r')
        so = open(os.devnull, 'a+')
        se = open(os.devnull, 'a+')

        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        # write pidfile
        atexit.register(self.delpid)

        pid = str(os.getpid())
        with open(self.pidfile,'w+') as f:
            f.write(pid + '\n')

    def delpid(self):
        os.remove(self.pidfile)

    def start(self):
        """Start the daemon."""

        # Check for a pidfile to see if the daemon already runs
        try:
            with open(self.pidfile,'r') as pf:

                pid = int(pf.read().strip())
        except IOError:
            pid = None

        if pid:
            message = "pidfile {0} already exist. " + \
                    "Daemon already running?\n"
            sys.stderr.write(message.format(self.pidfile))
            sys.exit(1)

        # Start the daemon
        self.daemonize()
        self.run()

    def stop(self):
        """Stop the daemon."""

        # Get the pid from the pidfile
        try:
            with open(self.pidfile,'r') as pf:
                pid = int(pf.read().strip())
        except IOError:
            pid = None

        if not pid:
            message = "pidfile {0} does not exist. " + \
                    "Daemon not running?\n"
            sys.stderr.write(message.format(self.pidfile))
            return # not an error in a restart

        # Try killing the daemon process
        try:
            while 1:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.1)
        except OSError as err:
            e = str(err.args)
            if e.find("No such process") > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                print (str(err.args))
                sys.exit(1)

    def restart(self):
        """Restart the daemon."""
        self.stop()
        self.start()

    def run(self):
        """You should override this method when you subclass Daemon.

        It will be called after the process has been daemonized by
        start() or restart()."""

def tryGraphTraining(daemonfile, configuration, plottime):
    daemonfile.write("Refreshing training and validation graphs at interval " +
        str(plottime) + "s\n")

    arguments = {}

    arguments['input_file']  = [configuration['checkpointing']['base-directory']]
    arguments['output_file'] = configuration['checkpointing']['base-directory']
    arguments['maximum_iterations'] = 0
    arguments['scale']       = False

    try:
        Visualizer(arguments).run()
    except Exception as e:
        daemonfile.write(" Creating graphs failed with error " + str(e) + "\n")
        return

    daemonfile.write(" Successfully created graphs\n")

class ExperimentDaemon(daemon):
    def __init__(self, configuration):
        daemon.__init__(self,configuration["pid-file"])

        self.configfile    = configuration["config-file-path"]
        self.logfile       = configuration["log-file-path"]
        self.daemonfile    = configuration["daemon-log-file-path"]
        self.executable    = configuration["benchmark-path"]
        self.configuration = configuration

    def run(self):
        with open(self.logfile, 'w+', 1) as logfile, open(self.daemonfile, 'w+', 1) as daemonfile:
            daemonfile.write('Starting daemon with id ' + str(os.getpid()) + "\n")

            try:
                command = [self.executable, "--input-path", str(self.configfile)]

                environment = self.configuration["environment"]

                if 'system' in self.configuration and 'cuda-device' in self.configuration['system']:
                    environment['CUDA_VISIBLE_DEVICES'] = self.configuration['system']['cuda-device']

                daemonfile.write('Launching training program with ' + str(command) + "\n")
                process = subprocess.Popen(command, stdout=logfile, stderr=logfile,
                    env=environment)
                process.poll()

                timestamp = time.time()
                plottime  = 0.3

                while process.returncode == None:
                    if time.time() > timestamp + 100.0 * plottime:
                        timestamp = time.time()
                        tryGraphTraining(daemonfile, self.configuration, plottime * 100.0)
                        plottime = time.time() - timestamp

                        timestamp = time.time()

                    time.sleep(1)
                    process.poll()

                daemonfile.write("Training program exited with code " +
                    str(process.returncode) + "\n")

            except Exception as e:
                daemonfile.write(str(e))

def resumeExperiment(experimentConfiguration):
    print 'Resuming experiment from ' + experimentConfiguration["config-file-path"]
    ExperimentDaemon(experimentConfiguration).start()

def makeDirectory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def nameDirectory(directory):
    extension = 0

    directory = os.path.abspath(directory)

    while os.path.exists(directory + '-' + str(extension)):
        extension += 1

    return directory + '-' + str(extension)

def createNewExperiment(experimentConfiguration):
    directory = nameDirectory(experimentConfiguration['checkpointing']['base-directory'])
    experimentConfiguration['checkpointing']['base-directory'] = directory

    experimentConfiguration["pid-file"] = os.path.join(directory, 'process.pid')
    experimentConfiguration["log-file-path"] = os.path.join(directory, 'program.output')
    experimentConfiguration["daemon-log-file-path"] = os.path.join(directory, 'daemon.output')
    experimentConfiguration["config-file-path"] = os.path.join(directory, 'configuration.json')
    experimentConfiguration["benchmark-path"] = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'lucius-benchmark-dataset'))
    experimentConfiguration["environment"] = dict(os.environ)

    configFilePath = experimentConfiguration["config-file-path"]

    makeDirectory(directory)

    print 'Creating new experiment at ' + directory

    with open(configFilePath, 'w') as experimentFile:
        experimentFile.write(json.dumps(experimentConfiguration))

    ExperimentDaemon(experimentConfiguration).start()

def loadConfigFile(inputFile, experimentName):
    with open(inputFile, 'r') as configFile:
        experimentConfiguration = json.loads(configFile.read())

    if len(experimentName) != 0:
        experimentConfiguration['name'] = experimentName
        experimentConfiguration['checkpointing']['base-directory'] = experimentName

    return experimentConfiguration

def isExistingExperiment(experimentConfiguration):
    return 'config-file-path' in experimentConfiguration

def runExperiment(arguments):
    experimentName = arguments["experiment_name"]
    inputFile      = arguments["input_file"]

    experimentConfiguration = loadConfigFile(inputFile, experimentName)

    if(isExistingExperiment(experimentConfiguration)):
        resumeExperiment(experimentConfiguration)
    else:
        createNewExperiment(experimentConfiguration)

def main():
    parser = ArgumentParser(description="A script for launching a daemon training process.")

    parser.add_argument("-i", "--input-file", default = "",
        help = "An input training experiment directory with log files.")
    parser.add_argument("-n", "--experiment-name", default = "",
        help = "A unique name for the experiment.")

    arguments = parser.parse_args()

    try:
        runExperiment(vars(arguments))
    except ValueError as e:
        print "Bad Inputs: " + str(e) + "\n\n"
        print parser.print_help()
    except SystemError as e:
        print >> sys.stderr, "Failed to launch training daemon: \n\n" + str(e)



################################################################################
## Guard Main
if __name__ == "__main__":
    main()

################################################################################



