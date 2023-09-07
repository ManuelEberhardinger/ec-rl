import numpy as np
import datetime
import glob
import os
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import gym
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.minigrid.main import run_dreamcoder
from dreamcoder.domains.minigrid.nn_model import MinigridFeatureExtractor
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.domains.minigrid.primitives import *
gym.logger.set_level(40)


#from dcd import train

def minigrid_options(parser):
    parser.add_argument("--primitives",
                        default="old", type=str,
                        choices=["new", "old"])
    parser.add_argument("--chunkSize", default=0, type=int)
    parser.add_argument("--groupby",
                        default="run_id", type=str)
    parser.add_argument("--maxTasks",
                        default=100, type=int)
    parser.add_argument("--randomChunks", action="store_true")
    parser.add_argument("--increaseChunks", action="store_true")


def run_iterations(tasks, outputDirectory, resumeIteration, iterations):
    args = commandlineArguments(
        enumerationTimeout=720,
        structurePenalty=1.5,
        recognitionSteps=5000,
        biasOptimal=False,
        contextual=True,
        a=3,
        topK=2,
        iterations=1,
        useRecognitionModel=True,
        helmholtzRatio=0.5,
        featureExtractor=MinigridFeatureExtractor,
        maximumFrontier=10,
        CPUs=numberOfCPUs(),
        pseudoCounts=30.0, 
        extras=minigrid_options)
    args.pop('resume')
    args.pop('iterations')
    if resumeIteration is not None:
        resumeIteration = str(resumeIteration)
    
    return run_dreamcoder(args, tasks, 
                          outputDirectory,
                          iterations=iterations,
                          resumeIteration=resumeIteration,
                          chunkSize=args.pop('chunkSize'),
                          maxTasks=args.pop('maxTasks'),
                          groupby=args.pop('groupby'),
                          randomChunks=args.pop('randomChunks'),
                          increaseChunks=args.pop('increaseChunks')
                          )


if __name__ == '__main__':
    g = Grammar.uniform(basePrimitives())
    # p = g.sample(arrow(tmap, tint, arrow(arrow(arrow(arrow(tmap, tint, tint, tobj), tobj, tbool)))),
    #             maximumDepth = 100, maxAttempts = 100)
    # print(p)

    timestamp = datetime.datetime.now().isoformat()
    #outputDirectory = "experimentOutputs/minigrid/%s" % timestamp
    outputDirectory = "experimentOutputs/minigrid/2022-09-22T13:50:53.850422"
    os.system("mkdir -p %s" % outputDirectory)
    
    #data_dir = '/home/ma/e/eberhardinger/workspaces/ec/dreamcoder/domains/minigrid/collected_data/ued-MultiGrid-GoalLastEmptyAdversarialEnv-Edit-v0-domain_randomization-noexpgrad-lstm256a-lr0.0001-epoch5-mb1-v0.5-gc0.5-henv0.0-ha0.0-plr0.8-rho0.5-n4000-st0.3-positive_value_loss-rank-t0.3-editor1.0-random-n5-baseeasy-tl_0/'
    #data_dir = '/home/ma/e/eberhardinger/workspaces/ec/dreamcoder/domains/minigrid/'
    data_dir = '/home/ma/e/eberhardinger/workspaces/ec/dreamcoder/domains/minigrid/collected_data/2022-09-05T17:15:43.677900/' 
    iteration = None

    if iteration is not None:
        iterations = iteration + 1
    else:
        iterations = 1

    for p in glob.glob(data_dir + '*.npy')[1:]:
        print('-'*200)
        print('-'*200)
        print('-'*200)
        print('resume from iteration:', iteration)
        print('use data:', os.path.basename(p))
        data = np.load(p, allow_pickle=True)

        tmp = run_iterations(data, outputDirectory, iteration, iterations)

        if tmp is None:
            # no tasks so no iteration was run
            continue
        else:
            # we always need to add 1 iteration so that dreamcoder actually runs
            # this way dreamcoder runs until there are no files left in the data dir
            iterations += 1
            iteration = tmp

    # print(tuple(reversed(tasks[0].examples[0][0])))

    # p = Program.parse('(lambda (lambda (#(lambda (lambda (if (eq? $1 $0) forward-action right-action))) (get $0 1 1) 2)))')
    # p = Program.parse(
    #    '(lambda (lambda (lambda (lambda (if (eq-obj? (get $0 0 0) empty-obj) forward-action right-action))) $0)))')
    #run = p.runWithArguments(reversed(tasks[0].examples[0][0]))
    # print(run)
    # run_iterations(data)
