import datetime
import os
import random
import pandas as pd
import numpy as np
import operator
import random

try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.task import Task
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.domains.minigrid.primitives import basePrimitives, tmap, taction, idx_to_action, tdirection
from dreamcoder.grammar import Grammar
from dreamcoder.utilities import testTrainSplit, eprint
from dreamcoder.type import arrow
from dreamcoder.domains.minigrid.nn_model import MinigridFeatureExtractor
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs

def all_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def parseData(taskData, groupby='run_id', verbose=False):
    columns = ['process', 'obs', 'obs direction', 
               'action', 'reward', 'done', 'run_id']

    df = pd.DataFrame(taskData, columns=columns)
    df = df.drop(['process'], axis=1)
    df.action = df.action.apply(lambda x: x[0])
    df.insert(0, 'run_id', df.pop('run_id'))

    group = df.groupby('run_id')

    groups_to_consider = []

    for key in group.groups.keys():
        g = group.get_group(key)
        if verbose:
            print(f'group {key}')
        if not g[g.reward > 0.0].count().all():
            if verbose:
                print(f'no reward..')
        else:
            reward = g[g.reward > 0.0].reward.iloc[0]
            if reward < 0.9:
                if verbose:
                    print(f'skip {key} because reward is to small')
                continue
            if verbose:
                print(f'needed {g.shape[0]} steps. Reward: {reward}')
            groups_to_consider.append(key)
    group = group.filter(lambda x: x.run_id.mean() in groups_to_consider)
    print(group.shape)
    return group.groupby(groupby)


def makeTasks(data, chunkSize):
    keys = data.groups.keys()
    print('keys:', len(keys))
    tasks = []
    for key in keys:
        to_imitate = data.get_group(key)
        examples = []
        part = 0
        for _, row in to_imitate.iterrows():
            input_ex = ((row.obs[0] * 10).astype(int).tolist(), int(row['obs direction'],))
            output_ex = int(row.action)
            examples.append((input_ex, output_ex))

            if chunkSize > 0 and chunkSize == len(examples):
                # we check that the chosen actions are not all the same
                # otherwise it is too easy to find a program if all actions/output examples are the same
                # this results in programs such as (lambda (lambda forward-action))
                all_chosen_actions = list(zip(*examples))[1]
                if not all_equal(all_chosen_actions):
                    tasks.append(Task(f'perfect maze {key} size {chunkSize} part {part}', arrow(tmap, tdirection, taction), examples))
                    part += 1
                    # we reset examples and add new chunkSize taskss
                    examples = []

    # select random obs and actions to test
    for key in keys:
        to_imitate = data.get_group(key)
        examples = []
        part = 0
        already_sampled = []
        while len(to_imitate.index) - len(already_sampled) > chunkSize:
            curr_sample = random.sample([x for x in to_imitate.index if x not in already_sampled], chunkSize)
            for i in curr_sample:
                row = to_imitate.loc[i]
                input_ex = ((row.obs[0] * 10).astype(int).tolist(), int(row['obs direction'],))
                output_ex = int(row.action)
                examples.append((input_ex, output_ex))

                if chunkSize > 0 and chunkSize == len(examples):
                    # we check that the chosen actions are not all the same
                    # otherwise it is too easy to find a program if all actions/output examples are the same
                    # this results in programs such as (lambda (lambda forward-action))
                    all_chosen_actions = list(zip(*examples))[1]
                    if not all_equal(all_chosen_actions):
                        tasks.append(Task(f'perfect maze {key} size {chunkSize} random {part}', arrow(tmap, tdirection, taction), examples))
                        part += 1
                        # we reset examples and add new chunkSize taskss
                        examples = []

            already_sampled += curr_sample

    print(f'Created {len(tasks)} tasks with chunkSize of {chunkSize}')
    return tasks


def run_dreamcoder(arguments, taskData, outputDirectory, chunkSize, resumeIteration=None, iterations=None):
    random.seed(42)
    tasks = makeTasks(taskData, chunkSize=chunkSize)
    # return tasks
    eprint("Got %d tasks..." % len(tasks))

    if len(tasks) == 0:
        return None

    arguments.pop('primitives', None)
    arguments.pop('resume', None)
    arguments.pop('iterations', None)
    # Create grammar
    grammar = Grammar.uniform(basePrimitives())

    # EC iterate
    generator = ecIterator(grammar,
                           tasks,
                           outputPrefix="%s/maze" % outputDirectory,
                           resume=resumeIteration,
                           iterations=iterations,
                           **arguments)
    for i, _ in enumerate(generator):
        pass
        #print('ecIterator count {}'.format(i))

    if resumeIteration is None:
        return 1

    return int(resumeIteration) + 1


class DreamCoder():
    def __init__(self, venv, ued_venv, level_sampler, level_store, outputDirectory, num_processes):
        self.venv = venv
        self.ued_venv = ued_venv
        self.level_sampler = level_sampler
        self.level_store = level_store
        self.outputDirectory = outputDirectory
        self.resumeIteration = None
        self.iterationStep = 2
        self.iterations = self.iterationStep
        self.num_processes = num_processes

    def run(self, taskData):
        args = commandlineArguments(
            enumerationTimeout=720,
            structurePenalty=1.5,
            recognitionSteps=5000,
            recognitionTimeout=None,
            biasOptimal=False,
            contextual=False,
            a=3,
            topK=2,
            iterations=1,
            useRecognitionModel=True,
            helmholtzRatio=0.5,
            featureExtractor=MinigridFeatureExtractor,
            maximumFrontier=10,
            CPUs=numberOfCPUs(),
            pseudoCounts=30.0)

        if self.resumeIteration is not None:
            self.resumeIteration = str(self.resumeIteration)
        
        tmp = run_dreamcoder(args, taskData, 
                            self.outputDirectory,
                            iterations=self.iterations,
                            resumeIteration=self.resumeIteration,
                            chunkSize=0,
                            maxTasks=100,
                            groupby='run_id',
                            randomChunks=False,
                            increaseChunks=True,
                            dcd={
                                'venv': self.venv,
                                'ued_venv': self.ued_venv,
                                'level_sampler': self.level_sampler,
                                'level_store': self.level_store,
                                'num_processes': self.num_processes
                            })
        
        if tmp is None:
            # no tasks so no iteration was run
            return
        else:
            # we always need to add iterationStep so that dreamcoder actually runs
            # this way dreamcoder runs until there are no files left in the data dir
            self.iterations += self.iterationStep
            self.resumeIteration = tmp

    
