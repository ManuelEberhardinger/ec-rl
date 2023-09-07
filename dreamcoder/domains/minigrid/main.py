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
    columns = ['step', 'process', 'obs', 'obs direction', 'next obs', 'next direction',
               'action', 'reward', 'done', 'level seed', 'run_id']
    run_id = 0
    new_arr = []
    sorted_data = np.asarray(sorted(taskData, key=operator.itemgetter(1, 0)))
    for row in sorted_data:
        new_arr.append(np.append(row, [run_id]))
        if row[8] == True:
            run_id += 1

    df = pd.DataFrame(new_arr, columns=columns)
    df = df.drop(['step', 'process'], axis=1)
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


def makeTasks(data, groupby, chunkSize=0, maxTasks=100, randomChunks=False, increaseChunks=False):
    grouped = parseData(data, groupby=groupby)
    keys = grouped.groups.keys()
    print('keys:', len(keys))
    tasks = []
    for key in keys:
        to_imitate = grouped.get_group(key)
        examples = []
        chunk = 0

        if increaseChunks and chunkSize > len(to_imitate):
            continue

        if randomChunks:
            chunkSize = random.randint(3, 20)

        for _, row in to_imitate.iterrows():
            input_ex = ((row.obs[0] * 10).astype(int).tolist(), int(row['obs direction'],))
            output_ex = row.action
            examples.append((input_ex, output_ex))

            if chunkSize > 0 and chunkSize == len(examples):
                # we check that the chosen actions are not all the same
                # otherwise it is too easy to find a program if all actions/output examples are the same
                # this results in programs such as (lambda (lambda forward-action))
                all_chosen_actions = list(zip(*examples))[1]
                if not all_equal(all_chosen_actions):
                    tasks.append(Task(f'minigrid {key} size {chunkSize} part {chunk}', arrow(tmap, tdirection, taction), examples))
                examples = []
                chunk += 1

        # we make the assumption that each new run of the RL env is a new Task
        # however all tasks are actually the same task
        # maybe we need to restructure this in the future into one single tasks with a lot of examples
        if chunkSize == 0:
            tasks.append(Task(f'minigrid {key}', arrow(tmap, tdirection, taction), examples))
        if len(tasks) > maxTasks:
            break
    print(f'Created {len(tasks)} tasks with chunkSize of {chunkSize} and grouped by {groupby}')
    return tasks


def run_dreamcoder(arguments, taskData, outputDirectory, resumeIteration=None, iterations=None,
                   chunkSize=0, groupby='run_id', maxTasks=None, randomChunks=False, increaseChunks=False, dcd=None):
    random.seed(42)

    if increaseChunks:
        print('create increasing chunks for tasks')
        tasks = []
        for i in range(3, 15):
            tasks.extend(makeTasks(taskData, groupby, chunkSize=i, maxTasks=maxTasks))
    else:
        tasks = makeTasks(taskData, groupby, chunkSize=chunkSize, maxTasks=maxTasks, randomChunks=randomChunks)
    # return tasks
    eprint("Got %d tasks..." % len(tasks))

    if len(tasks) == 0:
        return None

    #test, train = testTrainSplit(tasks, 0.7)
    test, train = tasks, tasks
    # random.shuffle(test)
    #test = test[:100]
    eprint("Training on", len(train), "tasks")
    eprint("Testing on", len(test), "tasks")

    arguments.pop('primitives', None)
    arguments.pop('resume', None)
    arguments.pop('iterations', None)
    # Create grammar
    grammar = Grammar.uniform(basePrimitives())

    # EC iterate
    generator = ecIterator(grammar,
                           train,
                           testingTasks=test,
                           outputPrefix="%s/minigrid" % outputDirectory,
                           resume=resumeIteration,
                           iterations=iterations,
                           dcd=dcd,
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

    
