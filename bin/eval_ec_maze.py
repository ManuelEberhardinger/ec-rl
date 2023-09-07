import datetime
import os
import random
import pandas as pd
import numpy as np
import operator
import random
from tqdm import tqdm

try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.task import Task
from dreamcoder.dreamcoder import *
from dreamcoder.domains.minigrid.primitives import basePrimitives, tmap, taction, idx_to_action, tdirection
from dreamcoder.grammar import Grammar
from dreamcoder.utilities import testTrainSplit, eprint, numberOfCPUs
from dreamcoder.type import arrow
from dreamcoder.domains.minigrid.nn_model_maze import *
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs
import transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AutoTokenizer, TrainingArguments, Seq2SeqTrainer
from bin.maze_T5 import parseData, all_equal
Grammar.uniform(basePrimitives())
os.environ["WANDB_PROJECT"] = "T5-Minigrid-Maze"


def makeTasks(data, chunkSize):
    keys = data.groups.keys()
    print('keys:', len(keys))
    tasks = []
    for key in keys:
        to_imitate = data.get_group(key)
        examples = []
        part = 0
        for _, row in to_imitate.iterrows():
            input_ex = (row.obs.astype(int).tolist(),
                        int(row['obs direction'],))
            output_ex = int(row.action)
            examples.append((input_ex, output_ex))
            if chunkSize > 0 and chunkSize <= len(examples):
                # we check that the chosen actions are not all the same
                # otherwise it is too easy to find a program if all actions/output examples are the same
                # this results in programs such as (lambda (lambda forward-action))
                all_chosen_actions = list(zip(*examples))[1]
                if not all_equal(all_chosen_actions):
                    tasks.append(Task(f'perfect maze {key} size {chunkSize} part {part}',
                                 arrow(tmap, tdirection, taction), examples))
                    part += 1
                    # we reset examples and add new chunkSize taskss
                    examples = []

    print(f'Created {len(tasks)} tasks with {chunkSize} chunk size')
    return tasks


data_file = "/home/ma/e/eberhardinger/workspaces/ec/dreamcoder/domains/perfect-maze-minigrid/collected_data/2022-12-10T15:26:33.798573.npy"


def evaluate_enumerative_search(testingTasks, path):
    with open(path, "rb") as handle:
        result = dill.load(handle)
    resume = len(result.grammars) - 1
    eprint("Loaded checkpoint from", path)
    grammar = Grammar.uniform(basePrimitives())
    grammar = result.grammars[-1] if result.grammars else grammar

    # only for baseline right now, remove later...
    # grammar = Grammar.uniform(basePrimitives())
    args = commandlineArguments(
        enumerationTimeout=720,
        structurePenalty=1.5,
        recognitionSteps=5000,
        biasOptimal=False,
        contextual=False,
        a=3,
        topK=5,
        iterations=1,
        useRecognitionModel=True,
        helmholtzRatio=0.5,
        featureExtractor=MinigridMazeFeatureExtractor,
        maximumFrontier=10,
        CPUs=numberOfCPUs(),
        pseudoCounts=30.0,
        extras=None)
    args.pop('solver')
    times = evaluateOnTestingTasks(result, testingTasks, grammar,
                                   CPUs=args.get('CPUs'), maximumFrontier=args.get('maximumFrontier'),
                                   solver='python',
                                   enumerationTimeout=args.get('enumerationTimeout'), evaluationTimeout=args.get('enumerationTimeout'))

    return times


def evaluate_model(data_file, path, results_path):
    # first check if a csv exists and load the csv then and start after last seq lenght...
    solved_tasks = []
    idx = []
    start_iter = 5
    if os.path.exists(results_path):
        df = pd.read_csv(results_path, index_col=0)
        idx = list(df.index)
        solved, all_tasks = df.to_dict('list').values()
        for s, a in zip(solved, all_tasks):
            solved_tasks.append({
                'solved': s,
                'tasks': a
            })
        print(f'loaded from {results_path}')
        print('start from found csv file:', solved_tasks)
        print('index:', idx)
        start_iter = idx[-1] + 1

    sequence_lengths = range(start_iter, 61)
    data = np.load(data_file, allow_pickle=True)
    data = parseData(data)
    for i in sequence_lengths:
        tasks = makeTasks(data, i)
        print(f'created {len(tasks)} tasks')
        hits = evaluate_enumerative_search(tasks, path)
        # return method(tasks, path, env_name)
        solved_tasks.append({
            'solved': hits,
            'tasks': len(tasks)
        })
        idx.append(i)
        df = pd.DataFrame(solved_tasks, index=idx)
        df.to_csv(results_path)
    return df


if __name__ == '__main__':
    data_file = "/home/ma/e/eberhardinger/workspaces/ec/dreamcoder/domains/perfect-maze-minigrid/collected_data/2022-12-10T15:26:33.798573.npy"

    path = '/home/ma/e/eberhardinger/workspaces/ec/experimentOutputs/perfect-maze'
    # ckpt = '2023-05-15T10:14:49.917185/maze_aic=1.0_arity=3_BO=False_CO=False_ES=1_ET=720_HR=0.5_it=40_MF=10_noConsolidation=False_pc=30.0_RS=5000_RT=720_RR=False_RW=False_solver=python_STM=True_L=1.5_TRR=default_K=5_topkNotMAP=False.pickle'
    ckpt = '2023-05-15T10:14:49.917185/maze_aic=1.0_arity=3_BO=False_CO=False_ES=1_ET=720_HR=0.5_it=39_MF=10_noConsolidation=False_pc=30.0_RS=5000_RT=720_RR=False_RW=False_solver=python_STM=True_L=1.5_TRR=default_K=5_topkNotMAP=False_graph=True.pickle'
    ckpt_path = os.path.join(path, ckpt)

    if 'graph=True.pickle' in ckpt:
        results_path = os.path.join(path, 'eval.csv')
    else:
        results_path = os.path.join(path, 'eval_nn.csv')
    # results_path = os.path.join(path, 'eval_baseline.csv')
    print(f'Save results in: {results_path}')
    Grammar.uniform(basePrimitives())
    evaluate_model(data_file, ckpt_path, results_path)
