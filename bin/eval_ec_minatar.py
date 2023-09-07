import datetime
import os
import random
import pandas as pd
import numpy as np
import operator
import random
from tqdm import tqdm
import dill
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.task import Task
from dreamcoder.dreamcoder import *
from dreamcoder.domains.minatar.primitives import basePrimitives, tmap, taction
from dreamcoder.grammar import Grammar
from dreamcoder.utilities import testTrainSplit, eprint, numberOfCPUs
from dreamcoder.type import arrow
from dreamcoder.domains.minatar.feature_extractor import *
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.domains.minatar.feature_extractor import convert_to_task_input
from dreamcoder.domains.minatar.utils_text_encoder import generate_samples_with_temp
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from dreamcoder.domains.minatar.utilities import *
os.environ["WANDB_PROJECT"] = "T5-Minatar"


def makeTasks(data, env_name, chunkSize, tolist=False):
    tasks = []
    examples = []
    part = 0
    states, actions, reward = data

    state_action_pairs = list(zip(states, actions))

    for i in range(len(state_action_pairs) - chunkSize):
        examples = []
        for state, action in state_action_pairs[i: i + chunkSize]:
            input_ex = (convert_to_task_input(
                state, jax_data=True, tolist=tolist),)
            output_ex = int(action)
            examples.append((input_ex, output_ex))

        # we check that the chosen actions are not all the same
        # otherwise it is too easy to find a program if all actions/output examples are the same
        # this results in programs such as (lambda (lambda forward-action))
        all_chosen_actions = list(zip(*examples))[1]
        if not all_equal(all_chosen_actions) and len(examples) == chunkSize:
            tasks.append(Task(f'{env_name} size {chunkSize} part {part}',
                              arrow(tmap, taction), examples))
            part += 1
    return tasks


def evaluate_enumerative_search(env_name, testingTasks, path, feature_extractor_class=MinatarFeatureExtractorToken):
    with open(path, "rb") as handle:
        result = dill.load(handle)
    resume = len(result.grammars) - 1
    eprint("Loaded checkpoint from", path)
    grammar = result.grammars[-1] if result.grammars else grammar

    # for eval of enum search without library
    grammar = Grammar.uniform(basePrimitives(env_name))
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
        featureExtractor=feature_extractor_class,
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


def evaluate_model(data, path, env_name, results_path, feature_extractor_class):
    # first check if a csv exists and load the csv then and start after last seq lenght...
    solved_tasks = []
    idx = []
    start_iter = 3
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

    sequence_lengths = range(start_iter, 31)
    for i in sequence_lengths:
        tasks = makeTasks(data, env_name, i, tolist=True)
        print(f'created {len(tasks)} tasks')
        hits = evaluate_enumerative_search(env_name,
                                           tasks, path, feature_extractor_class=feature_extractor_class)
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
    env_name = 'asterix'
    data = np.load(
        f"/home/ma/e/eberhardinger/workspaces/gymnax-blines/notebooks/{env_name}/rollouts.npy", allow_pickle=True)[0]

    path = '/home/ma/e/eberhardinger/workspaces/ec/experimentOutputs/space_invaders/CORRECT_clear-frontiers/2023-05-14T18:44:12.768978'
    path = '/home/ma/e/eberhardinger/workspaces/ec/experimentOutputs/asterix/CORRECT_clear-frontiers/2023-05-14T18:44:17.138576/'
    # ckpt = 'space_invaders_aic=1.0_arity=3_BO=False_CO=False_ES=1_ET=720_env_name=space_invaders_HR=0.5_it=5_MF=10_noConsolidation=False_pc=30.0_RS=5000_RT=720_RR=False_RW=False_solver=python_STM=True_L=1.5_TRR=default_K=5_topkNotMAP=False.pickle'
    ckpt = 'space_invaders_aic=1.0_arity=3_BO=False_CO=False_ES=1_ET=720_env_name=space_invaders_HR=0.5_it=5_MF=10_noConsolidation=False_pc=30.0_RS=5000_RT=720_RR=False_RW=False_solver=python_STM=True_L=1.5_TRR=default_K=5_topkNotMAP=False_graph=True.pickle'
    ckpt = 'asterix_aic=1.0_arity=3_BO=False_CO=False_ES=1_ET=720_env_name=asterix_HR=0.5_it=3_MF=10_noConsolidation=False_pc=30.0_RS=5000_RT=720_RR=False_RW=False_solver=python_STM=True_L=1.5_TRR=default_K=5_topkNotMAP=False_graph=True.pickle'
    # ckpt = 'asterix_aic=1.0_arity=3_BO=False_CO=False_ES=1_ET=720_env_name=asterix_HR=0.5_it=3_MF=10_noConsolidation=False_pc=30.0_RS=5000_RT=720_RR=False_RW=False_solver=python_STM=True_L=1.5_TRR=default_K=5_topkNotMAP=False.pickle'

    ckpt_path = os.path.join(path, ckpt)

    if 'graph=True.pickle' in ckpt:
        results_path = os.path.join(path, 'eval.csv')
    else:
        results_path = os.path.join(path, 'eval_nn.csv')
    results_path = os.path.join(path, 'eval_baseline.csv')
    print(f'Save results in: {results_path}')
    Grammar.uniform(basePrimitives(env_name))
    evaluate_model(data, ckpt_path, env_name,
                   results_path, feature_extractor_class=CNNFeatureExtractor)
