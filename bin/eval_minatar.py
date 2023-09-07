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


def evaluate_enumerative_search(testingTasks, path, feature_extractor_class=MinatarFeatureExtractorToken):
    with open(path, "rb") as handle:
        result = dill.load(handle)
    resume = len(result.grammars) - 1
    eprint("Loaded checkpoint from", path)
    grammar = result.grammars[-1] if result.grammars else grammar
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
    times = evaluateOnTestingTasks(result, testingTasks, grammar,
                                   CPUs=args.get('CPUs'), maximumFrontier=args.get('maximumFrontier'),
                                   solver=args.get('solver'),
                                   enumerationTimeout=args.get('enumerationTimeout'), evaluationTimeout=args.get('enumerationTimeout'))

    return times


def check_test_tasks(testTasks, grammar, generate_sample_fn, n_sampling=100, verbose=False):
    stats = []
    solved_tasks = set()
    solved = 0
    for tt in (pbar := tqdm(testTasks)):
        p, n = test_programs_on_task(
            tt, grammar, generate_sample_fn, n=n_sampling, verbose=verbose, use_multiprocess=True)
        stats.append((p, n))
        if 'random' not in str(tt[1]):
            solved_tasks.add(tuple([str(tt[1]), str(p)]))

        if p is not None:
            solved += 1
        pbar.set_description(f"Solved: {solved}")
    return stats, solved_tasks, solved


def evaluate_T5(testingTasks, path, iter_folder, env_name, no_spaces=True, compress=False):
    feature_extractor = MinatarFeatureExtractorToken(
        env_name=env_name, max_steps=500, no_spaces=no_spaces, compress=compress)
    # testingTasks = feature_extractor.create_test_tasks(3)
    testTasks = createTestDataFromTasks(feature_extractor, testingTasks, True)
    checkpoint_dir = get_latest_checkpoint_path(path)
    model = T5ForConditionalGeneration.from_pretrained(
        checkpoint_dir).to('cpu')
    model = model.eval()
    model = model.to('cuda')
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    grammar_file = os.path.join(path, iter_folder, 'results.pkl')
    with open(grammar_file, 'rb') as handle:
        result = dill.load(handle)
    grammar = [g['grammar'] for g in result.values()][-1]
    stats, solved_tasks, solved = check_test_tasks(
        testTasks, grammar, lambda x, y, z: generate_samples_with_temp(model, tokenizer, x, y, z, device='cuda'), n_sampling=500, verbose=True)
    return solved


def evaluate_model(data, path, iter_path, env_name, method, results_path):
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
        tasks = makeTasks(data, env_name, i)
        print(f'created {len(tasks)} tasks for seq length {i}')
        hits = method(tasks, path, iter_path, env_name)
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
    env_name = 'space_invaders'
    data = np.load(
        f"/home/ma/e/eberhardinger/workspaces/gymnax-blines/notebooks/{env_name}/rollouts.npy", allow_pickle=True)[0]

    path = [  # (f'/home/ma/e/eberhardinger/workspaces/T5-{env_name}-new_DSL/{env_name}/allActions-noLib-20000p-500s', 'iter-11_seqlength-7')
        (f'/home/ma/e/eberhardinger/workspaces/T5-{env_name}-new_DSL/{env_name}/COPY_allActions-50000p-500s', 'iter-2_seqlength-5'),
    ]  # T5-{env_name}-new_DSL ] # T5-{env_name}-new_DSL

    Grammar.uniform(basePrimitives(env_name))
    for p, iter_path in path:
        df = evaluate_model(data, p, iter_path, env_name,
                            evaluate_T5, os.path.join(p, 'eval.csv'))
