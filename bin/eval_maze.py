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
from dreamcoder.domains.minigrid.primitives import basePrimitives, tmap, taction, idx_to_action, tdirection
from dreamcoder.grammar import Grammar
from dreamcoder.utilities import testTrainSplit, eprint, numberOfCPUs
from dreamcoder.type import arrow
from dreamcoder.domains.minigrid.nn_model_maze import *
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs
import transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AutoTokenizer, TrainingArguments, Seq2SeqTrainer
from bin.maze_T5 import parseData, all_equal, createTestDataFromTasks, get_latest_checkpoint_path, LookupTableCollator, run_on_input_examples
Grammar.uniform(basePrimitives())


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


def generate_samples_with_temp(model, tokenizer, collator, txt, n_samples, temp):
    to_tokenizer = [txt for i in range(n_samples)]
    outputs = model.generate(collator.encode_obs(to_tokenizer).to(
        'cuda'), do_sample=True, max_length=128, temperature=temp)
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return results


def test_programs_on_task(model, tokenizer, collator, task, grammar, n=5, temp=1.0, verbose=False):
    progs = generate_samples_with_temp(
        model, tokenizer, collator, task[0], n, temp)
    found_progs = []
    for i, prog in enumerate(progs):
        if verbose:
            eprint(prog)
        log_prior = run_on_input_examples(
            task[1], prog, grammar, verbose=verbose)
        if log_prior is not None:
            found_progs.append((Program.parse(prog), log_prior))

    if len(found_progs) == 0:
        return None, -1

    found_progs.sort(key=lambda x: x[1], reverse=True)

    best = found_progs[0]
    return best[0], best[1]


def check_test_tasks(model, tokenizer, collator, testTasks, grammar, n_sampling=100, verbose=False):
    stats = []
    solved = 0
    processed = 0
    for tt in (pbar := tqdm(testTasks)):
        p, n = test_programs_on_task(
            model, tokenizer, collator, tt, grammar, n=n_sampling, verbose=verbose)
        stats.append((p, tt))
        processed += 1
        if p is not None:
            solved += 1
        pbar.set_description(f"Rate {solved}/{processed}")
    return stats


def evaluate_T5(testingTasks, path, no_spaces=True, compress=False):
    testTasks = createTestDataFromTasks(
        testingTasks, True, no_spaces=no_spaces, compress=compress)
    checkpoint_dir = get_latest_checkpoint_path(path)
    model = T5ForConditionalGeneration.from_pretrained(
        checkpoint_dir).to('cuda')
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint_dir)
    collator = LookupTableCollator(tokenizer)
    grammar_file = os.path.join(path, 'results.pkl')
    with open(grammar_file, 'rb') as handle:
        result = dill.load(handle)
    grammar = [g['grammar'] for g in result.values()][-1]
    stats = check_test_tasks(model, tokenizer, collator,
                             testTasks, grammar, n_sampling=100, verbose=False)
    solved = [x for x in stats if x[0] is not None]
    return len(solved)


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
    parsed_data = parseData(data)
    for i in sequence_lengths:
        tasks = makeTasks(parsed_data, i)
        hits = evaluate_T5(tasks, path)
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

    path = '/home/ma/e/eberhardinger/workspaces/T5-experimens/new-dsl/'
    path = '/home/ma/e/eberhardinger/workspaces/T5-experimens/noLib-newDsl/'

    Grammar.uniform(basePrimitives())
    df = evaluate_model(data_file, path, os.path.join(path, 'eval.csv'))
