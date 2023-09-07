import re
import os
import random
import pandas as pd
import numpy as np
import operator
import random
from tqdm import tqdm
import dill
import wandb

try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.task import Task
from dreamcoder.frontier import *
from dreamcoder.dreamcoder import *
from dreamcoder.domains.minigrid.primitives import basePrimitives, tmap, taction, idx_to_action, tdirection
from dreamcoder.grammar import Grammar
from dreamcoder.utilities import testTrainSplit, eprint, numberOfCPUs
from dreamcoder.type import arrow
from dreamcoder.domains.minigrid.nn_model_maze import *
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.program import Program
from dreamcoder.primitiveGraph import graphPrimitivesFromGrammar

import transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AutoTokenizer, TrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset, DataLoader


def get_out_string(out):
    if out == 0:
        return 'a'
    elif out == 1:
        return 'b'
    elif out == 2:
        return 'c'
    elif out == 3:
        return 'd'
    elif out == 4:
        return 'e'
    elif out == 5:
        return 'f'
    elif out == 6:
        return 'g'
    elif out == 7:
        return 'h'
    elif out == 8:
        return 'i'
    elif out == 9:
        return 'j'


def all_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def parseData(taskData, groupby='run_id', verbose=False):
    columns = ['process', 'obs', 'obs direction',
               'action', 'reward', 'done', 'run_id']

    df = pd.DataFrame(taskData, columns=columns)
    df = df.drop(['process'], axis=1)
    df.action = df.action.apply(lambda x: x[0])
    df.obs = df.obs.apply(lambda x: np.flip(x[0] * 10, 1))
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
            if reward < 0.85:
                if verbose:
                    print(f'skip {key} because reward is to small')
                continue
            if verbose:
                print(f'needed {g.shape[0]} steps. Reward: {reward}')
            groups_to_consider.append(key)
    group = group.filter(lambda x: x.run_id.mean() in groups_to_consider)
    print(group.shape)
    return group.groupby(groupby)


def makeTasks(data, rand_min=10, rand_max=50, randomChunkSize=True, fixedChunkSize=None):
    assert randomChunkSize or (not randomChunkSize and fixedChunkSize)
    keys = data.groups.keys()
    print('keys:', len(keys))
    tasks = []
    for key in keys:
        to_imitate = data.get_group(key)
        if randomChunkSize:
            chunkSize = random.randint(rand_min, rand_max)
        else:
            chunkSize = fixedChunkSize
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
                    if randomChunkSize:
                        chunkSize = random.randint(rand_min, rand_max)
                    else:
                        chunkSize = fixedChunkSize

        if len(examples) > 3:
            all_chosen_actions = list(zip(*examples))[1]
            if not all_equal(all_chosen_actions):
                tasks.append(Task(f'perfect maze {key} size {chunkSize} part {part}',
                             arrow(tmap, tdirection, taction), examples))

    # select random obs and actions to test
    # for key in keys:
    #     if randomChunkSize:
    #         chunkSize = random.randint(rand_min, rand_max)
    #     else:
    #         chunkSize = fixedChunkSize
    #     to_imitate = data.get_group(key)
    #     examples = []
    #     part = 0
    #     already_sampled = []
    #     while len(to_imitate.index) - len(already_sampled) > chunkSize:
    #         curr_sample = random.sample([x for x in to_imitate.index if x not in already_sampled], chunkSize)
    #         for i in curr_sample:
    #             row = to_imitate.loc[i]
    #             input_ex = (row.obs.astype(int).tolist(), int(row['obs direction'],))
    #             output_ex = int(row.action)
    #             examples.append((input_ex, output_ex))

    #             if chunkSize > 0 and chunkSize == len(examples):
    #                 # we check that the chosen actions are not all the same
    #                 # otherwise it is too easy to find a program if all actions/output examples are the same
    #                 # this results in programs such as (lambda (lambda forward-action))
    #                 all_chosen_actions = list(zip(*examples))[1]
    #                 if not all_equal(all_chosen_actions):
    #                     tasks.append(Task(f'perfect maze {key} size {chunkSize} random {part}', arrow(
    #                         tmap, tdirection, taction), examples))
    #                     part += 1
    #                     # we reset examples and add new chunkSize taskss
    #                     examples = []

    #         already_sampled += curr_sample

    print(f'Created {len(tasks)} tasks with {fixedChunkSize} chunk size')
    return tasks


def get_inp_string_for_task(inp_string, no_spaces=False, compress=False):
    inp_string = inp_string.replace(',', '').replace(
        '[', '').replace(']', '').replace('(', '').replace(')', '')
    if no_spaces:
        inp_string = inp_string.replace(' ', '')
    if compress:
        inp_string = inp_string.replace('2222', '4')

    return inp_string


def get_inp_prompt(task, no_spaces, compress):
    inp_prompt = ''
    for examples in task.examples:
        inp = str(examples[0])
        out = examples[1]
        inp_string = get_inp_string_for_task(
            inp, no_spaces=no_spaces, compress=compress)
        out_string = get_out_string(out)
        inp_prompt += f'{inp_string} {out_string} '
    return inp_prompt.strip()


class DatasetCreator():
    def __init__(self, featureExtractor, grammar):
        super().__init__()
        self.id = id
        self.featureExtractor = featureExtractor
        self.generativeModel = grammar

    def taskEmbeddings(self, tasks):
        return {task: self.featureExtractor.featuresOfTask(task).data.cpu().numpy()
                for task in tasks}

    def replaceProgramsWithLikelihoodSummaries(self, frontier):
        return Frontier(
            [FrontierEntry(
                program=self.grammar.closedLikelihoodSummary(
                    frontier.task.request, e.program),
                logLikelihood=e.logLikelihood,
                logPrior=e.logPrior) for e in frontier],
            task=frontier.task)

    def sampleHelmholtz(self, requests, statusUpdate=None, seed=None):
        if seed is not None:
            random.seed(seed)
        request = random.choice(requests)

        program = self.generativeModel.sample(
            request, maximumDepth=8, maxAttempts=100)
        if program is None:
            return None
        task = self.featureExtractor.taskOfProgram(program, request)

        if statusUpdate is not None:
            flushEverything()
        if task is None:
            return None

        if hasattr(self.featureExtractor, 'lexicon'):
            if self.featureExtractor.tokenize(task.examples) is None:
                return None

        ll = self.generativeModel.logLikelihood(request, program)
        frontier = Frontier([FrontierEntry(program=program,
                                           logLikelihood=0., logPrior=ll)],
                            task=task)
        return frontier

    def sampleProgramWithTask(self, requests, min_len, max_len, statusUpdate=None, seed=None):
        if seed is not None:
            random.seed(seed)
        request = random.choice(requests)

        program = self.generativeModel.sample(
            request, maximumDepth=6, maxAttempts=100)
        if program is None:
            return None
        task = self.featureExtractor.taskOfProgram(
            program, request, min_len=min_len, max_len=max_len)

        if statusUpdate is not None:
            flushEverything()
        if task is None:
            return None, None

        return task, program

    def sampleManyProgramsWithTasks(self, tasks, N, min_len, max_len, verbose=False):
        if verbose:
            eprint("Sampling %d programs from the prior..." % N)
        flushEverything()
        requests = list({t.request for t in tasks})

        frequency = N / 50
        startingSeed = random.random()

        if verbose:
            looper = tqdm(range(N))
        else:
            looper = range(N)

        # Sequentially for ensemble training.
        data = [self.sampleProgramWithTask(requests, min_len, max_len,
                                           statusUpdate='.' if n % frequency == 0 else None,
                                           seed=startingSeed + n) for n in looper]

        flushEverything()
        data = [z for z in data if not any(x is None for x in z)]
        if verbose:
            eprint()
            eprint("Got %d/%d valid datapoints." % (len(data), N))
        flushEverything()

        return data

    def sampleManyFrontiers(self, tasks, N):
        eprint("Sampling %d programs from the prior..." % N)
        flushEverything()
        requests = list({t.request for t in tasks})

        frequency = N / 50
        startingSeed = random.random()

        # Sequentially for ensemble training.
        frontiers = [self.sampleHelmholtz(requests,
                                          statusUpdate='.' if n % frequency == 0 else None,
                                          seed=startingSeed + n) for n in range(N)]

        eprint()
        flushEverything()
        frontiers = [z for z in frontiers if z is not None]
        eprint()
        eprint("Got %d/%d valid frontiers." % (len(frontiers), N))
        flushEverything()

        return frontiers

    def createDataset(self, tasks, N, with_tasks=False, no_spaces=False, compress=False, min_len=5, max_len=20, verbose=False):
        dataset = []
        data = self.sampleManyProgramsWithTasks(
            tasks, N, min_len, max_len, verbose=verbose)
        for task, program in data:
            inp_prompt = get_inp_prompt(task, no_spaces, compress)

            if with_tasks:
                dataset.append((inp_prompt, str(program), task))
            else:
                dataset.append((inp_prompt, str(program)))

        return dataset


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # entry[1] is the spec, need to call repr to turn it into a string. entry[0] is the prog_str already
        ret = {"input_ids": self.tokenizer([entry[0] for entry in batch], padding=True,  truncation=True, return_tensors='pt').input_ids,
               "labels": self.tokenizer([entry[1] for entry in batch], padding=True,  return_tensors='pt').input_ids}
        return ret


class LookupTableCollator:
    def __init__(self, tokenizer):
        self.lookup_table = self.generate_lookup_table()
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # entry[1] is the spec, need to call repr to turn it into a string. entry[0] is the prog_str already
        ret = {"input_ids": self.encode_obs([entry[0] for entry in batch], padding=True, truncation=True),
               "labels": self.tokenizer([entry[1] for entry in batch], padding=True, return_tensors='pt').input_ids}
        return ret

    def generate_lookup_table(self):
        tokens = []
        # encode from 1 - 1 * 25
        tokens += ['1' * i for i in range(1, 26)]

        # encode from 2 - 2 * 25
        tokens += ['2' * i for i in range(1, 26)]

        # encode also 0, 3 , 8 and zero, one, two, three, four, five, six
        tokens += ['0', '3', '8', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
        return {k: v+1 for v, k in enumerate(tokens)}

    def encode_obs(self, observations, padding=True, truncation=True):
        tokens = []

        for obs in observations:
            seq = []
            for item in re.finditer(r"(.)\1*", obs):
                word = item.group(0).strip()
                if not word:
                    continue

                token = self.lookup_table[word]
                seq.append(token)
            if padding:
                while len(seq) < 512:
                    seq.append(0)
            if truncation:
                seq = seq[:512]

            tokens.append(seq)
        return torch.tensor(tokens)


class FactoringDataset(Dataset):
    def __init__(self, dataset_itself):
        self.data = dataset_itself

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def createTestDataFromTasks(tasks, with_tasks=False, no_spaces=False, compress=False):
    dataset = []
    for task in tasks:
        inp_prompt = get_inp_prompt(task, no_spaces, compress)
        if with_tasks:
            dataset.append((inp_prompt, task))
        else:
            dataset.append((inp_prompt))

    return dataset


def generate_samples_with_temp(txt, n_samples, temp):
    to_tokenizer = [txt for i in range(n_samples)]
    outputs = model.generate(collator.encode_obs(to_tokenizer).to(
        'cuda'), do_sample=True, max_length=128, temperature=temp)
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return results


def run_on_input_examples(task, program, grammar, verbose=False):
    try:
        program = Program.parse(program)
        for inp, out in task.examples:
            # output ex is the action
            pred = runWithTimeout(lambda: program.runWithArguments(inp), None)
            if verbose:
                eprint('Input:', inp, 'Out:', out, 'Pred:', pred)
            if out != pred:
                return None

    except Exception as e:
        if verbose:
            eprint(e)
        return None
    try:
        logPrior = grammar.logLikelihood(task.request, program)
    except:
        if verbose:
            eprint(
                f'run_on_input_examples > logLikelihood failed for: {program}')
        # not a correctly typed program
        return None

    return logPrior


def test_programs_on_task(task, grammar, n=5, temp=1.0, verbose=False):
    progs = generate_samples_with_temp(task[0], n, temp)
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


def generate_frontiers(testTasks, stats):
    frontiers = []
    for tt, stat in zip(testTasks, stats):
        _, task = tt
        program, prior = stat
        if program is None:
            continue
        frontier = Frontier([FrontierEntry(program=program,
                            logLikelihood=0., logPrior=prior)],
                            task=task)
        frontiers.append(frontier)
    return frontiers


def check_test_tasks(testTasks, grammar, n_sampling=100, verbose=False):
    stats = []
    solved_tasks = set()
    looper = testTasks
    if verbose:
        looper = tqdm(testTasks)

    for tt in looper:
        p, n = test_programs_on_task(
            tt, grammar, n=n_sampling, verbose=verbose)
        stats.append((p, n))
        if 'random' not in str(tt[1]):
            solved_tasks.add(tuple([str(tt[1]), str(p)]))
    return stats, solved_tasks


TABLE_DATA = []


def log_solved_tasks(stats, seq_len, verbose=False):
    solved = [x for x in stats if x[0] is not None]
    rate = len(solved)/len(stats) * 100
    if verbose:
        print(f'{len(solved)}/{len(stats)} -> {rate}%')
    try:
        TABLE_DATA.append([seq_len, rate, len(solved), len(stats)])
        t = wandb.Table(columns=["sequence lenght",
                        "rate", "solved", "all"], data=TABLE_DATA)
        wandb.log({'solved_stats': t})
    except Exception as e:
        if verbose:
            print(e)

    return solved


def save_results(output_dir, results, stats, grammars):
    if output_dir is not None:
        path = os.path.join(output_dir, 'results.pkl')
        with open(path, "wb") as handle:
            try:
                dill.dump(results, handle)
            except TypeError as e:
                eprint(results)
                return False

        results_json = {
            'programs': [str(x[0]) for x in stats if x[0] is not None],
            'grammar': grammars[-1].json()
        }

        json_path = os.path.join(output_dir, 'results.json')
        with open(json_path, "wb") as handle:
            try:
                dill.dump(results_json, handle)
            except TypeError as e:
                eprint(results)
                return False

        try:
            i = list(results.keys())[-1]
            graphPrimitivesFromGrammar(
                grammars, f"{output_dir}T5_primitives_{i}_")
            wandb.save(json_path)
            wandb.save(path)
            wandb.save(f"{output_dir}T5_primitives_{i}_depth.pdf.pdf")
            wandb.save(f"{output_dir}T5_primitives_{i}_unordered.pdf.pdf")
        except Exception as e:
            print(e)

    return True


def get_latest_checkpoint_path(output_dir):
    checkpoint_dirs = list(filter(lambda x: os.path.isdir(os.path.join(output_dir, x))
                           and 'checkpoint-' in x, os.listdir(output_dir)))
    checkpoint_dir = sorted(checkpoint_dirs, key=lambda x: int(
        x.replace('checkpoint-', '')))[-1]
    checkpoint_dir = os.path.join(output_dir, checkpoint_dir)
    return checkpoint_dir


def ec_iterator_T5(grammar, parsed_data, training_args, start_iter=5, n_sampling=100, random_programs=100000, output_dir='', min_len_random_programs=5, max_len_random_programs=20, lib_learning=True, no_spaces=False, compress=False, verbose=False):
    results = {}
    all_solved_tasks = set()
    grammars = [grammar]

    for i in range(start_iter, 60):
        if verbose:
            print(f'Start iteration {i}')
            transformers.logging.set_verbosity_info()

        tasks = []
        # for j in range(i - 1, i + 1):
        #    tasks += makeTasks(parsed_data,
        #                       randomChunkSize=False, fixedChunkSize=j)
        tasks = makeTasks(parsed_data, randomChunkSize=False, fixedChunkSize=i)
        maze_feature_extractor = MinigridMazeFeatureExtractor(tasks)
        dataset_creator = DatasetCreator(maze_feature_extractor, grammar)

        dataset_file_name = os.path.join(
            output_dir, f'iter_{i}-ec_iterator_T5-gen_data_{random_programs}.npy')
        if os.path.exists(dataset_file_name):
            dataset = np.load(dataset_file_name)
            if verbose:
                print(f'loaded dataset {dataset_file_name}')
        else:
            dataset = dataset_creator.createDataset(
                tasks, random_programs, no_spaces=no_spaces, compress=compress, min_len=min_len_random_programs, max_len=max_len_random_programs)

        if output_dir:
            np.save(dataset_file_name, np.array(dataset))
            if verbose:
                print(f'saved dataset to {dataset_file_name}')

        dataset = FactoringDataset(dataset)
        trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=dataset,
                                 tokenizer=tokenizer, compute_metrics=None, data_collator=collator)
        trainer.train()

        transformers.logging.set_verbosity_error()
        testTasks = createTestDataFromTasks(
            tasks, True, no_spaces=no_spaces, compress=compress)
        stats, solved_tasks = check_test_tasks(
            testTasks, grammar, n_sampling=n_sampling, verbose=verbose)
        all_solved_tasks.update(solved_tasks)
        if output_dir:
            dataset_solved_tasks = os.path.join(
                output_dir, 'all_solved_tasks.npy')
            np.save(dataset_solved_tasks, np.array(list(all_solved_tasks)))
            if verbose:
                print(f'saved all solved tasks to {dataset_solved_tasks}')

        log_solved_tasks(stats, i, verbose=True)

        # compress data of generated frontiers
        if lib_learning:
            frontiers = generate_frontiers(testTasks, stats)
            result = ECResult(parameters={},
                              grammars=grammars,
                              taskSolutions={
                f.task: f for f in frontiers},
                recognitionModel=None, numTestingTasks=len(testTasks),
                allFrontiers={
                f.task: f for f in frontiers})
            grammar = consolidate(result, grammar, iteration=i, arity=3, aic=1.0, pseudoCounts=30.0,
                                  structurePenalty=1.5, compressor='ocaml', topK=1, CPUs=numberOfCPUs())
            grammars.append(grammar)

        results[i] = {'stats': stats, 'grammar': grammar}
        save_results(output_dir, results, stats, grammars)


if __name__ == '__main__':
    import argparse

    ################################## MAIN ########################################
    parser = argparse.ArgumentParser(description='Train CodeT5 for PCGRL')
    parser.add_argument('--config', type=str, required=False,
                        help='the path to the json config with the experiment parameters')
    parser.add_argument('--run_name', type=str, required=True,
                        help='the wandb run name and the folder where to save files')
    parser.add_argument('--api_key', type=str, required=False, default='',
                        help='the wandb api key')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start_iter', type=int, default=5,
                        help='the start iter sequence')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-l', '--sequence_lengths', metavar='N', type=int, nargs='+',
                        help='define sequence lengthts, only for eval', required=False)
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = "T5-Minigrid-Maze"
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_API_KEY'] = args.api_key
    run_name = args.run_name
    verbose = args.verbose
    only_eval = args.eval
    resume = args.resume
    start_iter = args.start_iter
    lib_learning = True
    no_spaces = True
    compress = False

    output_dir = f'/home/ma/e/eberhardinger/workspaces/T5-experimens/{run_name}/'
    os.makedirs(output_dir, exist_ok=True)

    data_file = "/home/ma/e/eberhardinger/workspaces/ec/dreamcoder/domains/perfect-maze-minigrid/collected_data/2022-12-10T15:32:12.354349.npy"
    data = np.load(data_file, allow_pickle=True)
    parsed_data = parseData(data)

    # we need to call this one time, as otherwise we get parsing errors
    grammar = Grammar.uniform(basePrimitives())
    if resume or only_eval or start_iter > 5:
        checkpoint_dir = get_latest_checkpoint_path(output_dir)
        model = T5ForConditionalGeneration.from_pretrained(
            checkpoint_dir).to('cuda')
        tokenizer = RobertaTokenizer.from_pretrained(checkpoint_dir)
        grammar_file = os.path.join(output_dir, 'results.pkl')
        with open(grammar_file, 'rb') as handle:
            result = dill.load(handle)
        grammar = [g['grammar'] for g in result.values()][-1]
        key = list(result.keys())[-1]
        print(
            f'loaded checkpoint from {checkpoint_dir} and grammar from {grammar_file} after sequence length of {key}')
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            'Salesforce/codet5-small').to('cuda')
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
        print('created new model and use base primitives for the grammar')
    print(f'Grammar # primitives: {len(grammar.primitives)}')

    collator = LookupTableCollator(tokenizer)
    if only_eval:
        sequence_lengths = args.sequence_lengths
        for i in sequence_lengths:
            tasks = makeTasks(
                parsed_data, randomChunkSize=False, fixedChunkSize=i)
            testTasks = createTestDataFromTasks(
                tasks, True, no_spaces=no_spaces, compress=compress)
            stats = check_test_tasks(
                testTasks, grammar, n_sampling=5, verbose=verbose)
            solved = [x for x in stats if x[0] is not None]
            rate = len(solved)/len(stats) * 100
            print(f'{len(solved)}/{len(stats)} -> {rate}%')
    else:
        training_args = TrainingArguments(per_device_train_batch_size=40,
                                          gradient_accumulation_steps=3,
                                          save_steps=500,
                                          save_total_limit=3,
                                          num_train_epochs=5,
                                          output_dir=output_dir,
                                          report_to='wandb',
                                          run_name=run_name)
        ec_iterator_T5(grammar, parsed_data, training_args, start_iter=start_iter, random_programs=50000, lib_learning=lib_learning,
                       output_dir=output_dir, no_spaces=no_spaces, compress=compress, max_len_random_programs=60, verbose=verbose)
