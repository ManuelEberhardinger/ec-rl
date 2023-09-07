try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.minatar.primitives import *
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.domains.minatar.feature_extractor import CNNFeatureExtractor
from dreamcoder.domains.minatar.utilities import makeTasks
from dreamcoder.utilities import eprint
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
import gym
import numpy as np
import datetime
import random
import os
import wandb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gym.logger.set_level(40)
TABLE_DATA = []


def log_solved_tasks(stats, seq_len, table_name='solved_stats', verbose=False):
    last_rate = 0
    try:
        t = None
        for stat in stats:
            all = stat['tasks']
            top = stat['tasksHitTopDown']
            bottom = stat.get('tasksHitBottomUp', None)
            rate_top = top/all * 100
            last_rate = rate_top
            if bottom is not None:
                rate_bottom = bottom/all * 100
                total = stat.get('totalHits')
                total_rate = total/all

                if verbose:
                    print(f'{top}/{all} -> {rate_top}%')
                    print(f'{bottom}/{all} -> {rate_bottom}%')

                TABLE_DATA.append(
                    [seq_len, total_rate, rate_top, rate_bottom, total, top, bottom, all])
                t = wandb.Table(columns=["sequence length", "rate total", "rate top down", "rate bottom up",
                                         "total", "top down", "bottom up", "all"], data=TABLE_DATA)
            else:
                if verbose:
                    print(f'{top}/{all} -> {rate_top}%')
                TABLE_DATA.append([seq_len, rate_top, top, all])
                t = wandb.Table(
                    columns=["sequence lenght", "rate top down", "top down", "all"], data=TABLE_DATA)
        if t is not None:
            wandb.log({table_name: t})
    except Exception as e:
        if verbose:
            print(e)
            print(TABLE_DATA)

    return last_rate


def save_results(output_dir, path, i):
    if i is None:
        i = 0
    try:
        wandb.save(path)
        wandb.save(f"{output_dir}_primitives_{i}_depth.pdf.pdf")
        wandb.save(f"{output_dir}_primitives_{i}_unordered.pdf.pdf")
    except Exception as e:
        print(e)

    return True


def minatar_options(parser):
    parser.add_argument("--env_name", required=True, type=str)
    parser.add_argument('--run_name', type=str, required=True,
                        help='the wandb run name and the folder where to save files')
    parser.add_argument('--api_key', type=str, required=False, default='',
                        help='the wandb api key')


def run_dreamcoder(env_name, arguments, parsed_data, outputPrefix, chunkSize, resumeIteration=None, iterations=None):
    random.seed(42)
    # tasks = makeTasks(parsed_data, env_name, randomChunkSize=False,
    #                  fixedChunkSize=chunkSize, tolist=True)[:300]
    test_task_feature_extractor = CNNFeatureExtractor([], env_name=env_name)
    tasks = test_task_feature_extractor.create_test_tasks(
        chunkSize, tolist=True)
    if len(tasks) > 100:
        tasks = random.sample(tasks, 100)

    eprint("Got %d tasks..." % len(tasks))

    if len(tasks) == 0:
        return None

    # Create grammar
    grammar = Grammar.uniform(basePrimitives(env_name))

    arguments.pop('resume', None)
    arguments.pop('iterations', None)
    # EC iterate
    generator = ecIterator(grammar,
                           tasks,
                           outputPrefix=outputPrefix,
                           resume=resumeIteration,
                           iterations=iterations,
                           env_name=env_name,
                           **arguments)

    last_rate = 0.0
    for result, path, stats in generator:
        print('ecIterator chunkSize {}'.format(chunkSize))
        save_results(outputPrefix, path, resumeIteration)
        last_rate = log_solved_tasks(stats, chunkSize, verbose=True)
    added_primitives = abs(
        len(result.grammars[-1].primitives) - len(result.grammars[-2].primitives))
    if resumeIteration is None:
        return 1, last_rate, added_primitives

    return int(resumeIteration) + 1, last_rate, added_primitives


def evaluate_dreamcoder(env_name, arguments, parsed_data, outputPrefix, chunkSize, resumeIteration=None, iterations=None):
    random.seed(42)
    tasks = makeTasks(parsed_data, env_name, randomChunkSize=False,
                      fixedChunkSize=chunkSize, tolist=True)

    eprint("Got %d tasks..." % len(tasks))

    if len(tasks) == 0:
        return None

    # Create grammar
    grammar = Grammar.uniform(basePrimitives(env_name))

    arguments.pop('resume', None)
    arguments.pop('iterations', None)
    arguments.pop('noConsolidation', None)
    # EC iterate
    generator = ecIterator(grammar,
                           tasks,
                           outputPrefix=outputPrefix,
                           resume=resumeIteration,
                           iterations=iterations,
                           noConsolidation=True,
                           env_name=env_name,
                           **arguments)

    last_rate = 0.0
    for result, path, stats in generator:
        print('ecIterator chunkSize {}'.format(chunkSize))
        save_results(outputPrefix, path, resumeIteration)
        last_rate = log_solved_tasks(
            stats, chunkSize, table_name='evaluation stats', verbose=True)
    added_primitives = abs(
        len(result.grammars[-1].primitives) - len(result.grammars[-2].primitives))
    if resumeIteration is None:
        return 1, last_rate, added_primitives

    return int(resumeIteration) + 1, last_rate, added_primitives


if __name__ == '__main__':

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
        featureExtractor=CNNFeatureExtractor,
        maximumFrontier=10,
        CPUs=numberOfCPUs(),
        pseudoCounts=30.0,
        cuda=False,
        extras=minatar_options)
    env_name = args.pop('env_name')
    timestamp = datetime.datetime.now().isoformat()
    run_name = args.pop('run_name')
    outputDirectory = "/home/ma/e/eberhardinger/workspaces/ec/experimentOutputs/%s/%s/%s" % (
        env_name, run_name,  timestamp)
    os.system("mkdir -p %s" % outputDirectory)

    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_API_KEY'] = args.pop('api_key')

    wandb.init(
        project=f"DreamCoder-MinAtar-{env_name}", config=args, name=run_name)
    # args.pop('run_name')
    data = np.load(
        f"/home/ma/e/eberhardinger/workspaces/gymnax-blines/notebooks/{env_name}/rollouts.npy", allow_pickle=True)[0]
    iteration = None
    if iteration is not None:
        iterations = iteration + 1
    else:
        iterations = 1

    chunk_size = 3
    improvement = True
    while improvement:
        eprint('-'*200)
        eprint('-'*200)
        eprint('-'*200)
        eprint('resume from iteration:', iteration)

        if iteration is not None:
            iteration = str(iteration)

        tmp, last_rate, added_primitives = run_dreamcoder(env_name, args, data,
                                                          f"{outputDirectory}/{env_name}",
                                                          chunk_size,
                                                          iterations=iterations,
                                                          resumeIteration=iteration)
        print('last_rate', last_rate, 'added_primitives', added_primitives)
        if last_rate >= 10.0:
            chunk_size += 1
        elif last_rate < 10.0 and added_primitives == 0:
            improvement = False
            break

        if tmp is None:
            # no tasks so no iteration was run
            continue
        else:
            # we always need to add 1 iteration so that dreamcoder actually runs
            # this way dreamcoder runs until there are no files left in the data dir
            iterations += 1
            iteration = tmp

    # print('Evalution on jax data')
    # iteration = None
    # iterations = 1
    # for seq_length in range(3, 31):
    #     eprint('-'*200)
    #     eprint('-'*200)
    #     eprint('-'*200)
    #     eprint('Eval resume from iteration:', iteration)

    #     if iteration is not None:
    #         iteration = str(iteration)

    #     tmp, last_rate, added_primitives = evaluate_dreamcoder(env_name, args, data,
    #                                                            f"{outputDirectory}/{env_name}",
    #                                                            seq_length,
    #                                                            iterations=iterations,
    #                                                            resumeIteration=iteration)
    #     # we always need to add 1 iteration so that dreamcoder actually runs
    #     # this way dreamcoder runs until there are no files left in the data dir
    #     iterations += 1
    #     iteration = tmp
