try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.minigrid.primitives import *
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.domains.minigrid.nn_model_action import MinigridMazeFeatureExtractor
from bin.maze_T5 import makeTasks, parseData
from dreamcoder.utilities import eprint
from dreamcoder.dreamcoder import ecIterator
import gym
import numpy as np
import datetime
import random
import os
import wandb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gym.logger.set_level(40)
TABLE_DATA = []


def get_max_seq_len(group_df):
    max_len = 0

    for key, item in group_df:
        seq_len = group_df.get_group(key).shape[0]
        if seq_len > max_len:
            max_len = seq_len

    return max_len


def log_solved_tasks(stats, seq_len, verbose=False):
    try:
        t = None
        for stat in stats:
            all = stat['tasks']
            top = stat['tasksHitTopDown']
            bottom = stat.get('tasksHitBottomUp', None)
            rate_top = top/all * 100

            if bottom is not None:
                rate_bottom = bottom/all * 100
                total = stat.get('totalHits')
                total_rate = total/all

                if verbose:
                    print(f'{top}/{all} -> {rate_top}%')
                    print(f'{bottom}/{all} -> {rate_bottom}%')

                TABLE_DATA.append(
                    [seq_len, total_rate, rate_top, rate_bottom, total, top, bottom, all])
                t = wandb.Table(columns=["sequence lenght", "rate total", "rate top down", "rate bottom up",
                                         "total", "top down", "bottom up", "all"], data=TABLE_DATA)
            else:
                if verbose:
                    print(f'{top}/{all} -> {rate_top}%')
                TABLE_DATA.append([seq_len, rate_top, top, all])
                t = wandb.Table(
                    columns=["sequence lenght", "rate top down", "top down", "all"], data=TABLE_DATA)
        if t is not None:
            wandb.log({'solved_stats': t})
    except Exception as e:
        if verbose:
            print(e)
            print(TABLE_DATA)


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


def minigrid_options(parser):
    parser.add_argument("--data_file",
                        default='/home/ma/e/eberhardinger/workspaces/ec/dreamcoder/domains/perfect-maze-minigrid/collected_data/2022-12-10T15:32:12.354349.npy', type=str)
    parser.add_argument('--run_name', type=str, required=True,
                        help='the wandb run name and the folder where to save files')
    parser.add_argument('--api_key', type=str, required=False, default='',
                        help='the wandb api key')


def run_dreamcoder(arguments, parsed_data, outputPrefix, chunkSize, resumeIteration=None, iterations=None):
    random.seed(42)
    tasks = makeTasks(parsed_data, randomChunkSize=False,
                      fixedChunkSize=chunkSize)
    if len(tasks) > 50:
        tasks = random.sample(tasks, 50)

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
                           outputPrefix=outputPrefix,
                           resume=resumeIteration,
                           iterations=iterations,
                           **arguments)
    for result, path, stats in generator:
        print('ecIterator chunkSize {}'.format(chunkSize))
        save_results(outputPrefix, path, resumeIteration)
        log_solved_tasks(stats, chunkSize, verbose=True)

    if resumeIteration is None:
        return 1

    return int(resumeIteration) + 1


if __name__ == '__main__':
    g = Grammar.uniform(basePrimitives())
    # p = g.sample(arrow(tmap, tint, arrow(arrow(arrow(arrow(tmap, tint, tint, tobj), tobj, tbool)))),
    #             maximumDepth = 100, maxAttempts = 100)
    # print(p)

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "/home/ma/e/eberhardinger/workspaces/ec/experimentOutputs/perfect-maze/%s" % timestamp
    os.system("mkdir -p %s" % outputDirectory)

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
        cuda=False,
        extras=minigrid_options)
    args.pop('resume')
    args.pop('iterations')

    os.environ["WANDB_PROJECT"] = "DreamCoder-Maze"
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_API_KEY'] = args.pop('api_key')

    wandb.init(project="DreamCoder-Maze",
               config=args, name=args.pop('run_name'))

    data_file = args.pop('data_file')
    data = np.load(data_file, allow_pickle=True)
    parsed_data = parseData(data)
    max_iterations = get_max_seq_len(parsed_data)
    eprint(f'longest action sequence: {max_iterations}')

    iteration = None
    if iteration is not None:
        iterations = iteration + 1
    else:
        iterations = 1

    for chunk_size in range(8, max_iterations):
        eprint('-'*200)
        eprint('-'*200)
        eprint('-'*200)
        eprint('resume from iteration:', iteration)

        if iteration is not None:
            iteration = str(iteration)

        tmp = run_dreamcoder(args, parsed_data,
                             "%s/maze" % outputDirectory,
                             chunk_size,
                             iterations=iterations,
                             resumeIteration=iteration)

        if tmp is None:
            # no tasks so no iteration was run
            continue
        else:
            # we always need to add 1 iteration so that dreamcoder actually runs
            # this way dreamcoder runs until there are no files left in the data dir
            iterations += 1
            iteration = tmp
