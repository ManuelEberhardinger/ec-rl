import wandb
import os
import signal
from dreamcoder.primitiveGraph import graphPrimitivesFromGrammar
from dreamcoder.frontier import *
from dreamcoder.dreamcoder import *
from tqdm.auto import tqdm
from dreamcoder.domains.minatar.primitives import basePrimitives, tmap, taction
from dreamcoder.domains.minatar.feature_extractor import convert_to_task_input
from dreamcoder.task import Task
import traceback
import multiprocessing
from functools import partial


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
                program=self.grammar.closedLikelihoodSummary(frontier.task.request, e.program),
                logLikelihood=e.logLikelihood,
                logPrior=e.logPrior) for e in frontier],
            task=frontier.task)

    def sampleHelmholtz(self, requests, statusUpdate=None, seed=None):
        if seed is not None:
            random.seed(seed)
        request = random.choice(requests)

        program = self.generativeModel.sample(request, maximumDepth=30, maxAttempts=100)
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
        try:
            program = self.generativeModel.sample(request, maximumDepth=6, maxAttempts=100)
            if program is None:
                return None, None, None
            task = self.featureExtractor.taskOfProgram(program, request, min_len=min_len, max_len=max_len)
            if statusUpdate is not None:
                flushEverything()
            if task is None:
                return None, None, None
            features = self.featureExtractor.featuresOfTask(task)
            return task, features, program
        except Exception as e:
            eprint('sampleProgramWithTask', e)
            traceback.print_exc()
            return None, None, None

    def sampleManyProgramsWithTasks(self, tasks, N, min_len, max_len, verbose=True):
        if verbose:
            eprint("Sampling %d programs from the prior..." % N)
        flushEverything()
        requests = list({t.request for t in tasks})
        frequency = N / 50
        startingSeed = random.random()

        # Sequentially for ensemble training.
        data = []
        for n in tqdm(range(N)):
            data_point = self.sampleProgramWithTask(requests, min_len, max_len,
                                                    statusUpdate='.' if n % frequency == 0 else None,
                                                    seed=startingSeed + n)
            data.append(data_point)

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

    def createDataset(self, tasks, N, with_tasks=False, min_len=5, max_len=20, verbose=False):
        dataset = []
        data = self.sampleManyProgramsWithTasks(tasks, N, min_len, max_len, verbose=verbose)
        for task, features, program in data:
            if with_tasks:
                dataset.append((features, str(program), task))
            else:
                dataset.append((features, str(program)))

        return dataset


def all_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def makeTasks(data, env_name, rand_min=10, rand_max=50, randomChunkSize=True, fixedChunkSize=None, tolist=False):
    assert randomChunkSize or (not randomChunkSize and fixedChunkSize)

    if randomChunkSize:
        chunkSize = random.randint(rand_min, rand_max)
    else:
        chunkSize = fixedChunkSize

    tasks = []
    examples = []
    part = 0
    states, actions, reward = data

    state_action_pairs = list(zip(states, actions))

    for i in range(len(state_action_pairs) - chunkSize):
        examples = []
        for state, action in state_action_pairs[i: i + chunkSize]:
            input_ex = (convert_to_task_input(state, jax_data=True, tolist=tolist),)
            output_ex = int(action)
            examples.append((input_ex, output_ex))

        # we check that the chosen actions are not all the same
        # otherwise it is too easy to find a program if all actions/output examples are the same
        # this results in programs such as (lambda (lambda forward-action))
        all_chosen_actions = list(zip(*examples))[1]
        if not all_equal(all_chosen_actions):
            tasks.append(Task(f'{env_name} size {chunkSize} part {part}',
                              arrow(tmap, taction), examples))
            part += 1
            # we reset examples and add new chunkSize tasks
            if randomChunkSize:
                chunkSize = random.randint(rand_min, rand_max)
            else:
                chunkSize = fixedChunkSize

    if len(examples) > 3:
        all_chosen_actions = list(zip(*examples))[1]
        if not all_equal(all_chosen_actions):
            tasks.append(Task(f'{env_name} size {chunkSize} part {part}',
                         arrow(tmap, taction), examples))

    if len(tasks) < 200:
        # random state action sequences for more tasks
        if randomChunkSize:
            chunkSize = random.randint(rand_min, rand_max)
        else:
            chunkSize = fixedChunkSize
        examples = []
        part = 0
        already_sampled = []
        while len(state_action_pairs) - len(already_sampled) > chunkSize:
            try:
                curr_sample = random.sample([x for x in state_action_pairs if str(x) not in already_sampled], chunkSize)
            except ValueError as e:
                print('makeTasks', e)
                break

            for state, action in curr_sample:
                input_ex = (convert_to_task_input(state, jax_data=True, tolist=tolist),)
                output_ex = int(action)
                examples.append((input_ex, output_ex))

                if chunkSize > 0 and chunkSize == len(examples):
                    # we check that the chosen actions are not all the same
                    # otherwise it is too easy to find a program if all actions/output examples are the same
                    # this results in programs such as (lambda (lambda forward-action))
                    all_chosen_actions = list(zip(*examples))[1]
                    if not all_equal(all_chosen_actions):
                        tasks.append(Task(f'{env_name} size {chunkSize} random {part}', arrow(
                            tmap, taction), examples))
                        part += 1
                        # we reset examples and add new chunkSize taskss
                        examples = []
                        # random state action sequences for more tasks
                        if randomChunkSize:
                            chunkSize = random.randint(rand_min, rand_max)
                        else:
                            chunkSize = fixedChunkSize

            already_sampled += [str(x) for x in curr_sample]

    print(f'Created {len(tasks)} tasks with {fixedChunkSize} chunk size')
    return tasks


class WandbLogger():

    def __init__(self):
        self.table_data = []

    def log_solved_tasks(self, stats, iter, seq_len, added_functions, verbose=False):
        solved = [x for x in stats if x[0] is not None]
        rate = len(solved)/len(stats) * 100
        if verbose:
            print(f'{len(solved)}/{len(stats)} -> {rate}%')
        try:
            self.table_data.append([iter, seq_len, rate, len(solved), len(stats), added_functions])
            t = wandb.Table(columns=["iter", "sequence length", "rate", "solved",
                            "all", "added functions"], data=self.table_data)
            wandb.log({'solved_stats': t})
        except Exception as e:
            if verbose:
                print(e)

        return solved

    def save_results(self, output_dir, results, stats, grammars, seq_length):
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
                graphPrimitivesFromGrammar(grammars, f"{output_dir}/T5_primitives_{i}_{seq_length}_")
                wandb.save(json_path)
                wandb.save(path)
                wandb.save(f"{output_dir}/T5_primitives_{i}_{seq_length}_depth.pdf.pdf")
                wandb.save(f"{output_dir}/T5_primitives_{i}_{seq_length}_unordered.pdf.pdf")
            except Exception as e:
                print(e)

        return True


def get_latest_checkpoint_path(output_dir):
    checkpoint_dirs = list(filter(lambda x: os.path.isdir(os.path.join(output_dir, x))
                           and 'checkpoint-' in x, os.listdir(output_dir)))
    checkpoint_dir = sorted(checkpoint_dirs, key=lambda x: int(x.replace('checkpoint-', '')))[-1]
    checkpoint_dir = os.path.join(output_dir, checkpoint_dir)
    return checkpoint_dir


def run_on_input_examples(task, grammar, program, verbose=False):
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
            eprint(f'run_on_input_examples > logLikelihood failed for: {program}')
        # not a correctly typed program
        return None

    return logPrior


def run_on_input_examples_list(task, grammar, programs, verbose=False):
    log_priors = []
    for program in programs:
        try:
            program = Program.parse(program)
            for inp, out in task.examples:
                # output ex is the action
                pred = runWithTimeout(lambda: program.runWithArguments(inp), None)
                if verbose:
                    eprint('Input:', inp, 'Out:', out, 'Pred:', pred)
                if out != pred:
                    raise Exception("out != pred")

        except Exception as e:
            if verbose:
                eprint(e)
            log_priors.append(None)
            continue
        try:
            logPrior = grammar.logLikelihood(task.request, program)
        except:
            if verbose:
                eprint(f'run_on_input_examples > logLikelihood failed for: {program}')
            # not a correctly typed program
            log_priors.append(None)
            continue

        log_priors.append(logPrior)
    return log_priors


def test_programs_on_task(task, grammar, generate_sample_fn, n=5, temp=1.0, verbose=False, use_multiprocess=True):
    progs = np.array(generate_sample_fn(task[0], n, temp))
    num_progs = len(progs)
    found_progs = []

    if use_multiprocess:
        def init_worker():
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        num_workers = int(multiprocessing.cpu_count())

        chunks = np.array_split(progs, num_workers)

        try:
            pool = multiprocessing.Pool(num_workers, init_worker)
            fn = partial(run_on_input_examples_list, task[1], grammar)
            all_results = pool.map(fn, chunks)
            pool.close()

            row = 0
            for i, results in enumerate(all_results):
                for idx, log_prior in enumerate(results):
                    if log_prior is not None:
                        found_progs.append((Program.parse(progs[idx + row]), log_prior))
                row += len(chunks[i])
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

        # for i in range(0, num_progs, num_workers):
        #     try:
        #         end = min(i+num_workers, num_progs)
        #         pool = multiprocessing.Pool(num_workers, init_worker)
        #         fn = partial(run_on_input_examples, task[1], grammar)
        #         results = pool.map(fn, progs[i:end])
        #         pool.close()

        #         for idx, log_prior in enumerate(results):
        #             if log_prior is not None:
        #                 found_progs.append((Program.parse(prog[idx + i]), log_prior))
        #     except KeyboardInterrupt:
        #         pool.terminate()
        #         pool.join()
    else:
        for prog in progs:
            if verbose:
                eprint(prog)
            log_prior = run_on_input_examples(task[1], grammar, prog, verbose=verbose)
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


def check_test_tasks(testTasks, grammar, generate_sample_fn, n_sampling=100, verbose=False):
    stats = []
    solved_tasks = set()
    solved = 0
    for tt in (pbar := tqdm(testTasks)):
        p, n = test_programs_on_task(tt, grammar, generate_sample_fn, n=n_sampling, verbose=verbose)
        stats.append((p, n))
        if 'random' not in str(tt[1]):
            solved_tasks.add(tuple([str(tt[1]), str(p)]))

        if p is not None:
            solved += 1
        pbar.set_description(f"Solved: {solved}")
    return stats, solved_tasks, solved


def createTestDataFromTasks(feature_extractor, tasks, with_tasks=False):
    dataset = []
    for task in tasks:
        features = feature_extractor.featuresOfTask(task)

        if with_tasks:
            dataset.append((features, task))
        else:
            dataset.append((features))

    return dataset


def get_productions(productions):
    new_prods = []
    prod_strings = []

    productions.sort(key=lambda x: len(str(x[-1])))
    for production in productions:
        l = production[0]
        p = production[-1]
        p_str = str(p)

        if p_str not in prod_strings:
            new_prods.append((l, p.infer(), Program.parse(p_str)))
            prod_strings.append(p_str)
            if '#' in p_str and '-action' in p_str:
                new_p_str = None
                # is inventend
                if 'left-action' in p_str:
                    new_p_str = p_str.replace('left-action', 'right-action')
                elif 'right-action' in p_str:
                    new_p_str = p_str.replace('right-action', 'left-action')
                elif 'up-action' in p_str:
                    new_p_str = p_str.replace('up-action', 'down-action')
                elif 'down-action' in p_str:
                    new_p_str = p_str.replace('down-action', 'up-action')
                if new_p_str is not None:
                    new_prods.append((l, p.infer(), Program.parse(new_p_str)))
                    prod_strings.append(new_p_str)

    return new_prods


def convert_to_invented_with_name(grammar):
    new_primitives = []
    i = 0
    for ll, tp, p in grammar.productions:
        if isinstance(p, Invented):
            new_primitives.append((ll, tp, InventedWithName(f"_f{i}", p.body)))
            i += 1
        else:
            new_primitives.append((ll, tp, p))
    return Grammar(grammar.logVariable, new_primitives, continuationType=grammar.continuationType)
