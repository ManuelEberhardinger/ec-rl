from torch.utils.data import Dataset
from dreamcoder.frontier import *
from dreamcoder.dreamcoder import *
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.domains.minatar.utilities import *
import transformers
from transformers import Seq2SeqTrainer
import torch


class TokenDataset(Dataset):
    def __init__(self, dataset_itself):
        self.data = dataset_itself

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 512

    def __call__(self, batch):
        # entry[1] is the spec, need to call repr to turn it into a string. entry[0] is the prog_str already
        ret = {"input_ids": self.tokenizer([entry[0] for entry in batch], padding='max_length',  truncation=True, max_length=self.max_length, return_tensors='pt').input_ids,
               "labels": self.tokenizer([entry[1] for entry in batch], padding='max_length',  truncation=True, max_length=self.max_length, return_tensors='pt').input_ids}
        return ret


# generate n samples with t temperature
def generate_samples_with_temp(model, tokenizer, txt, n_samples, temp, device='cuda'):
    # we do 20 batches
    outer_loop = int(n_samples / 20)
    results = []
    for i in range(outer_loop):
        to_tokenizer = [txt for j in range(20)]
        outputs = model.generate(tokenizer(to_tokenizer, return_tensors='pt', padding=True).input_ids.to(
            device), do_sample=True, max_length=128, temperature=temp)
        results += tokenizer.batch_decode(outputs, skip_special_tokens=True)
        temp = temp - 0.005
    return results


def ec_iterator_T5(env_name, model, tokenizer, collator, feature_extractor, grammar, parsed_data, training_args, device='cuda', i=0,
                   seq_length=3, n_sampling=100, random_programs=100000, output_dir='', min_len_random_programs=5,
                   max_len_random_programs=20, lib_learning=True, sim_prior=False, verbose=False):
    results = {}
    all_solved_tasks = set()
    grammars = [grammar]
    wandb_logger = WandbLogger()
    while seq_length < 20:
        if output_dir:
            if '_seqlength-' in output_dir:
                output_dir, _ = os.path.split(output_dir)
            output_dir = os.path.join(
                output_dir, f'iter-{i}_seqlength-{seq_length}')
            print('output_dir', output_dir)
            os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print(f'Start iteration {i}')
            transformers.logging.set_verbosity_info()

        # old test task creation
        # tasks = makeTasks(parsed_data, env_name, randomChunkSize=False, fixedChunkSize=seq_length)
        tasks = feature_extractor.create_test_tasks(seq_length)
        dataset_creator = DatasetCreator(feature_extractor, grammar)

        dataset_file_name = os.path.join(
            output_dir, f'iter_{i}-ec_iterator_T5-gen_data_{random_programs}.npy')
        if os.path.exists(dataset_file_name):
            dataset = np.load(dataset_file_name)
            if verbose:
                print(f'loaded dataset {dataset_file_name}')
        else:
            dataset = dataset_creator.createDataset(
                tasks, random_programs, min_len=min_len_random_programs, max_len=max_len_random_programs)

        if output_dir:
            np.save(dataset_file_name, np.array(dataset))
            if verbose:
                print(f'saved dataset to {dataset_file_name}')

        dataset = TokenDataset(dataset)
        torch.cuda.empty_cache()
        trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=dataset,
                                 compute_metrics=None, data_collator=collator)
        trainer.train()

        transformers.logging.set_verbosity_error()
        if len(tasks) > 300:
            tasks = random.sample(tasks, 300)
        testTasks = createTestDataFromTasks(feature_extractor, tasks, True)
        stats, solved_tasks, solved = check_test_tasks(
            testTasks, grammar, lambda x, y, z: generate_samples_with_temp(model, tokenizer, x, y, z, device=device), n_sampling=n_sampling, verbose=verbose)
        all_solved_tasks.update(solved_tasks)
        if output_dir:
            dataset_solved_tasks = os.path.join(
                output_dir, 'all_solved_tasks.npy')
            np.save(dataset_solved_tasks, np.array(list(all_solved_tasks)))
            if verbose:
                print(f'saved all solved tasks to {dataset_solved_tasks}')

        added_functions = 0
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

            if sim_prior:
                grammar = Grammar(grammar.logVariable, get_productions(
                    grammar.productions), continuationType=grammar.continuationType)

            added_functions = len(grammar.primitives) - \
                len(grammars[-1].primitives)
            # convert all Invented to InventedWithName so that the programs are shorter
            # grammar = convert_to_invented_with_name(grammar)
            grammars.append(grammar)

        wandb_logger.log_solved_tasks(
            stats, i, seq_length, added_functions, verbose=True)

        results[i] = {'stats': stats, 'grammar': grammar}
        wandb_logger.save_results(
            output_dir, results, stats, grammars, seq_length)

        # increase iter and check if seq length should be increased
        i += 1
        # only increase seq_length if 10 percent are solved?
        if solved/len(stats) >= 0.1:
            seq_length += 1
