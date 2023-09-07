import os
import numpy as np
import random

try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.frontier import *
from dreamcoder.dreamcoder import *
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.domains.minatar.utilities import *
import transformers
from transformers import Trainer
from torch.utils.data import Dataset


class VisionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.sequence_length = 512

    def __call__(self, batch):
        # entry[1] is the spec, need to call repr to turn it into a string. entry[0] is the prog_str already
        ret = {"pixel_values": torch.stack([entry['pixel_values'] for entry in batch]),
               "labels": self.tokenizer([entry['labels'] for entry in batch], padding='max_length', truncation=False, return_tensors='pt', max_length=self.sequence_length).input_ids}
        return ret


class VisionDataset(Dataset):
    def __init__(self, dataset_itself):
        self.data = dataset_itself

    def __getitem__(self, idx):
        data = self.data[idx]
        return {'pixel_values': data[0], 'labels': data[1]}

    def __len__(self):
        return len(self.data)


def generate_samples_with_temp(model, tokenizer, txt, n_samples, temp):
    batch = [[txt] for i in range(n_samples)]
    outputs = model.generate(torch.stack([entry[0] for entry in batch]).to(
        'cuda'), do_sample=True, max_length=128, temperature=temp)
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return results


def ec_iterator_T5(env_name, model, collator, feature_extractor_class, grammar, parsed_data, training_args, n_sampling=100, random_programs=100000, output_dir='', min_len_random_programs=5, max_len_random_programs=20, lib_learning=True, verbose=False):
    results = {}
    all_solved_tasks = set()
    grammars = [grammar]

    wandb_logger = WandbLogger()

    for i in range(5, 60):
        if verbose:
            print(f'Start iteration {i}')
            transformers.logging.set_verbosity_info()

        tasks = []
        for j in range(i, i + 1):
            tasks += makeTasks(parsed_data, env_name, randomChunkSize=False, fixedChunkSize=j)
        feature_extractor = feature_extractor_class(tasks, env_name=env_name, max_steps=260)
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

        dataset = VisionDataset(dataset)
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                          data_collator=collator, compute_metrics=None)
        trainer.train()

        transformers.logging.set_verbosity_error()
        testTasks = createTestDataFromTasks(feature_extractor, tasks, True)
        stats, solved_tasks, solved = check_test_tasks(
            testTasks, grammar, generate_samples_with_temp, n_sampling=n_sampling, verbose=verbose)
        all_solved_tasks.update(solved_tasks)
        if output_dir:
            dataset_solved_tasks = os.path.join(output_dir, 'all_solved_tasks.npy')
            np.save(dataset_solved_tasks, np.array(list(all_solved_tasks)))
            if verbose:
                print(f'saved all solved tasks to {dataset_solved_tasks}')

        wandb_logger.log_solved_tasks(stats, i, verbose=True)

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
        wandb_logger.save_results(output_dir, results, stats, grammars)
