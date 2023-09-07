import os
import numpy as np
import dill
import wandb
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module


from dreamcoder.domains.minatar.primitives import basePrimitives
from dreamcoder.grammar import Grammar
from dreamcoder.domains.minatar.utils_text_encoder import get_latest_checkpoint_path, makeTasks, createTestDataFromTasks, check_test_tasks, Collator, ec_iterator_T5
from dreamcoder.domains.minatar.feature_extractor import MinatarFeatureExtractorToken
from transformers import RobertaTokenizer, T5ForConditionalGeneration, TrainingArguments

# copied from utils_text_encoder because of cyclic imports
# generate n samples with t temperature


def generate_samples_with_temp(model, tokenizer, txt, n_samples, temp, device='cuda'):
    # we do 20 batches
    bs = 50
    outer_loop = int(n_samples / bs)
    results = []
    for i in range(outer_loop):
        to_tokenizer = [txt for j in range(bs)]
        outputs = model.generate(tokenizer(to_tokenizer, return_tensors='pt', padding=True).input_ids.to(
            device), do_sample=True, max_length=128, temperature=temp)
        results += tokenizer.batch_decode(outputs, skip_special_tokens=True)
        temp = temp - 0.005
    return results


if __name__ == '__main__':
    import argparse

    ################################## MAIN ########################################
    parser = argparse.ArgumentParser(description='Train LibT5 for MinAtar')
    parser.add_argument('--config', type=str, required=False,
                        help='the path to the json config with the experiment parameters')
    parser.add_argument('--num_programs', type=int, required=True,
                        help='the number of random programs')
    parser.add_argument('--start_iter', type=int, default=0,
                        help='the start iter sequence')
    parser.add_argument('--start_seqlength', type=int, default=3,
                        help='the start sequence length')
    parser.add_argument('--n_sampling', type=int, default=1000,
                        help='the number of programs to sample for each task')
    parser.add_argument('--env_name', type=str, required=True,
                        help='the minatar env name')
    parser.add_argument('--device', type=str, default='cuda',
                        help='the device: cuda or cpu')
    parser.add_argument('--api_key', type=str, required=False, default='',
                        help='the wandb api key')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-l', '--sequence_lengths', metavar='N', type=int, nargs='+',
                        help='define sequence lengthts, only for eval', required=False)
    args = parser.parse_args()

    env_name = args.env_name
    wandb_project_name = f"T5-{env_name}-new_DSL"
    os.environ["WANDB_PROJECT"] = wandb_project_name
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_API_KEY'] = args.api_key
    verbose = args.verbose
    only_eval = args.eval
    resume = args.resume
    lib_learning = True
    no_spaces = True
    compress = False
    num_programs = args.num_programs
    start_iter = args.start_iter
    seq_length = args.start_seqlength
    device = args.device
    n_sampling = args.n_sampling
    run_name = f'allActions-{num_programs}p-{n_sampling}s'
    output_dir = f'/home/ma/e/eberhardinger/workspaces/{wandb_project_name}/{env_name}/{run_name}'
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(
        f"/home/ma/e/eberhardinger/workspaces/gymnax-blines/notebooks/{env_name}/rollouts.npy", allow_pickle=True)[0]

    # we need to call this one time, as otherwise we get parsing errors
    primitives = basePrimitives(env_name)
    grammar = Grammar.uniform(primitives)
    print('start grammar:', grammar)
    if resume or only_eval or start_iter > 0:
        checkpoint_dir = get_latest_checkpoint_path(output_dir)
        model = T5ForConditionalGeneration.from_pretrained(
            checkpoint_dir).to(device)
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
        grammar_file = os.path.join(
            output_dir, 'iter-0_seqlength-3/results.pkl')
        with open(grammar_file, 'rb') as handle:
            result = dill.load(handle)
        grammar = [g['grammar'] for g in result.values()][-1]
        key = list(result.keys())[-1]
        print(
            f'loaded checkpoint from {checkpoint_dir} and grammar from {grammar_file} after sequence length of {key}')
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            'Salesforce/codet5-small').to(device)
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
        print('created new model and use base primitives for the grammar')
    print(f'Grammar # primitives: {len(grammar.primitives)}')
    feature_extractor = MinatarFeatureExtractorToken(
        env_name=env_name, max_steps=500, no_spaces=no_spaces, compress=compress)
    collator = Collator(tokenizer)
    if only_eval:
        sequence_lengths = args.sequence_lengths
        for i in sequence_lengths:
            tasks = makeTasks(
                data, env_name, randomChunkSize=False, fixedChunkSize=i)
            testTasks = createTestDataFromTasks(feature_extractor, tasks, True)
            stats, solved_tasks, solved = check_test_tasks(testTasks, grammar, lambda x, y, z: generate_samples_with_temp(
                model, tokenizer, x, y, z, device=device), n_sampling=1000, verbose=verbose)
            rate = solved/len(stats) * 100
            print(f'{solved}/{len(stats)} -> {rate}%')
    else:
        training_args = TrainingArguments(per_device_train_batch_size=10,
                                          gradient_accumulation_steps=22,
                                          save_steps=100,
                                          save_total_limit=3,
                                          num_train_epochs=5,
                                          output_dir=output_dir,
                                          report_to='wandb',
                                          run_name=run_name)
        ec_iterator_T5(env_name, model, tokenizer, collator, feature_extractor, grammar, data, training_args, i=start_iter, seq_length=seq_length, device=device, n_sampling=n_sampling,
                       random_programs=num_programs, lib_learning=lib_learning, output_dir=output_dir, sim_prior=False, verbose=verbose)
