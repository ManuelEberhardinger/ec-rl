import numpy as np
import random
from dreamcoder.domains.minatar.primitives import *
from dreamcoder.program import *
from dreamcoder.task import Task
import torch.nn.functional as F
import torch.nn as nn
import torch
from minatar import Environment


def dSiLU(x): return torch.sigmoid(x)*(1+x*(1-torch.sigmoid(x)))
def SiLU(x): return x*torch.sigmoid(x)


class ACNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(ACNetwork, self).__init__()
        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(
            in_features=num_linear_units, out_features=128)

        # Output layer:
        self.policy = nn.Linear(in_features=128, out_features=num_actions)
        self.value = nn.Linear(in_features=128, out_features=1)

    # As per implementation instructions, the forward function should be overwritten by all subclasses
    def forward(self, x):
        # Output from the first conv with sigmoid linear activation
        x = SiLU(self.conv(x))

        # Output from the final hidden layer with derivative of sigmoid linear activation
        x = dSiLU(self.fc_hidden(x.view(x.size(0), -1)))

        # Return policy and value outputs
        return F.softmax(self.policy(x), dim=1), self.value(x)


def get_state(s, device):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def world_dynamics(s, env, network, device):
    # network(s)[0] specifies the policy network, which we use to draw an action according to a multinomial
    # distribution over axis 1, (axis 0 iterates over samples, and is unused in this case. torch._no_grad()
    # avoids tracking history in autograd.
    with torch.no_grad():
        action = torch.multinomial(network(s)[0], 1)[0]

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime
    s_prime = get_state(env.state(), device)

    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)


def convert_to_task_input(arr, jax_data=False, tolist=False):
    if isinstance(arr, torch.Tensor):
        arr = arr.to('cpu')
    if jax_data:
        channels = arr.shape[-1]

        new_arr = np.copy(arr[:, :, 0])
        for i in range(1, channels):
            val = i + 1
            new_arr[np.array(arr[:, :, i] == 1)] = val
    else:
        arr = arr.squeeze()
        channels = arr.shape[0]

        new_arr = np.copy(arr[0, :, :])
        for i in range(1, channels):
            new_arr[np.array(arr[i, :, :] == 1)] = i + 1

    new_arr = new_arr.astype(int)
    if tolist:
        return new_arr.tolist()
    else:
        return new_arr


def rollout_episode_for_program(env, network, start_frame, seq_length, program, device):
    env.reset()
    s = get_state(env.state(), device)
    # first simulate until start_frame
    for _ in range(start_frame):
        s, action, reward, is_terminated = world_dynamics(
            s, env, network, device)
        if is_terminated:
            env.reset()
            s = get_state(env.state(), device)

    examples = []
    for _ in range(seq_length):
        obs_inp = convert_to_task_input(s)
        input_ex = (obs_inp,)
        # output ex is the action
        output_ex = runWithTimeout(
            lambda: program.runWithArguments(input_ex), None)
        examples.append((input_ex, output_ex))
        reward, done = env.act(output_ex)
        s = get_state(env.state(), device)
        if done:  # or t_counter == max_frames:
            break

    return examples


def rollout_episode(env, network, device):
    env.reset()
    s = get_state(env.state(), device)
    examples = []

    is_not_terminated = True
    while is_not_terminated:
        new_s, action, reward, is_terminated = world_dynamics(
            s, env, network, device)
        examples.append((s.to('cpu'), action.to('cpu')))
        if is_terminated:
            env.reset()
            new_s = get_state(env.state(), device)

            if len(examples) > 300:
                is_not_terminated = False

        s = new_s

    return examples


def all_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)


class MinatarFeatureExtractorBase(nn.Module):
    special = None

    def __init__(self, **kwargs):
        super().__init__()
        self.env_name = kwargs['env_name']
        self.max_steps = kwargs['max_steps']
        self.env = Environment(self.env_name)

        # Get in_channels and num_actions
        self.in_channels = self.env.state_shape()[2]
        self.num_actions = self.env.num_actions()

        # Instantiate networks, optimizer, loss and buffer
        self.network = ACNetwork(self.in_channels, self.num_actions)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            load_path = f"/home/ma/e/eberhardinger/workspaces/MinAtar/{self.env_name}_data_and_weights"
            checkpoint = torch.load(load_path, map_location=self.device)
            print(f'loaded checkpoint from {load_path}')
        except Exception as e:
            print(e)
            load_path = f"/home/ma/e/eberhardinger/workspaces/MinAtar/{self.env_name}_checkpoint_data_and_weights"
            checkpoint = torch.load(load_path, map_location=self.device)
            print(f'loaded checkpoint from {load_path}')
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.eval()
        self.network.to(self.device)

    def make_features(self, examples):
        raise NotImplementedError

    def featuresOfTask(self, t):
        inp = t.examples
        # convert the grids into a representation for VITs
        return self.make_features(inp)

    def run_program_on_env(self, program, chunkSize=None, min_len=5, max_len=20, verbose=False):
        examples = []
        if chunkSize is None:
            chunkSize = random.randint(min_len, max_len)
        start_frame = random.randint(0, self.max_steps - chunkSize)
        if verbose:
            print('sequence_length:', chunkSize, 'start_frame', start_frame)

        examples = rollout_episode_for_program(
            self.env, self.network, start_frame, chunkSize, program, self.device
        )
        return examples

    def taskOfProgram(self, p, t, min_len=5, max_len=20):
        # we need to run the sampled program on the dcd env
        # maybe we should use the seeds used for collecting the data
        examples = self.run_program_on_env(p, min_len=min_len, max_len=max_len)
        return Task("Helmhotz Dream", arrow(tmap, taction), examples)

    def create_test_tasks(self, seq_length, tolist=False):
        test_tasks = []
        state_action_pairs = rollout_episode(
            self.env, self.network, self.device
        )

        for i in range(len(state_action_pairs) - seq_length):
            examples = []
            for state, action in state_action_pairs[i: i + seq_length]:
                input_ex = (convert_to_task_input(state, tolist=tolist),)
                output_ex = int(action)
                examples.append((input_ex, output_ex))

            # we check that the chosen actions are not all the same
            # otherwise it is too easy to find a program if all actions/output examples are the same
            # this results in programs such as (lambda (lambda forward-action))
            all_chosen_actions = list(zip(*examples))[1]
            if not all_equal(all_chosen_actions):
                test_tasks.append(Task(f'{self.env_name} size {seq_length} part {i}',
                                       arrow(tmap, taction), examples))
        print(f'Created {len(test_tasks)} test tasks')
        return test_tasks


class MinatarFeatureExtractorVision(MinatarFeatureExtractorBase):
    def make_features(self, examples):
        inp_examples = [torch.from_numpy(ex[0][0]).to(
            torch.int64) for ex in examples]
        input_tensor = torch.stack(inp_examples)
        return input_tensor


class CNNFeatureExtractor(MinatarFeatureExtractorBase):

    def __init__(self, tasks, testingTasks=[], cuda=False, H=64, dcd=None, env_name=None):
        super().__init__(env_name=env_name, max_steps=500)

        self.recomputeTasks = False
        self.tasks = tasks
        self.testingTasks = testingTasks
        self.outputDimensionality = 128 * 2
        halfOutputDim = int(self.outputDimensionality/2)
        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(num_linear_units, halfOutputDim)

        self.action_embed = nn.Linear(self.num_actions, halfOutputDim)

    def forward(self, states, actions):
        # Output from the first conv with sigmoid linear activation
        x = SiLU(self.conv(states))

        # Output from the final hidden layer with derivative of sigmoid linear activation
        x = dSiLU(self.fc_hidden(x.view(x.size(0), -1)))

        # sum features over examples like in Dreaming with ARC
        # (num_examples, intermediate_dim) to (intermediate_dim)
        x = torch.sum(x, axis=0).unsqueeze(0)
        x_a = dSiLU(self.action_embed(actions))
        x_a = torch.sum(x_a, axis=0)
        in_embedded = torch.cat((x, x_a), dim=-1)
        return in_embedded

    def featuresOfTask(self, t):
        inp = t.examples
        # convert the grids into a representation for VITs
        states, actions = self.make_features(inp)
        return self(states, actions)

    def make_features(self, examples):
        inp_examples = [torch.from_numpy(
            np.array([ex[0][0]])) for ex in examples]
        input_tensor = torch.stack(inp_examples)
        out_examples = [torch.from_numpy(np.array([ex[1]])) for ex in examples]
        out_tensor = torch.stack(
            [F.one_hot(o, num_classes=self.num_actions) for o in out_examples])
        return input_tensor.to(torch.float32), out_tensor.to(torch.float32)


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


def get_inp_string_for_task(inp_string, compress=False, no_spaces=False):
    inp_string = inp_string.replace(',', '').replace('[', '').replace(
        ']', '').replace('(', '').replace(')', '').replace('array', '')
    if no_spaces:
        inp_string = inp_string.replace(' ', '')

    if compress:
        inp_string = inp_string.replace('00000', 'z')

    return inp_string.strip()


class MinatarFeatureExtractorToken(MinatarFeatureExtractorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compress = kwargs['compress']
        self.no_spaces = kwargs['no_spaces']

    def make_features(self, examples):
        inp_prompt = ''
        for example in examples:
            inp = str(example[0])
            out = example[1]
            inp_string = get_inp_string_for_task(
                inp, no_spaces=self.no_spaces, compress=self.compress)
            out_string = get_out_string(out)
            inp_prompt += f'{inp_string} {out_string} '
        return inp_prompt
