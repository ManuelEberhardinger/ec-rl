import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import *
from dreamcoder.task import Task
from dreamcoder.program import *
from dreamcoder.domains.minigrid.primitives import *
from envs.registration import make as gym_make
import random
import numpy as np
import eval as eval_minigrid


def _make_env(env_name='MultiGrid-PerfectMazeMedium-v0'):
    env = gym_make(env_name)
    return env


class MinigridMazeFeatureExtractorBase(DeviceAwareModule):
    special = None

    def __init__(self, tasks, obs_shape=(1, 5, 5),
                 testingTasks=None,
                 cuda=False,
                 conv_filters=16,
                 conv_kernel_size=3,
                 scalar_fc=5,
                 scalar_dim=4,
                 random_z_dim=0,
                 xy_dim=0,
                 recurrent_arch='lstm',
                 recurrent_hidden_size=256,
                 dcd=None):
        super().__init__()
        self.tasks = tasks
        self.env_name = [
            'MultiGrid-PerfectMazeSmall-v0',
            'MultiGrid-PerfectMazeMedium-v0',
            'MultiGrid-PerfectMazeLarge-v0',
            'MultiGrid-PerfectMazeXL-v0',
            'MultiGrid-NineRoomsFewerDoors-v0',
            'MultiGrid-NineRooms-v0',
            'MultiGrid-SixteenRoomsFewerDoors-v0',
            'MultiGrid-SixteenRooms-v0',
            'MultiGrid-Labyrinth2-v0',
            'MultiGrid-Labyrinth-v0',
            'MultiGrid-LargeCorridor-v0',
            'MultiGrid-SmallCorridor-v0',
            'MultiGrid-Maze3-v0',
            'MultiGrid-Maze2-v0',
            'MultiGrid-Maze-v0',
            'MultiGrid-MediumMaze-v0',
            # 'MultiGrid-MiniMaze-v0',
        ]

    def forward(self, inputs):
        raise NotImplementedError

    def featuresOfTask(self, t):
        inp = {
            'image': torch.tensor([[inp[0]] for (inp, out) in t.examples], dtype=torch.float),
            'direction': torch.tensor([inp[1] for (inp, out) in t.examples], dtype=torch.float),
        }
        return self(inp)

    # we need to use 1 process for dreamcoder compatibility
    def run_program_on_env(self, program, chunkSize=None, min_len=5, max_len=20, verbose=False):
        # we always choose a random env from the list
        env_name = random.choice(self.env_name)
        env = _make_env(env_name=env_name)
        obs = env.reset()
        if verbose:
            print(program)
        assert len(obs.get('image')) > 0

        examples = []
        if chunkSize is None:
            chunkSize = random.randint(min_len, max_len)
        if verbose:
            print('sequence_length:', chunkSize)
        for _ in range(chunkSize):
            obs_inp = np.flip(obs.get('image')[:, :, 0], 1).tolist()
            direction_inp = obs.get('direction')[0]
            input_ex = (obs_inp, direction_inp)
            # output ex is the action
            output_ex = runWithTimeout(lambda: program.runWithArguments(input_ex), None)
            examples.append((input_ex, output_ex))
            if verbose:
                print('Input:', input_ex)
                print('Output:', output_ex)
            # reset Random
            obs, _, _, _,  = env.step(output_ex)

        return examples

    def taskOfProgram(self, p, t, min_len=5, max_len=20):
        # we need to run the sampled program on the dcd env
        # maybe we should use the seeds used for collecting the data
        examples = self.run_program_on_env(p, min_len=min_len, max_len=max_len)
        return Task("Helmhotz Dream", arrow(tmap, tdirection, taction), examples)


class MinigridMazeFeatureExtractor(MinigridMazeFeatureExtractorBase):
    special = None

    def __init__(self, tasks, obs_shape=(1, 5, 5),
                 testingTasks=None,
                 cuda=False,
                 conv_filters=16,
                 conv_kernel_size=3,
                 scalar_fc=5,
                 scalar_dim=4,
                 random_z_dim=0,
                 xy_dim=0,
                 recurrent_arch='lstm',
                 recurrent_hidden_size=256,
                 dcd=None,
                 env_name='MultiGrid-PerfectMazeMedium-v0'):
        super().__init__(tasks)

        # Image embeddings
        m = obs_shape[-2]  # x input dim
        n = obs_shape[-1]  # y input dim
        c = obs_shape[-3]  # channel input dim

        self.image_conv = nn.Sequential(
            Conv2d_tf(c, conv_filters, kernel_size=conv_kernel_size, stride=1, padding='valid'),
            nn.Flatten(),
            nn.ReLU()
        )
        self.image_embedding_size = (n-conv_kernel_size+1)*(m-conv_kernel_size+1)*conv_filters
        self.preprocessed_input_size = self.image_embedding_size
        self.scalar_embed = nn.Linear(scalar_dim, scalar_fc)
        self.scalar_dim = scalar_dim
        self.preprocessed_input_size += scalar_fc

        self.lstm = nn.LSTM(
            input_size=self.preprocessed_input_size,
            hidden_size=recurrent_hidden_size,
            num_layers=1
        )

        self.outputDimensionality = recurrent_hidden_size

        apply_init_(self.modules())

        self.train()

    def forward(self, inputs):
        # Unpack input key values
        image = inputs.get('image')
        direction = inputs.get('direction')
        in_image = self.image_conv(image)
        in_direction = F.one_hot(direction.long(), num_classes=self.scalar_dim).to(self.device)
        in_direction = self.scalar_embed(in_direction.float())
        in_embedded = torch.cat((in_image, in_direction), axis=-1).unsqueeze(dim=0)
        core_features, rnn_hxs = self.lstm(in_embedded)
        hidden = rnn_hxs[0] + rnn_hxs[1]
        hidden = hidden.mean(dim=1)
        return hidden


class MinigridMazeFeatureExtractorOld(MinigridMazeFeatureExtractorBase):
    special = None

    def __init__(self, tasks, obs_shape=(1, 5, 5),
                 testingTasks=None,
                 cuda=False,
                 conv_filters=16,
                 conv_kernel_size=3,
                 scalar_fc=5,
                 scalar_dim=4,
                 random_z_dim=0,
                 xy_dim=0,
                 recurrent_arch='lstm',
                 recurrent_hidden_size=256,
                 dcd=None,
                 env_name='MultiGrid-PerfectMazeMedium-v0'):
        super().__init__(tasks)

        # Image embeddings
        m = obs_shape[-2]  # x input dim
        n = obs_shape[-1]  # y input dim
        c = obs_shape[-3]  # channel input dim
        print('x input dim', m, 'y input dim', n, 'channel input dim', c)
        self.image_conv = nn.Sequential(
            Conv2d_tf(c, conv_filters, kernel_size=conv_kernel_size, stride=1, padding='valid'),
            nn.Flatten(),
            nn.ReLU()
        )
        self.image_embedding_size = (n-conv_kernel_size+1)*(m-conv_kernel_size+1)*conv_filters
        self.preprocessed_input_size = self.image_embedding_size

        # x, y positional embeddings
        self.xy_embed = None
        self.xy_dim = xy_dim
        if xy_dim:
            self.preprocessed_input_size += 2*xy_dim

        # Scalar embedding
        self.scalar_embed = None
        self.scalar_dim = scalar_dim
        if scalar_dim:
            self.scalar_embed = nn.Linear(scalar_dim, scalar_fc)
            self.preprocessed_input_size += scalar_fc

        self.preprocessed_input_size += random_z_dim
        # self.outputDimensionality = self.preprocessed_input_size

        self.rnn = RNN_DreamCoder(
            input_size=self.preprocessed_input_size,
            hidden_size=recurrent_hidden_size,
            arch=recurrent_arch)
        self.outputDimensionality = recurrent_hidden_size

        apply_init_(self.modules())

        self.train()

    def forward(self, inputs):
        # Unpack input key values
        image = inputs.get('image')
        scalar = inputs.get('direction')
        if scalar is None:
            scalar = inputs.get('time_step')

        x = inputs.get('x')
        y = inputs.get('y')

        in_z = inputs.get('random_z', torch.tensor([], device=self.device))

        in_image = self.image_conv(image)

        if self.xy_embed:
            x = one_hot(self.xy_dim, x, device=self.device)
            y = one_hot(self.xy_dim, y, device=self.device)
            in_x = self.xy_embed(x)
            in_y = self.xy_embed(y)
        else:
            in_x = torch.tensor([], device=self.device)
            in_y = torch.tensor([], device=self.device)

        if self.scalar_embed:
            in_scalar = one_hot(self.scalar_dim, scalar).to(self.device)
            in_scalar = self.scalar_embed(in_scalar)
        else:
            in_scalar = torch.tensor([], device=self.device)

        in_embedded = torch.cat((in_image, in_x, in_y, in_scalar, in_z), dim=-1)
        # print('in_embedded', in_embedded.size())

        core_features, rnn_hxs = self.rnn(in_embedded)
        return rnn_hxs[0] + rnn_hxs[1]
