import os
import pickle
import torch.nn as nn
from models.common import *
from dreamcoder.task import Task
from dreamcoder.program import *
from dreamcoder.domains.minigrid.primitives import *
from envs.registration import make as gym_make
from envs.wrappers import ParallelAdversarialVecEnv, VecMonitor, VecNormalize, \
    VecPreprocessImageWrapper, VecFrameStack, MultiGridFullyObsWrapper, CarRacingWrapper, TimeLimit

def _make_env(seed, env_name='MultiGrid-GoalLastEmptyAdversarialEnv-Edit-v0'):
    env_kwargs = {'seed': seed}
    env = gym_make(env_name, **env_kwargs)
    return env

def create_parallel_env(num_processes, seed=88, adversary=True):
    is_multigrid = True
    make_fn = lambda: _make_env(seed)
    venv = ParallelAdversarialVecEnv([make_fn]*num_processes, adversary=adversary)
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False, ret=True)

    obs_key = None
    scale = None
    transpose_order = [2,0,1] # Channels first
    if is_multigrid:
        obs_key = 'image'
        scale = 10.0

    venv = VecPreprocessImageWrapper(venv=venv, obs_key=obs_key,
            transpose_order=transpose_order, scale=scale)

    if is_multigrid:
        ued_venv = venv
    seeds = [i for i in range(num_processes)]
    venv.set_seed(seeds)

    return venv, ued_venv

# we need to use 1 process for dreamcoder compatibility 
def run_program_on_env(program, num_processes=1, verbose=False):
    data_dir = '/home/ma/e/eberhardinger/workspaces/ec/dreamcoder/domains/minigrid/collected_data/2022-09-19T15:47:46.198763/level_sampler/'
    level_sampler_file = os.path.join(data_dir, 'levelSampler-iter-0.pkl')
    with open(level_sampler_file, 'rb') as f:
        obj = pickle.load(f)
        level_sampler = obj['level_sampler'].get('agent')
        level_store = obj['level_store']

    venv, ued_venv = create_parallel_env(num_processes)

    current_level_seeds = [level_sampler.sample_replay_level() for _ in range(num_processes)]
    levels = [level_store.get_level(seed) for seed in current_level_seeds]
    ued_venv.reset_to_level_batch(levels)
    obs = venv.reset_agent()
    if verbose:
        print(program)
    assert len(obs.get('image')) > 0

    examples = []
    chunkSize = random.randint(3, 10)
    if verbose:
        print('sequence_length:', chunkSize)
    for _ in range(chunkSize):
        obs_inp = (obs.get('image')[0][0] * 10).cpu().numpy().astype(int).tolist()
        direction_inp = obs.get('direction')[0].cpu().item()
        input_ex = (obs_inp, direction_inp)

        # output ex is the action
        output_ex = runWithTimeout(lambda: program.runWithArguments(input_ex), None)
        examples.append((input_ex, output_ex))
        if verbose:
            print('Input:', input_ex) 
            print('Output:', output_ex)
        # reset Random
        obs, _, _, _ = venv.step_env(np.array([[output_ex]]), reset_random=False)

    return examples

class MinigridFeatureExtractor(DeviceAwareModule):
    special = None
    def __init__(self, tasks, obs_shape=(1,5,5),
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
        super(MinigridFeatureExtractor, self).__init__()
        self.recomputeTasks = True
        self.tasks = tasks

        self.level_sampler = dcd.get('level_sampler')
        self.level_store = dcd.get('level_store')
        self.venv = dcd.get('venv')
        self.ued_venv = dcd.get('ued_venv')
        self.num_processes = dcd.get('num_processes')

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
        #self.outputDimensionality = self.preprocessed_input_size

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
        #print('in_embedded', in_embedded.size())

        core_features, rnn_hxs = self.rnn(in_embedded)
        return rnn_hxs[0] + rnn_hxs[1]

    def featuresOfTask(self, t):
        inp = {
            'image': torch.tensor([[inp[0]] for (inp, out) in t.examples], dtype=torch.float),
            'direction': torch.tensor([[inp[1]] for (inp, out) in t.examples], dtype=torch.float),
            'x': None, 'y': None
        }
        return self(inp)

    # we need to use 1 process for dreamcoder compatibility 
    def run_program_on_env(self, program, verbose=False):
        level_sampler = self.level_sampler.get('agent')
        level_store = self.level_store
        venv = self.venv
        ued_venv = self.ued_venv
        current_level_seeds = [level_sampler.sample_replay_level() for _ in range(self.num_processes)]
        levels = [level_store.get_level(seed) for seed in current_level_seeds]
        ued_venv.reset_to_level_batch(levels)
        obs = venv.reset_agent()
        if verbose:
            print(program)
        assert len(obs.get('image')) > 0

        examples = []
        chunkSize = random.randint(3, 15)
        if verbose:
            print('sequence_length:', chunkSize)
        for _ in range(chunkSize):
            obs_inp = (obs.get('image')[0][0] * 10).cpu().numpy().astype(int).tolist()
            direction_inp = obs.get('direction')[0].cpu().item()
            input_ex = (obs_inp, direction_inp)

            # output ex is the action
            output_ex = runWithTimeout(lambda: program.runWithArguments(input_ex), None)
            examples.append((input_ex, output_ex))
            if verbose:
                print('Input:', input_ex) 
                print('Output:', output_ex)
            # reset Random
            obs, _, _, _ = venv.step_env(np.array([[output_ex]]*self.num_processes), reset_random=False)

        return examples
        
    def taskOfProgram(self, p, t):
        # we need to run the sampled program on the dcd env
        # maybe we should use the seeds used for collecting the data
        examples = self.run_program_on_env(p)
        return Task("Helmhotz Dream", arrow(tmap, tdirection, taction), examples)

    def __getstate__(self):

        # this method is called when you are
        # going to pickle the class, to know what to pickle
        state = self.__dict__.copy()
        
        # we delete the venv as we do not need them later when the model is trained
        # we only need them for dreaming
        del state['venv']
        del state['ued_venv']
        del state['level_sampler']
        del state['level_store']
        del state['num_processes']
        return state
