from tinygrad import Tensor, nn, Context, TinyJit
import ale_py, gymnasium as gym, random, numpy as np, wandb

# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
class QNet:
  def __init__(self):
    self.layers = [
      nn.Conv2d(4, 16, 8, 4), Tensor.relu,
      nn.Conv2d(16, 32, 4, 2), Tensor.relu,
      lambda x: x.flatten(1),
      nn.Linear(2816, 256), Tensor.relu,
      nn.Linear(256, 4)
    ]

  def __call__(self, s:Tensor) -> Tensor:
    return s.sequential(self.layers)

# stack and random idx
N, T, n_epochs, batch_size, epsilon = 256, 5000, 40, 32, 0.05
model = QNet()
optim = nn.optim.Adam(nn.state.get_parameters(model))

gym.register_envs(ale_py)
# https://ale.farama.org/environments/breakout/
env = gym.make("ALE/Breakout-v5", obs_type="grayscale")

def obs(f): return f[::2, ::2]
def phi(frames) -> np.ndarray: return np.stack(frames, axis=0).reshape(4, 80, 105)

env.reset()
done, validation_states = False, []
while not done:
  s, _, done, _, _ = env.step(env.action_space.sample())
  validation_states.append(obs(s))
env.reset()

# jit the greedy forward too, and give it an already-realized tensor
@TinyJit
def greedy(q:Tensor) -> Tensor: return model(q).argmax(-1).realize()

# one batched forward over every validation window instead of one forward per window
def q_avg() -> float:
  frames = np.stack(validation_states)
  windows = np.stack([frames[i-3:i+1] for i in range(3, len(frames))])
  return model(Tensor(windows).reshape(-1, 4, 80, 105)).max(axis=1).numpy().mean()

@TinyJit
def train_step(batch_r_t, batch_phi_t, batch_phi_t_1, batch_a_t):
  q_t, q_t_1 = model(batch_phi_t), model(batch_phi_t_1)
  # is this correct? or should I be gathering from a_t_1 from q_t_1 to sync time?
  q_t, q_t_1 = q_t.gather(1, batch_a_t), q_t_1.gather(1, batch_a_t)
  y = (batch_r_t + q_t_1.argmax(-1)).unsqueeze(-1)

  optim.zero_grad()
  loss = ((y - q_t) ** 2).sum().backward()
  optim.step()

@Context(TRAINING=1)
def train_epoch():
  D_s_t, D_a_t, D_r_t, D_s_t_1, = [], [], [], []
  env.reset(seed=42)
  frames, phi_t = [], None

  while len(D_s_t) < T:
    if phi_t is None and len(frames) >= 4: phi_t = phi(frames[-4:])

    # epsilon greedy strategy, choose random action E% of the time else
    # choose action that maximizes expected reward (argmax_a_t Q(s; params))
    if phi_t is not None and random.random() > epsilon:
      a_t = greedy(Tensor(phi_t[None])).item()
    else:
      a_t = env.action_space.sample()

    s_t1, r_t, game_over, _, _ = env.step(a_t)
    frames.append(obs(s_t1))

    if phi_t is not None:
      D_s_t.append(phi_t)
      D_a_t.append(np.uint32(a_t))
      D_r_t.append(np.float32(r_t))
      phi_t = phi(frames[-4:])  # this is the next step's phi_t, so only compute it once
      D_s_t_1.append(phi_t)

    # learn from randomly sampled memory batch
    if phi_t is not None and len(D_s_t) > batch_size:
      idx = np.random.randint(0, min(N, len(D_s_t), batch_size), size=batch_size)
      batch_phi_t = Tensor(np.stack(D_s_t)[idx])
      batch_a_t = Tensor(np.stack(D_a_t)[idx]).unsqueeze(-1)
      batch_r_t = Tensor(np.stack(D_r_t)[idx])
      batch_phi_t_1 = Tensor(np.stack(D_s_t_1)[idx])
      # - do I need a discount factor?
      # - how to determine terminal?
      train_step(batch_r_t, batch_phi_t, batch_phi_t_1, batch_a_t)

    if game_over:
      env.reset()
      frames, phi_t = [], None

  env.close()

run = wandb.init(
  entity="raine1-me",
  project="DQN-breakout",
  config = {
    "epsilon": epsilon,
    "batch_size" : batch_size,
    "epochs" : n_epochs,
    "memory_size" : N
  }
)

from tinygrad.nn.state import get_state_dict, safe_save
for epoch in range(n_epochs):
  safe_save(get_state_dict(model), "qnet.safetensor")
  train_epoch()
  run.log({"valid_q_avg":q_avg()})

run.finish()
