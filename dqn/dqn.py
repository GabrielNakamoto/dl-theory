from tinygrad import Tensor, nn, Context, TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
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
# TODO: linearly anneal epsilon, start random for first ~1/10 epochs
N, T, n_epochs, batch_size, epsilon = 256, 5000, 40, 32, 0.1
n_anneal, target_delay = 5, 500
model, target = QNet(), QNet()
def sync_target():
  for p, q in zip(get_parameters(model), get_parameters(target)): q.replace(p.detach())
sync_target()
optim = nn.optim.Adam(nn.state.get_parameters(model))

gym.register_envs(ale_py)
# https://ale.farama.org/environments/breakout/
env = gym.make("ALE/Breakout-v5", obs_type="grayscale")

def obs(f): return f[::2, ::2]
def phi(frames) -> np.ndarray: return np.stack(frames, axis=0).reshape(4, 105, 80)

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
  windows = Tensor(windows).reshape(-1, 4, 105, 80)
  return model(windows).max(axis=1).numpy().mean()

@TinyJit
def train_step(batch_r_t, batch_phi_t, batch_phi_t_1, batch_a_t, discount):
  # q_t_1 estimates have to come from frozen network? detached from backprop signal
  q_t, q_t_1 = model(batch_phi_t), target(batch_phi_t_1)
  # compare action taken and best future Q value
  q_t = q_t.gather(1, batch_a_t)
  bellman = (batch_r_t  + discount * q_t_1.max(-1)).unsqueeze(-1)

  optim.zero_grad()
  loss = ((bellman - q_t) ** 2).mean().backward()
  optim.step()

@Context(TRAINING=1)
def train_epoch():
  D_s_t, D_a_t, D_r_t, D_s_t_1, = [], [], [], []
  env.reset(seed=42)
  frames, phi_t, steps = [], None, 0

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
      idx = np.random.randint(0, len(D_s_t), size=batch_size)
      batch_phi_t = Tensor(np.stack(D_s_t)[idx])
      batch_a_t = Tensor(np.stack(D_a_t)[idx]).unsqueeze(-1)
      batch_r_t = Tensor(np.stack(D_r_t)[idx])
      batch_phi_t_1 = Tensor(np.stack(D_s_t_1)[idx])
      steps += 1

      if steps >= target_delay:
        sync_target()
        steps = 0

      train_step(batch_r_t, batch_phi_t, batch_phi_t_1, batch_a_t, 0)

    if game_over:
      sync_target()
      env.reset()
      frames, phi_t = [], None

  env.close()

"""
run = wandb.init(
  entity="raine1-me",
  project="DQN-breakout",
  config = {
    "epsilon": epsilon,
    "batch_size" : batch_size,
    "epochs" : n_epochs,
    "memory_size" : N
  }
)"""

for epoch in range(n_epochs):
  epsilon = 1.0 - (min(epoch, n_anneal) / n_anneal)
  safe_save(get_state_dict(model), "qnet.safetensor")
  train_epoch()
  print(f"q_avg: {q_avg():.4f}, epsilon: {epsilon}")
  # run.log({"valid_q_avg":q_avg()})

# run.finish()
