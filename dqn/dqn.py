from tinygrad import Tensor, nn, dtypes
import ale_py, gymnasium as gym, random, numpy as np

# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
class QNet:
  def __init__(self):
    self.layers = [
      nn.Conv2d(1, 16, 8, 4), Tensor.relu,
      nn.Conv2d(16, 32, 4, 2), Tensor.relu,
      lambda x: x.flatten(1),
      nn.Linear(2816, 256), Tensor.relu,
      nn.Linear(256, 4)
    ]

  def __call__(self, s:Tensor) -> Tensor:
    return s.sequential(self.layers)

# stack and random idx
D_s_t, D_a_t, D_r_t, D_s_t_1, = [], [], [], []
N, n_epochs, batch_size = 64, 100, 8
epsilon = 0.15
model = QNet()
optim = nn.optim.Adam(nn.state.get_parameters(model))

gym.register_envs(ale_py)
# https://ale.farama.org/environments/breakout/
env = gym.make("ALE/Breakout-v5", render_mode="human", obs_type="grayscale")

def stack(ts): return ts[0].stack(*ts[1:])
def phi(frames) -> Tensor: return Tensor(np.stack([f[::2, ::2] for f in frames], axis=0)).reshape(4, 1, 80, 105)

for epoch in range(n_epochs):
  s_t, info = env.reset(seed=42)
  done, t, frames = False, 0, []
  while not done:
    phi_t = None if len(frames) < 4 else phi(frames[-4:])

    # epsilon greedy strategy, choose random action E% of the time else
    # choose action that maximizes expected reward (argmax_a_t Q(s; params))
    if phi_t is None or random.random() > epsilon:
      a_t = env.action_space.sample()
    else:
      a_t = model(phi_t).argmax(-1)

    print(a_t.tolist())
    s_t1, r_t, done, _, _ = env.step(a_t)
    frames.append(s_t1)

    if len(frames) >= 4:
      D_s_t.append(phi_t)
      D_a_t.append(Tensor([a_t], dtype=dtypes.uint32).one_hot(4))
      D_r_t.append(Tensor(r_t))
      D_s_t_1.append(phi(frames[-4:]))

    t, s_t = t+1, s_t1

    # learn from randomly sampled memory batch
    if len(D_s_t) > batch_size:
      idx = Tensor.randint(batch_size, high=N)
      batch_phi_t = stack(D_s_t[-batch_size:])[idx]
      batch_a_t = stack(D_a_t[-batch_size:])[idx]
      batch_r_t = stack(D_r_t[-batch_size:])[idx]
      batch_phi_t_1 = stack(D_s_t_1[-batch_size:])[idx]

      # - how to choose discount per time step?
      # - how to determine terminal?

      q_t, q_t_1 = model(batch_phi_t), model(batch_phi_t_1)
      y = batch_r_t + q_t_1

      optim.zero_grad()
      loss = ((y - q_t) ** 2).backward()
      optim.step()

env.close()
