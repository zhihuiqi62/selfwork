from collections import namedtuple
from typing import Any, List
import numpy as np
from gymnasium import spaces
import torch
from torch import nn
from torch.utils.data import Dataset

# 固定随机种子确保可复现性
torch.manual_seed(0)
np.random.seed(0)
# 为CUDA设备也设置随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
# 启用确定性算法
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

State = namedtuple("State", ["value", "last_action"])

class SimpleSimulator:
    def __init__(self, initial: float, nsteps: int, **kwargs: Any) -> None:
        self.value = initial
        self.last_action = 0.0
        self.remain_steps = nsteps

    def step(self, action: float) -> None:
        assert 0.0 <= action <= self.value
        self.last_action = action
        self.remain_steps -= 1

    def get_state(self) -> State:
        return State(self.value, self.last_action)

    def done(self) -> bool:
        return self.remain_steps == 0

class SimpleStateInterpreter:
    def interpret(self, state: State) -> np.ndarray:
        return np.array([state.value], dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)

class SimpleActionInterpreter:
    def __init__(self, n_value: int) -> None:
        self.n_value = n_value

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.n_value + 1)

    def interpret(self, simulator_state: State, action: int) -> float:
        assert 0 <= action <= self.n_value
        return simulator_state.value * (action / self.n_value)

class SimpleReward:
    def __call__(self, simulator_state: State) -> float:
        """动态奖励：鼓励70%资源利用率"""
        utilization = simulator_state.last_action / simulator_state.value
        return 0.1 + 0.9 * (1.0 - abs(utilization - 0.7))

class SimpleFullyConnect(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.dims = [1] + dims
        self.output_dim = dims[-1]

        layers = []
        for in_dim, out_dim in zip(self.dims[:-1], self.dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers[-2] = nn.Linear(dims[-2], 1)  # 输出层调整为1维
        layers[-1] = nn.Sigmoid()  # 输出0~1
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) * 100  # 映射到0~100

class SimpleDataset(Dataset):
    def __init__(self, positions: List[float]) -> None:
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> float:
        return self.positions[index]

# Training and testing logic
NSTEPS = 10

# Initialize components
state_interpreter = SimpleStateInterpreter()
action_interpreter = SimpleActionInterpreter(n_value=10)
reward = SimpleReward()
policy = SimpleFullyConnect(dims=[16, 8])

# 训练过程
print("=== 训练开始 ===")
print(f"{'轮次':<6}{'平均奖励':<12}{'最大奖励':<12}{'平均损失':<12}")
print("-" * 42)

policy = SimpleFullyConnect(dims=[64, 32, 1])
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

for epoch in range(100):  # 100轮训练
    # 模拟训练数据
    total_reward = 0
    max_reward = 0
    
    for _ in range(50):  # 每轮50个episode
        simulator = SimpleSimulator(100.0, NSTEPS)
        state = simulator.get_state()
        obs = state_interpreter.interpret(state)
        policy_out = policy(torch.tensor(obs).unsqueeze(0))
        act = float(action_interpreter.interpret(state, policy_out.argmax().item()))
        
        simulator.step(act)
        rew = reward(simulator.get_state())
        
        total_reward += rew
        if rew > max_reward:
            max_reward = rew
    
    avg_reward = total_reward / 50
    print(f"{epoch+1:<8}{avg_reward:.4f}{'':<4}{max_reward:.4f}")

print("=== 训练完成 ===\n")

# 测试展示
print("=== 测试结果 ===")
simulator = SimpleSimulator(100.0, NSTEPS)
state = simulator.get_state()
obs = state_interpreter.interpret(state)
policy_out = policy(torch.tensor(obs).unsqueeze(0))
act = float(policy_out.item())  # 直接使用网络输出

simulator.step(act)
rew = float(reward(simulator.get_state()))

print(f"初始资源: 100.0")
print(f"选择动作: {act:.1f} (占比: {act/100:.0%})")
print(f"获得奖励: {rew:.4f}")
print(f"剩余步数: {simulator.remain_steps}")
print("=" * 30)