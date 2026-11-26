"""Step 1: the smallest text RL environment, "guess the number", in about 50 lines.

An environment is a game master with two buttons:

    reset()        -> the first observation (what the player sees)
    step(action)   -> (observation, reward, done, info)

The player (a policy: a script, a model, a person) reads the observation and sends back an action, a string here.
One run from reset() to done is an **episode**; the list of (observation, action, reward) it produced is the
**trajectory** (also called a **rollout**). RL changes the policy so that trajectories with more reward get likelier.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


@dataclass
class GuessEnv:
    """Guess a secret integer in [low, high] within max_turns. Reward 1 for a correct guess, 0 otherwise."""

    low: int = 1
    high: int = 100
    max_turns: int = 7
    secret: int = 0
    turns: int = 0
    done: bool = True
    rng: random.Random = field(default_factory=random.Random)

    def reset(self, seed: int | None = None) -> str:
        if seed is not None:
            self.rng.seed(seed)
        self.secret = self.rng.randint(self.low, self.high)
        self.turns, self.done = 0, False
        return f"I picked a number from {self.low} to {self.high}. You have {self.max_turns} guesses. Reply with one number."

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode is over: call reset() first")
        self.turns += 1
        numbers = re.findall(r"-?\d+", action)
        if len(numbers) != 1:  # a malformed action costs a turn and earns nothing
            obs, reward = "Reply with exactly one number.", 0.0
        elif int(numbers[0]) == self.secret:
            obs, reward, self.done = "Correct!", 1.0, True
        else:
            obs, reward = ("Higher." if int(numbers[0]) < self.secret else "Lower."), 0.0
        if self.turns >= self.max_turns and not self.done:
            obs, self.done = obs + f" Out of guesses; it was {self.secret}.", True
        return obs, reward, self.done, {"turns": self.turns, "secret": self.secret}


# --- policies and the rollout loop (not part of the environment) ---------------------------------------------------


class BinarySearchPolicy:
    """A scripted policy: halve the interval each turn. Solves 1..100 in at most 7 guesses."""

    def __init__(self, low: int = 1, high: int = 100):
        self.low, self.high, self.last = low, high, None

    def act(self, observation: str) -> str:
        if observation.startswith("Higher"):
            self.low = self.last + 1
        elif observation.startswith("Lower"):
            self.high = self.last - 1
        self.last = (self.low + self.high) // 2
        return str(self.last)


class RandomPolicy:
    """Guess uniformly at random each turn (a baseline: what does doing nothing clever earn?)."""

    def __init__(self, seed: int = 0, low: int = 1, high: int = 100):
        self.rng, self.low, self.high = random.Random(seed), low, high

    def act(self, observation: str) -> str:
        return str(self.rng.randint(self.low, self.high))


def rollout(env: GuessEnv, policy, seed: int | None = None) -> list[dict]:
    """Run one episode and return its trajectory: one dict per step."""
    observation = env.reset(seed)
    trajectory, done = [], False
    while not done:
        action = policy.act(observation)
        next_observation, reward, done, info = env.step(action)
        trajectory.append({"observation": observation, "action": action, "reward": reward, "done": done, **info})
        observation = next_observation
    return trajectory


def episode_return(trajectory: list[dict]) -> float:
    """The return: the sum of an episode's rewards (here 0 or 1)."""
    return sum(step["reward"] for step in trajectory)
