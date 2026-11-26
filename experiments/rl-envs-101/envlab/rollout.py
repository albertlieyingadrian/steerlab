"""Multi-turn rollouts for a chat model, kept as token ids so training sees exactly what the model sampled from.

One episode is one growing list of token ids, built in the ChatML layout every Qwen model uses:

    <|im_start|>system ...<|im_end|>\n<|im_start|>user {task}<|im_end|>\n<|im_start|>assistant\n
    {sampled turn 1}<|im_end|>                                  <- mask 1: the policy wrote these tokens
    \n<|im_start|>user\nTool result: {...}<|im_end|>\n<|im_start|>assistant\n   <- mask 0: the environment wrote these
    {sampled turn 2}<|im_end|> ...

The sampler (vLLM on a GPU, or a scripted fake in the tests) is passed in, so this file needs no GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from envlab.desk import SYSTEM_PROMPT, DeskEnv, Task

Sampler = Callable[[list[list[int]]], list[tuple[str, list[int]]]]


@dataclass
class Episode:
    task: Task
    env: DeskEnv
    ids: list[int]
    mask: list[int]
    messages: list[dict]
    done: bool = False
    info: dict = field(default_factory=dict)


def start_ids(tokenizer, instruction: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def run_episodes(
    tokenizer, sample: Sampler, pairs: list[tuple[Task, DeskEnv]], max_turns: int = 8, max_context: int = 3500
) -> list[Episode]:
    """Step every episode one turn at a time, batching the model calls across the episodes still running."""
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    episodes = []
    for task, env in pairs:
        obs = env.reset(task)
        ids = start_ids(tokenizer, obs)
        episodes.append(Episode(task, env, ids, [0] * len(ids), [{"role": "user", "content": obs}]))
    for _ in range(max_turns):
        live = [e for e in episodes if not e.done]
        if not live:
            break
        for episode, (text, new_ids) in zip(live, sample([e.ids for e in live])):
            if not new_ids or new_ids[-1] != im_end:  # cut off by the token cap: close the turn ourselves
                new_ids = [*new_ids, im_end]
            episode.ids += new_ids
            episode.mask += [1] * len(new_ids)
            episode.messages.append({"role": "assistant", "content": text})
            obs, _, done, info = episode.env.step(text)
            if done or len(episode.ids) > max_context:
                episode.done = True
                episode.info = info if done else episode.env._finish("context full")[3]
                continue
            episode.messages.append({"role": "tool", "content": obs})
            delta = f"\n<|im_start|>user\nTool result: {obs}<|im_end|>\n<|im_start|>assistant\n"
            delta_ids = tokenizer(delta, add_special_tokens=False)["input_ids"]
            episode.ids += delta_ids
            episode.mask += [0] * len(delta_ids)
    for episode in episodes:
        if not episode.done:  # ran out of turns while the model was still calling tools
            episode.done, episode.info = True, episode.env._finish("out of turns")[3]
    return episodes
