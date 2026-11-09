# How the pieces connect

Three codebases, each drawn the same way: files are boxes, classes and functions inside them, arrows are "uses".
Source: Auditron at commit `9acab0c` (`letter_env/`, `auditron_env/`); ours is this folder.

## Words used in all three

| Word | Means | In code |
|---|---|---|
| **Action** | one move by one player | `LetterAction(letter="a")`, `AuditronAction(agent_id, content=<JSON>)` |
| **Observation** | what a player is shown after a move, plus `reward` and `done` | `LetterObservation(message, reward, done)` |
| **State** | everything the game remembers between moves, including secrets | `LetterState(stage)`, `AuditronState(...)` |
| **Environment** | the rules: `reset()` starts a game, `step(action)` applies one move | `LetterEnv`, `AuditronEnv` |
| **Step / round / game** | one move / one auction (4 bids, an audit, a purchase) / 50 rounds; "episode" = game | |

`Action`, `Observation`, `State`, `Environment` and `create_app` come from **OpenEnv** (`openenv.core`). OpenEnv's job is
the plumbing: it fixes these four shapes so any trainer can talk to any game, and `create_app` turns an Environment
into a web server (`POST /reset`, `POST /step`, `GET /state`) for the Hugging Face Space.

## 1. The demo letter game (Auditron's `letter_env/`, 213 lines)

```
 openenv.core ─────────────────────────────────────────────────────────────┐
   Action   Observation   State   Environment[A, O, S]   create_app()       │
     │          │           │            │                    │            │
     ▼          ▼           ▼            ▼                    │            │
 server.py ─────────────────────────────────────────────────  │  ─────────┐│
 │ LetterAction        LetterObservation     LetterState      │           ││
 │   letter: str         message: str          stage: 0|1|2   │           ││
 │        │               (+ reward, done)       ▲            │           ││
 │        │                     ▲                │ holds      │           ││
 │        ▼                     │ returns        │            │           ││
 │ LetterEnv(Environment) ──────┴────────────────┘            │           ││
 │   reset()            -> LetterObservation                   │           ││
 │   step(LetterAction) -> LetterObservation   rewards: 0 / 0.5 for "a" / 1.0 for "z"
 │   state              -> LetterState                         │           ││
 │                                                             ▼           ││
 │ app = create_app(env=LetterEnv, action_cls=..., observation_cls=...) ──► web server (Dockerfile, HF Space)
 └──────────────────────────────────────────────────────────────────────────┘│
        ▲ from server import LetterAction, LetterEnv      ▲                   │
        │                                                  │                   │
 play.py: main()                                    train.py                   │
   env = LetterEnv(); obs = env.reset()               extract_letter(text)     │
   loop: input() -> LetterAction -> env.step()        run_episode(model, tok, env)
   (you are the player)                                 prompt <- obs.message
                                                        model.generate() -> letter -> env.step()
                                                      main(): 10 episodes, prints rewards
                                                      (no optimizer, no update: an eval)
```

## 2. The real auction game (Auditron's `auditron_env/`, 1,828 lines)

```
 rewards.py: ~40 constants (REWARD_*, PENALTY_*, TOTAL_PARTS=50, NUM_SUPPLIERS=4, ...)
     │ imported by                                                         │ imported by
     ▼                                                                     ▼
 server.py ──────────────────────────────────────────────────────────────────────────────┐
 │ SUPPLIER_PERSONALITIES (Honest / Random / Shrewd / Dishonest: name + system prompt)    │
 │ AuditronAction(agent_id, content)   AuditronObservation(agent_id, phase, observation)  │
 │ AuditronState: round, phase 0-3, required_strength, supplier_bids (incl. SECRET        │
 │                actual_strength), supplier_history, event_log, rewards, personalities   │
 │                                                                                        │
 │ AuditronEnv(Environment)                                                               │
 │   reset(seed) ── pre-draws 50 specs + costs, shuffles personalities ──► AuditronState  │
 │   step(action) ── routes by phase:                                                     │
 │       phase 0 ─► _handle_supplier()  (×4: parse bid_price + actual_strength)           │
 │       phase 1 ─► _handle_auditor()   (pick, reason, flags)                             │
 │       phase 2 ─► _handle_buyer()     ─► _resolve_round()  ◄── the "god-engine":        │
 │                                           failure draw, flag rewards from the SECRET   │
 │                                           strengths, event_log (public, no secrets)    │
 │                     last round ─► _calc_episode_rewards() (ranks, TPR/FPR bonuses)     │
 │   get_supplier_obs() / get_auditor_obs() / get_buyer_obs()  ◄── each role's view       │
 │   _reasoning_score()  (regex score of the auditor's reason)                            │
 │ app = create_app(env=AuditronEnv, ...) ──► web server (HF Space demo)                  │
 └────────────────────────────────────────────────────────────────────────────────────────┘
        ▲ from server import AuditronAction, AuditronEnv, SUPPLIER_IDS
        │
 train.py ────────────────────────────────────────────────────────────────────────────────┐
 │ 1. generate_prompts(20 games): AuditronEnv + RANDOM moves ─► get_*_obs() ─► build_prompt()│
 │        = a fixed dataset of single-turn prompts (supplier / auditor / buyer)            │
 │ 2. GRPOTrainer(model, reward_funcs=[format_reward, reasoning_reward, economic_reward])  │
 │        each completion scored ALONE by these functions (the env is not called):         │
 │          format_reward     valid JSON per role (+2), flags is a list (+0.5)             │
 │          reasoning_reward  auditor: +1 per keyword type, +1 ≥20 words, +1 ≥50 words     │
 │          economic_reward   suppliers: profitable bid, sane strength, [RANK_REWARD:x]    │
 │ 3. evaluate_model() every 40 steps (15-round games) and a final 50-round game (seed     │
 │        99999): the trained model plays ALL roles through AuditronEnv.step()            │
 │        generate_action() / generate_actions_batch() ─► AuditronAction ─► env.step()    │
 │ logs: reasoning_*.jsonl, episodes_*.jsonl, eval_*.json ─► reports/gen_report_template.py │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
 play.py: you play all six roles in the terminal through AuditronEnv.step()
 auditron_colab.ipynb: cell 3 starts server.py's web server; cell 4 runs train.py, which imports
                       AuditronEnv directly and never calls that server
```

## 3. Our rewrite (this folder)

```
 demo_letter_game.py                           real_auction_game.py
   LetterGame: reset() / step(letter)            Bid(price, strength=SECRET).cheats(required)
   play(model, tok) -> reward, log-probs         supplier_bid(personality, ...)  scripted suppliers
   evaluate()  (eval: no update)                 Game(seed, hide_history)
   main(): eval ─► 30 REINFORCE updates ─► eval     reset(): deals all 50 rounds up front
                                                     observe(): the auditor's view (no secrets)
                                                     step(pick, flags, reason): buyer follows pick,
                                                       referee scores flags with the secret
                                                     rounds: private log for scoring
                                                        ▲ from real_auction_game import Game, SUPPLIERS
                       ┌────────────────────────────────┼────────────────────────┐
                 auditors.py                      evaluate.py                 play.py
                 random / cheapest / history /    play(seed, auditor) ─► counts() ─► summarise()
                 price / oracle                   (20 held-out games, game bootstrap, both halves)
                                                  ─► results/<auditor>.json
 modal_notebook.py: runs Auditron's own notebook (and the Run 30 replication) on a rented GPU,
                    with requirements-auditron-2026-03.txt
 tests/test_real_auction_game.py: same seed ─► same game for any auditor; no leaks; scoring
```

Differences that matter: in Auditron, training never calls `env.step()` (prompts are fixed, rewards are text rules);
in ours, every score comes from `Game.step()` and the referee.
