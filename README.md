# Exact-Physics Planner — Conv-Cup '26 submission

An autonomous soccer agent that wins by **simulating the official physics exactly** and
searching over every legal move-and-kick combination every iteration. No training data,
no third-party packages: pure Python standard library, deterministic, typically 2-300 ms
per decision.

## How to run

```
python -m team_bot.bot --model team_bot/models/planner_params.json
```

The process reads one JSON observation per line on standard input and writes exactly one
JSON action per line on standard output (diagnostics only go to standard error).
`requirements.txt` is intentionally empty: only the Python 3.11+ standard library is used.

## How it thinks

1. **Canonical frame.** Every observation is mirrored so that the agent always attacks
   upward. The official obstacle generator places obstacles in vertically mirrored pairs,
   so the mirror is an exact symmetry and one brain plays both sides.
2. **Bit-exact ball replica.** `Arena.trace` reproduces the engine's sub-stepping, wall
   rebounds, obstacle rebounds and goal-mouth test with identical floating-point
   operations (verified against the engine on hundreds of random kicks with rebounds).
   Every one of the 24 kicks (8 directions x 3 powers) from every reachable kick origin is
   traced, including bank shots off walls and obstacles.
3. **Guaranteed interception margins.** After *k* moves a player can only be inside *k*
   copies of the 8-direction move octagon. The agent computes the Euclidean gap between the
   ball path and everything the opponent can possibly reach in time (exactly for the next
   move, octagon bound afterwards). A positive margin means **no opponent policy can stop
   the shot**.
4. **Expected-goal valuation.** Shots, dribbles and passes share one currency: an estimate
   of the possession's goal value. Shot success is a sigmoid of the interception margin;
   failed shots and lost races are charged by where the opponent regains the ball.
5. **Two-ply minimax dribbling.** When no shot is good enough, the agent searches its
   move, the opponent's worst reply, and its own follow-up move or shot. It respects the
   tackle rule (protected for the first three possession iterations, then any moving
   opponent in contact steals the ball), the 10-iteration possession limit, and contact
   resolution.
6. **Self-passes and releases.** Under pressure, non-scoring kicks (including rebounds off
   obstacles and walls) are evaluated by racing both players to the ball.
7. **Defence.** With the opponent in possession, the agent enumerates every shot the
   opponent could take after any of its moves and positions itself to minimise the best
   one, while staying goal-side and stepping into legal tackles when protection ends.
8. **Loose balls.** It predicts the exact ball path, takes the move that gives the earliest
   interception (exact for the next two moves), never touches its own unstoppable shot,
   always intercepts a ball heading into its own goal, and otherwise guards goal-side if it
   would lose the race.
9. **Obstacle-aware movement.** A visibility graph around the inflated obstacles provides
   shortest paths so the agent never gets stuck on obstacles.

## Files

```
submission.json                  launch command
requirements.txt                 empty: standard library only
team_bot/bot.py                  JSON-lines protocol loop
team_bot/policy.py               physics replica, planner and tactics
team_bot/models/planner_params.json   tuned strategy weights
```
