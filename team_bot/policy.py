from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

MOVE_VECTORS = {
    "STAY": (0.0, 0.0),
    "UP": (0.0, 1.0),
    "UP_RIGHT": (1.0, 1.0),
    "RIGHT": (1.0, 0.0),
    "DOWN_RIGHT": (1.0, -1.0),
    "DOWN": (0.0, -1.0),
    "DOWN_LEFT": (-1.0, -1.0),
    "LEFT": (-1.0, 0.0),
    "UP_LEFT": (-1.0, 1.0),
}


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    return (0.0, 0.0) if length == 0 else (vector[0] / length, vector[1] / length)


# Unit vectors computed exactly like the engine so simulated floats match.
UNITS = {name: _unit(vector) for name, vector in MOVE_VECTORS.items()}
MOVES = list(MOVE_VECTORS)
KICK_DIRECTIONS = [name for name in MOVES if name != "STAY"]
FLIP = {
    "STAY": "STAY", "UP": "DOWN", "DOWN": "UP", "LEFT": "LEFT", "RIGHT": "RIGHT",
    "UP_LEFT": "DOWN_LEFT", "UP_RIGHT": "DOWN_RIGHT",
    "DOWN_LEFT": "UP_LEFT", "DOWN_RIGHT": "UP_RIGHT",
}
TAN = math.sqrt(2.0) - 1.0
COS = math.cos(math.pi / 8.0)
INF = float("inf")



def octnorm(dx: float, dy: float) -> float:
    dx = abs(dx)
    dy = abs(dy)
    return dx + TAN * dy if dx >= dy else dy + TAN * dx


class Arena:
    """Exact ball physics plus planning helpers in the canonical frame."""

    def __init__(self, field: dict[str, Any], obstacles: list[tuple[float, float, float, float]],
                 kick_distances: list[float], ball_speed: float, ball_radius: float,
                 possession_radius: float) -> None:
        self.W = float(field["width"])
        self.H = float(field["height"])
        self.goal_left = (self.W - float(field["goal_width"])) / 2
        self.goal_right = self.goal_left + float(field["goal_width"])
        self.pr = float(field.get("player_radius", 3.0))
        self.speed = float(field.get("player_speed", 4.0))
        self.br = ball_radius
        self.bs = ball_speed
        self.poss_r = possession_radius
        self.kick_distances = kick_distances
        self.obstacles = obstacles
        self.catch = self.br + self.pr  # interception distance
        r = self.br
        self.ball_boxes = [(ox - r, ox + w + r, oy - r, oy + h + r, ox, oy, w, h) for ox, oy, w, h in obstacles]
        self._trace_cache: dict[tuple, tuple] = {}
        self._build_graph()

   
    def valid(self, x: float, y: float) -> bool:
        r = self.pr
        if x - r < 0 or x + r > self.W or y - r < 0 or y + r > self.H:
            return False
        for ox, oy, w, h in self.obstacles:
            cx = min(max(x, ox), ox + w)
            cy = min(max(y, oy), oy + h)
            if math.hypot(x - cx, y - cy) < r:
                return False
        return True

    def step_player(self, x: float, y: float, move: str) -> tuple[float, float]:
        ux, uy = UNITS[move]
        nx = x + ux * self.speed
        ny = y + uy * self.speed
        return (nx, ny) if self.valid(nx, ny) else (x, y)

    def resolve(self, me_old, me_move, op_old, op_move, ball, me_is_p1: bool):
        """Exact replica of the engine's simultaneous movement and contact
        resolution. Returns (my_new_position, opponent_new_position)."""
        ids = ("player_1", "player_2")
        me_id, op_id = (ids if me_is_p1 else ids[::-1])
        old = {me_id: me_old, op_id: op_old}
        moves = {me_id: me_move, op_id: op_move}
        speed = self.speed
        proposed = {}
        for pid in ids:
            ux, uy = UNITS[moves[pid]]
            cand = (old[pid][0] + ux * speed, old[pid][1] + uy * speed)
            proposed[pid] = cand if self.valid(*cand) else old[pid]
        a, b = proposed["player_1"], proposed["player_2"]
        minimum = 2 * self.pr
        if math.hypot(a[0] - b[0], a[1] - b[1]) < minimum:
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            sep = _unit((old["player_1"][0] - old["player_2"][0], old["player_1"][1] - old["player_2"][1]))
            if sep == (0.0, 0.0):
                sep = (1.0, 0.0)
            resolved = {
                "player_1": (mid[0] + sep[0] * self.pr, mid[1] + sep[1] * self.pr),
                "player_2": (mid[0] - sep[0] * self.pr, mid[1] - sep[1] * self.pr),
            }
            movement = sum(math.hypot(old[q][0] - resolved[q][0], old[q][1] - resolved[q][1]) for q in ids)
            if movement > 0.1 and all(self.valid(*resolved[q]) for q in ids):
                proposed = resolved
            else:
                solo = []
                for q in ids:
                    other = ids[1] if q == ids[0] else ids[0]
                    if self.valid(*proposed[q]) and math.hypot(proposed[q][0] - old[other][0], proposed[q][1] - old[other][1]) >= minimum:
                        progress = math.hypot(old[q][0] - ball[0], old[q][1] - ball[1]) - math.hypot(proposed[q][0] - ball[0], proposed[q][1] - ball[1])
                        solo.append((progress, q))
                if solo:
                    _, mover = max(solo, key=lambda item: (item[0], item[1]))
                    ux, uy = UNITS[moves[mover]]
                    cand = (old[mover][0] + ux * speed, old[mover][1] + uy * speed)
                    proposed = dict(old)
                    proposed[mover] = cand
                    if not self.valid(*cand):
                        proposed = dict(old)
                else:
                    alternatives = []
                    for q in ids:
                        other = ids[1] if q == ids[0] else ids[0]
                        for name in MOVES:
                            if name == "STAY":
                                continue
                            ux, uy = UNITS[name]
                            cand = (old[q][0] + ux * speed, old[q][1] + uy * speed)
                            if self.valid(*cand) and math.hypot(cand[0] - old[other][0], cand[1] - old[other][1]) >= minimum:
                                progress = math.hypot(old[q][0] - ball[0], old[q][1] - ball[1]) - math.hypot(cand[0] - ball[0], cand[1] - ball[1])
                                alternatives.append((progress, q, cand))
                    proposed = dict(old)
                    if alternatives:
                        _, mover, cand = max(alternatives, key=lambda item: (item[0], item[1], item[2]))
                        proposed[mover] = cand
        return proposed[me_id], proposed[op_id]

    def neighbours(self, x: float, y: float) -> list[tuple[str, float, float]]:
        out = []
        for move in MOVES:
            nx, ny = self.step_player(x, y, move)
            out.append((move, nx, ny))
        return out

    def trace_kick(self, px: float, py: float, direction: str, power: int):
        key = (px, py, direction, power)
        cached = self._trace_cache.get(key)
        if cached is not None:
            return cached
        ux, uy = UNITS[direction]
        clearance = self.pr + self.br + 0.05
        result = self.trace(px + ux * clearance, py + uy * clearance,
                            ux * self.bs, uy * self.bs, self.kick_distances[power - 1])
        if len(self._trace_cache) > 20000:
            self._trace_cache.clear()
        self._trace_cache[key] = result
        return result

    def trace(self, x: float, y: float, vx: float, vy: float, remaining: float, max_steps: int = 40):
        """Return (steps, outcome, end) where steps[k-1] lists substep ball centres
        during iteration k, outcome is 'for', 'against' or None."""
        steps: list[list[tuple[float, float]]] = []
        if remaining <= 0 or (vx == 0.0 and vy == 0.0):
            return steps, None, (x, y)
        W, H, r = self.W, self.H, self.br
        gl, gr = self.goal_left, self.goal_right
        limit = max(0.25, r * 0.45)
        boxes = self.ball_boxes
        while remaining > 0 and not (vx == 0.0 and vy == 0.0) and len(steps) < max_steps:
            travel = min(self.bs, remaining)
            substeps = max(1, math.ceil(travel / limit))
            sd = travel / substeps
            points: list[tuple[float, float]] = []
            steps.append(points)
            for _ in range(substeps):
                length = math.hypot(vx, vy)
                ux, uy = vx / length, vy / length
                px, py = x, y
                cx = px + ux * sd
                cy = py + uy * sd
                if gl <= cx <= gr:
                    if cy + r >= H:
                        return steps, "for", (cx, cy)
                    if cy - r <= 0:
                        return steps, "against", (cx, cy)
                if cx - r < 0 or cx + r > W:
                    vx = -vx
                    cx = min(max(cx, r), W - r)
                if cy - r < 0 or cy + r > H:
                    vy = -vy
                    cy = min(max(cy, r), H - r)
                for left, right, bottom, top, ox, oy, w, h in boxes:
                    if cx < left or cx > right or cy < bottom or cy > top:
                        continue
                    qx = min(max(cx, ox), ox + w)
                    qy = min(max(cy, oy), oy + h)
                    if math.hypot(cx - qx, cy - qy) >= r:
                        continue
                    crossed_x = px <= left or px >= right
                    crossed_y = py <= bottom or py >= top
                    if crossed_x:
                        vx = -vx
                    if crossed_y:
                        vy = -vy
                    if not crossed_x and not crossed_y:
                        vx, vy = -vx, -vy
                    length = math.hypot(vx, vy)
                    step = min(0.05, r / 10)
                    cx = px + vx / length * step
                    cy = py + vy / length * step
                    break
                x, y = cx, cy
                remaining = max(0.0, remaining - sd)
                points.append((x, y))
            if remaining <= 1e-9:
                remaining = 0.0
                break
        return steps, None, (x, y)

    def margin(self, steps, ox: float, oy: float, extra: int = 0, upto: int | None = None) -> float:
        """Guaranteed Euclidean clearance between the ball path and a player at
        (ox, oy) who gets (k + extra) moves before the ball's k-th iteration."""
        best = INF
        speed = self.speed
        catch = self.catch
        for k, points in enumerate(steps, start=1):
            if upto is not None and k > upto:
                break
            reach = speed * (k + extra)
            if reach < 0:
                reach = 0.0
            for x, y in points:
                dx = abs(x - ox)
                dy = abs(y - oy)
                g = dx + TAN * dy if dx >= dy else dy + TAN * dx
                if reach == 0.0:
                    d = math.hypot(x - ox, y - oy) - catch
                else:
                    d = (g - reach) * COS - catch
                if d < best:
                    best = d
        return best

    def first_touch(self, steps, ox: float, oy: float, extra: int = 0) -> tuple[int, float, float]:
        """Earliest iteration in which a player could possibly touch the ball."""
        speed = self.speed
        for k, points in enumerate(steps, start=1):
            reach = max(0.0, speed * (k + extra))
            for x, y in points:
                if math.hypot(x - ox, y - oy) - reach <= self.catch:
                    return k, x, y
        return 999, 0.0, 0.0

    def _build_graph(self) -> None:
        pad = self.pr + 0.6
        e = self.pr - 0.05
        self.blocks = [(ox - e, oy - e, ox + w + e, oy + h + e) for ox, oy, w, h in self.obstacles]
        nodes = []
        for ox, oy, w, h in self.obstacles:
            for nx, ny in ((ox - pad, oy - pad), (ox + w + pad, oy - pad), (ox - pad, oy + h + pad), (ox + w + pad, oy + h + pad)):
                if self.pr <= nx <= self.W - self.pr and self.pr <= ny <= self.H - self.pr:
                    nodes.append((nx, ny))
        self.nodes = nodes
        n = len(nodes)
        self.node_edges = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self.clear(nodes[i], nodes[j]):
                    d = octnorm(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
                    self.node_edges[i].append((j, d))
                    self.node_edges[j].append((i, d))

    def clear(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        ax, ay = a
        dx = b[0] - ax
        dy = b[1] - ay
        for x0, y0, x1, y1 in self.blocks:
            t0, t1 = 0.0, 1.0
            ok = False
            for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
                if p == 0:
                    if q < 0:
                        ok = True
                        break
                else:
                    t = q / p
                    if p < 0:
                        if t > t1:
                            ok = True
                            break
                        if t > t0:
                            t0 = t
                    else:
                        if t < t0:
                            ok = True
                            break
                        if t < t1:
                            t1 = t
            if not ok and t0 < t1:
                return False
        return True

    def free_point(self, tx: float, ty: float) -> tuple[float, float]:
        """Nearest point to (tx, ty) where a player centre can stand."""
        r = self.pr + 0.05
        tx = min(max(tx, r), self.W - r)
        ty = min(max(ty, r), self.H - r)
        for x0, y0, x1, y1 in self.blocks:
            if x0 < tx < x1 and y0 < ty < y1:
                options = [(tx - x0, x0 - 0.6, ty), (x1 - tx, x1 + 0.6, ty),
                           (ty - y0, tx, y0 - 0.6), (y1 - ty, tx, y1 + 0.6)]
                _, tx, ty = min(options)
        return tx, ty

    def path_field(self, tx: float, ty: float) -> list[float]:
        """Shortest octagonal path length from each graph node to the target."""
        tx, ty = self.free_point(tx, ty)
        n = len(self.nodes)
        dist = [INF] * n
        for i, node in enumerate(self.nodes):
            if self.clear(node, (tx, ty)):
                dist[i] = octnorm(node[0] - tx, node[1] - ty)
        done = [False] * n
        for _ in range(n):
            best, bi = INF, -1
            for i in range(n):
                if not done[i] and dist[i] < best:
                    best, bi = dist[i], i
            if bi < 0:
                break
            done[bi] = True
            for j, d in self.node_edges[bi]:
                if best + d < dist[j]:
                    dist[j] = best + d
        return dist

    def path_len(self, x: float, y: float, tx: float, ty: float, field: list[float] | None) -> float:
        tx, ty = self.free_point(tx, ty)
        if not self.obstacles or self.clear((x, y), (tx, ty)):
            return octnorm(x - tx, y - ty)
        if field is None:
            field = self.path_field(tx, ty)
        best = INF
        for i, node in enumerate(self.nodes):
            if field[i] < INF and self.clear((x, y), node):
                d = octnorm(x - node[0], y - node[1]) + field[i]
                if d < best:
                    best = d
        return best if best < INF else octnorm(x - tx, y - ty) + 40.0

    def move_toward(self, x: float, y: float, tx: float, ty: float,
                    avoid: tuple[float, float] | None = None) -> str:
        tx, ty = self.free_point(tx, ty)
        field = None
        if self.obstacles and not self.clear((x, y), (tx, ty)):
            field = self.path_field(tx, ty)
        best_move, best_cost = "STAY", INF
        for move, nx, ny in self.neighbours(x, y):
            cost = self.path_len(nx, ny, tx, ty, field)
            if move != "STAY" and (nx, ny) == (x, y):
                cost += 50.0
            if avoid is not None and math.hypot(nx - avoid[0], ny - avoid[1]) < 2 * self.pr:
                cost += 8.0
            if cost < best_cost - 1e-9:
                best_cost, best_move = cost, move
        return best_move



    def clearance(self, steps, centers, upto: int | None = None) -> float:
        """Guaranteed gap between the ball and a player whose possible positions
        during the ball's first iteration are exactly `centers`; later
        iterations use the octagonal reach bound around those centres."""
        best = INF
        catch = self.catch
        speed = self.speed
        for k, points in enumerate(steps, start=1):
            if upto is not None and k > upto:
                break
            if k == 1:
                for x, y in points:
                    for cx, cy in centers:
                        d = math.hypot(x - cx, y - cy) - catch
                        if d < best:
                            best = d
            else:
                reach = speed * (k - 1)
                for x, y in points:
                    for cx, cy in centers:
                        dx = abs(x - cx)
                        dy = abs(y - cy)
                        g = dx + TAN * dy if dx >= dy else dy + TAN * dx
                        d = (g - reach) * COS - catch
                        if d < best:
                            best = d
            if best < -6.0:
                return best
        return best


    def first_reach(self, steps, centers):
        """Earliest (k, x, y) where a player with first-step positions `centers`
        could touch the ball (same bounds as clearance), or None."""
        catch = self.catch
        speed = self.speed
        for k, points in enumerate(steps, start=1):
            for x, y in points:
                for cx, cy in centers:
                    if k == 1:
                        d = math.hypot(x - cx, y - cy) - catch
                    else:
                        dx = abs(x - cx)
                        dy = abs(y - cy)
                        g = dx + TAN * dy if dx >= dy else dy + TAN * dx
                        d = (g - speed * (k - 1)) * COS - catch
                    if d <= 0:
                        return k, x, y
        return None


def _sig(x: float) -> float:
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


DEFAULT_PARAMS = {
    # Shot success model: P(goal) = sigmoid((clearance - shot_center) / shot_scale)
    "shot_center": -2.5,
    "shot_scale": 1.2,
    # Lookahead (next-turn) shots use a looser, continuous reach bound.
    "future_center": -1.5,
    "future_scale": 1.5,
    "discount": 0.93,
    # Value of keeping the ball without a shot.
    "hold_base": 0.05,
    "hold_progress": 0.45,
    "pressure": 0.25,
    "clock": 0.02,
    "turnover_base": -0.15,
    "turnover_depth": -0.35,
    # Defence
    "guard_distance": 10.0,
    "guard_weight": 0.012,
    "press_bonus": 0.08,
    "tackle_bonus": 0.3,
    "danger_floor": 0.0,
    "time_budget": 0.5,
    "lookahead_weight": 0.6,
    "keeper_depth": 12.0,
    "keeper_mix": 0.5,
    "counter_weight": 0.5,
    "commit_penalty": 2.0,
    "learn_rate": 0.8,
    "theta_min": -5.0,
    "theta_max": 0.0,
}


class Policy:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})
        self.arena: Arena | None = None
        self.arena_key = None
        self.deadline = INF
        self._counter_cache: dict = {}
        self._powers = [1, 2, 3]
        self.theta = self.params["shot_center"]
        self.pending_shot = None
        self.last_shot_clearance = None

    @classmethod
    def load(cls, path: Path) -> "Policy":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        return cls(raw.get("params") if isinstance(raw, dict) else None)

    # -------------------------------------------------------------- frame setup
    def _setup(self, observation: dict[str, Any]):
        state = observation["state"]
        field = state["field"]
        H = float(field["height"])
        flip = observation.get("attack_direction", "UP") != "UP"
        obstacles = []
        for o in state.get("obstacles", []):
            ox, oy, w, h = float(o["x"]), float(o["y"]), float(o["width"]), float(o["height"])
            obstacles.append((ox, H - oy - h if flip else oy, w, h))
        space = observation.get("action_space", {}).get("kick", {})
        powers = [int(p) for p in space.get("power", [1, 2, 3])]
        self._powers = powers
        key = (flip, tuple(obstacles), H, float(field["width"]), tuple(powers))
        if key != self.arena_key:
            kick_distances = [32.0 * p for p in powers]
            self.arena = Arena(field, obstacles, kick_distances, 8.0, 1.5, 5.0)
            self.arena_key = key

        def tf(px: float, py: float) -> tuple[float, float]:
            return (px, H - py) if flip else (px, py)

        me_raw = state["players"][observation["player_id"]]
        op_raw = state["players"][observation["opponent_id"]]
        ball = state["ball"]
        vel = ball.get("velocity", {}) or {}
        vy = float(vel.get("y", 0.0))
        return {
            "flip": flip,
            "me": tf(float(me_raw["x"]), float(me_raw["y"])),
            "op": tf(float(op_raw["x"]), float(op_raw["y"])),
            "ball": tf(float(ball["x"]), float(ball["y"])),
            "bv": (float(vel.get("x", 0.0)), -vy if flip else vy),
            "remaining": float(ball.get("remaining_kick_distance", 0.0) or 0.0),
            "owner": "me" if ball.get("possession") == observation["player_id"] else (
                "op" if ball.get("possession") == observation["opponent_id"] else None),
            "steps": int(ball.get("possession_steps", 0) or 0),
            "powers": powers,
            "is_p1": observation["player_id"] == "player_1",
            "score_me": int(state.get("score", {}).get(observation["player_id"], 0)),
            "iteration": int(state.get("iteration", 0)),
        }

    def choose_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.deadline = time.perf_counter() + self.params["time_budget"]
        self._counter_cache = {}
        try:
            ctx = self._setup(observation)
            self.learn(ctx)
            if ctx["owner"] == "me":
                self.last_shot_clearance = None
                action = self.attack(ctx)
                if "kick" in action and self.last_shot_clearance is not None:
                    self.pending_shot = (self.last_shot_clearance, ctx["score_me"], ctx["iteration"])
            elif ctx["owner"] == "op":
                action = self.defend(ctx)
            else:
                action = self.loose(ctx)
        except Exception as error:  # never let a planning bug cost an action
            print(f"planner error: {error!r}", file=sys.stderr, flush=True)
            return self.emergency(observation)
        if ctx["flip"]:
            action = dict(action)
            action["move"] = FLIP[action["move"]]
            if "kick" in action:
                action["kick"] = {"direction": FLIP[action["kick"]["direction"]], "power": action["kick"]["power"]}
        return action

    @staticmethod
    def emergency(observation: dict[str, Any]) -> dict[str, Any]:
        """Minimal safe behaviour if planning ever fails: chase / shoot forward."""
        try:
            state = observation["state"]
            me = state["players"][observation["player_id"]]
            ball = state["ball"]
            up = observation.get("attack_direction", "UP") == "UP"
            if ball.get("possession") == observation["player_id"]:
                d = "UP" if up else "DOWN"
                return {"move": d, "kick": {"direction": d, "power": 3}}
            dx = ball["x"] - me["x"]
            dy = ball["y"] - me["y"]
            h = "RIGHT" if dx > 1 else "LEFT" if dx < -1 else ""
            v = "UP" if dy > 1 else "DOWN" if dy < -1 else ""
            return {"move": (v + "_" + h) if v and h else (v or h or "STAY")}
        except Exception:
            return {"move": "STAY"}

    def p_goal(self, clearance: float) -> float:
        p = self.params
        return _sig((clearance - self.theta) / p["shot_scale"])

    def learn(self, ctx) -> None:
        """Online logistic update of the shot model from our resolved shots."""
        pending = self.pending_shot
        if pending is None:
            return
        clearance, score_before, iteration = pending
        outcome = None
        if ctx["score_me"] > score_before:
            outcome = 1.0
        elif ctx["owner"] == "op":
            outcome = 0.0
        elif ctx["owner"] == "me" or ctx["iteration"] - iteration > 20:
            self.pending_shot = None
            return
        if outcome is None:
            return
        self.pending_shot = None
        if clearance > 0.5:
            return  # guaranteed shots carry no information about the opponent
        p = self.params
        prob = _sig((clearance - self.theta) / p["shot_scale"])
        self.theta -= p["learn_rate"] * (outcome - prob)
        self.theta = min(p["theta_max"], max(p["theta_min"], self.theta))

    def p_goal_future(self, clearance: float) -> float:
        p = self.params
        return _sig((clearance - p["future_center"]) / p["future_scale"])

    def turnover(self, y: float) -> float:
        p = self.params
        return p["turnover_base"] + p["turnover_depth"] * (1.0 - y / self.arena.H)

    def hold(self, x: float, y: float, ox: float, oy: float, s: int = 0) -> float:
        """Value of keeping the ball at (x, y) with no shot taken this turn."""
        p = self.params
        arena = self.arena
        gx = min(max(x, arena.goal_left + 4), arena.goal_right - 4)
        progress = 1.0 - min(1.0, octnorm(x - gx, arena.H - y) / arena.H)
        value = p["hold_base"] + p["hold_progress"] * progress
        gap = math.hypot(x - ox, y - oy)
        # Opponent able to reach tackling contact soon while our protection ends.
        if s >= 2 and gap < 2 * arena.pr + 0.15 + 2 * arena.speed:
            value -= p["pressure"] * (2 * arena.pr + 0.15 + 2 * arena.speed - gap) / (2 * arena.speed)
        # Possession clock: forced release after the limit.
        value -= p["clock"] * max(0, s - 5)
        return value

    def shot_value(self, pg: float, y: float) -> float:
        return pg + (1.0 - pg) * self.turnover(y)

    def counter_threat(self, ix: float, iy: float, mx: float, my: float, extra: int) -> float:
        """Probability-like threat of an immediate opponent shot from (ix, iy)
        while we start from (mx, my) with `extra` additional moves of reach."""
        key = (round(ix, 1), round(iy, 1), round(mx, 1), round(my, 1), extra)
        cached = self._counter_cache.get(key)
        if cached is not None:
            return cached
        arena = self.arena
        shots = []
        for direction in KICK_DIRECTIONS:
            for power in self._powers:
                steps, outcome, _ = arena.trace_kick(ix, iy, direction, power)
                if outcome == "against":
                    shots.append(steps)
        if not shots:
            worst = 0.0
        elif extra == 0:
            # We commit our move before seeing their shot (simultaneous turns).
            worst = INF
            for _, nx, ny in arena.neighbours(mx, my):
                local = 0.0
                for steps in shots:
                    pg = self.p_goal(arena.clearance(steps, [(nx, ny)]))
                    if pg > local:
                        local = pg
                        if local > 0.98:
                            break
                if local < worst:
                    worst = local
        else:
            worst = 0.0
            for steps in shots:
                pg = self.p_goal(arena.margin(steps, mx, my, extra - 1) - self.params["commit_penalty"])
                if pg > worst:
                    worst = pg
        self._counter_cache[key] = worst
        return worst

    def lost_ball(self, steps, op_centers, mx, my, fallback_y: float) -> float:
        """Value when the opponent wins the ball somewhere on this path."""
        p = self.params
        hit = self.arena.first_reach(steps, op_centers)
        if hit is None:
            return self.turnover(fallback_y)
        k, ix, iy = hit
        threat = self.counter_threat(ix, iy, mx, my, max(0, k - 1))
        return self.turnover(iy) - p["counter_weight"] * threat

    def shots_from(self, px: float, py: float, powers: list[int]):
        arena = self.arena
        shots = []
        for direction in KICK_DIRECTIONS:
            for power in powers:
                steps, outcome, _ = arena.trace_kick(px, py, direction, power)
                if outcome == "for" and steps:
                    if any(math.hypot(x - px, y - py) <= arena.catch for x, y in steps[0]):
                        continue  # the kicker would trap its own shot
                    shots.append((direction, power, steps))
        return shots

    def attack(self, ctx) -> dict[str, Any]:
        arena = self.arena
        p = self.params
        mx, my = ctx["me"]
        ox, oy = ctx["op"]
        s = ctx["steps"]
        powers = ctx["powers"]
        contact = 2 * arena.pr + 0.15
        op_next = arena.neighbours(ox, oy)
        op_centers = [(x, y) for _, x, y in op_next]
        shot_cache: dict[tuple[float, float], list] = {}

        def shots_at(x, y):
            key = (x, y)
            got = shot_cache.get(key)
            if got is None:
                got = self.shots_from(x, y, powers)
                shot_cache[key] = got
            return got

        def best_future(x, y, qx, qy, extra):
            best = -INF
            for direction, power, steps in shots_at(x, y):
                v = self.shot_value(self.p_goal_future(arena.margin(steps, qx, qy, extra)), y)
                if v > best:
                    best = v
            return best

        options = []  # (value, action)
        my_moves = [(m, x, y) for m, x, y in arena.neighbours(mx, my) if m == "STAY" or (x, y) != (mx, my)]
        for move, nx, ny in my_moves:
            contact_risk = any(math.hypot(cx - nx, cy - ny) < 2 * arena.pr for cx, cy in op_centers)
            branches = None
            if contact_risk:
                branches = [arena.resolve((mx, my), move, (ox, oy), omove, (mx, my), ctx["is_p1"])
                            for omove, _, _ in op_next]
            for direction, power, steps in shots_at(nx, ny):
                if branches is None:
                    c = arena.clearance(steps, op_centers)
                else:
                    c = INF
                    for (bx2, by2), (qx, qy) in branches:
                        if (bx2, by2) == (nx, ny):
                            st2 = steps
                        else:
                            st2, out2, _ = arena.trace_kick(bx2, by2, direction, power)
                            if out2 != "for":
                                c = -INF
                                break
                        c2 = arena.clearance(st2, [(qx, qy)])
                        if c2 < c:
                            c = c2
                    if c == -INF:
                        c = -20.0
                pg = self.p_goal(c)
                lost = self.lost_ball(steps, op_centers, nx, ny, ny) if pg < 0.97 else self.turnover(ny)
                options.append((pg + (1.0 - pg) * lost,
                                {"move": move, "kick": {"direction": direction, "power": power}, "_c": c}))

        if s + 1 < 10:
            second = {}
            for move, nx, ny in my_moves:
                second[(nx, ny)] = [(x, y) for m, x, y in arena.neighbours(nx, ny) if m == "STAY" or (x, y) != (nx, ny)]
            for move, nx, ny in my_moves:
                if time.perf_counter() > self.deadline:
                    break
                worst = INF
                for omove, onx, ony in op_next:
                    gap = math.hypot(onx - nx, ony - ny)
                    # Overlaps are resolved to exactly 2 * radius apart: still a tackle.
                    if s >= 3 and omove != "STAY" and gap <= contact:
                        value = self.turnover(ny)
                    else:
                        if gap < 2 * arena.pr:
                            onx, ony = ox, oy
                        value = -INF
                        threat = s + 1 >= 3
                        for x2, y2 in second[(nx, ny)]:
                            # Next turn: shoot from (x2, y2) while the opponent moves from (onx, ony).
                            v = best_future(x2, y2, onx, ony, 0)
                            if s + 2 < 10:
                                if threat and math.hypot(x2 - onx, y2 - ony) <= contact + arena.speed:
                                    keep = self.turnover(y2)
                                else:
                                    keep = max(best_future(x2, y2, onx, ony, 1) * p["discount"],
                                               self.hold(x2, y2, onx, ony, s + 2))
                                keep *= p["discount"]
                                if keep > v:
                                    v = keep
                            if v > value:
                                value = v
                        if value == -INF:
                            value = self.turnover(ny)
                    if value < worst:
                        worst = value
                        if worst <= -1.0:
                            break
                options.append((worst, {"move": move}))

        if s + 1 >= 10 or s >= 3:
            options.extend(self.release_options(ctx))
        if not options:
            return {"move": "STAY"}
        options.sort(key=lambda item: item[0], reverse=True)
        best = dict(options[0][1])
        if "_c" in best:
            self.last_shot_clearance = best.pop("_c")
        return best

    def release_options(self, ctx):
        """Non-scoring kicks valued by who wins the race to the ball."""
        arena = self.arena
        mx, my = ctx["me"]
        ox, oy = ctx["op"]
        out = []
        for move, nx, ny in arena.neighbours(mx, my):
            if move != "STAY" and (nx, ny) == (mx, my):
                continue
            for direction in KICK_DIRECTIONS:
                for power in ctx["powers"]:
                    steps, outcome, end = arena.trace_kick(nx, ny, direction, power)
                    if outcome is not None or not steps:
                        continue
                    t_op, px_op, py_op = self.touch_time(steps, end, ox, oy, 0)
                    t_me, px_me, py_me = self.touch_time(steps, end, nx, ny, -1)
                    lead = t_op - t_me
                    pw = _sig((lead - 0.5) / 0.8)
                    keep = self.hold(px_me, py_me, ox, oy, 0) * self.params["discount"] ** max(1.0, t_me)
                    if pw < 0.97:
                        threat = self.counter_threat(px_op, py_op, nx, ny, max(0, int(t_op) - 1))
                        lost = self.turnover(py_op) - self.params["counter_weight"] * threat
                    else:
                        lost = self.turnover(py_op)
                    value = pw * keep + (1 - pw) * lost
                    out.append((value, {"move": move, "kick": {"direction": direction, "power": power}}))
        return out

    def touch_time(self, steps, end, px, py, extra):
        """Earliest (iteration, x, y) at which a player could touch the ball."""
        arena = self.arena
        speed = arena.speed
        for k, points in enumerate(steps, start=1):
            reach = speed * max(0, k + extra)
            for x, y in points:
                if reach == 0:
                    if math.hypot(x - px, y - py) <= arena.catch:
                        return float(k), x, y
                elif octnorm(x - px, y - py) * COS - reach <= arena.catch:
                    return float(k), x, y
        ex, ey = end
        need = max(0.0, octnorm(ex - px, ey - py) - arena.poss_r) / speed
        return max(float(len(steps)), need - extra), ex, ey

    # ------------------------------------------------------------------ defence
    def defend(self, ctx) -> dict[str, Any]:
        arena = self.arena
        p = self.params
        mx, my = ctx["me"]
        ox, oy = ctx["op"]
        s = ctx["steps"]
        threats = []
        for omove, onx, ony in arena.neighbours(ox, oy):
            for direction in KICK_DIRECTIONS:
                for power in ctx["powers"]:
                    steps, outcome, _ = arena.trace_kick(onx, ony, direction, power)
                    if outcome == "against":
                        threats.append(steps)
        tx, ty = self.guard_point(ox, oy)
        field = arena.path_field(tx, ty) if arena.obstacles and not arena.clear((mx, my), (tx, ty)) else None
        contact = 2 * arena.pr + 0.15
        # Shots the opponent could take next turn after dribbling one step.
        future = []
        if p["lookahead_weight"] > 0:
            for omove, onx, ony in arena.neighbours(ox, oy):
                if omove != "STAY" and (onx, ony) == (ox, oy):
                    continue
                shots = []
                for direction in KICK_DIRECTIONS:
                    for power in ctx["powers"]:
                        steps, outcome, _ = arena.trace_kick(onx, ony, direction, power)
                        if outcome == "against":
                            shots.append(steps)
                future.append((onx, ony, shots))
        best_move, best_cost = "STAY", INF
        for move, nx, ny in arena.neighbours(mx, my):
            if move != "STAY" and (nx, ny) == (mx, my):
                continue
            danger = 0.0
            for steps in threats:
                c = arena.clearance(steps, [(nx, ny)])
                pg = self.p_goal(c)
                if pg > danger:
                    danger = pg
            later = 0.0
            for onx, ony, shots in future:
                worst = 0.0
                for steps in shots:
                    pg = self.p_goal_future(arena.margin(steps, nx, ny, 0))
                    if pg > worst:
                        worst = pg
                        if worst > 0.98:
                            break
                if worst > later:
                    later = worst
            cost = danger + p["lookahead_weight"] * later
            cost += p["guard_weight"] * arena.path_len(nx, ny, tx, ty, field)
            gap = math.hypot(nx - ox, ny - oy)
            if s >= 3 and move != "STAY" and gap <= contact:
                cost -= p["tackle_bonus"]  # a non-kicking holder is dispossessed now
            elif s >= 2 and gap <= contact + arena.speed:
                cost -= p["press_bonus"]
            if cost < best_cost:
                best_cost, best_move = cost, move
        return {"move": best_move}

    def guard_point(self, ox: float, oy: float) -> tuple[float, float]:
        """Goal-side guard spot: on the line from our goal centre to the ball,
        either close to the attacker or deep like a goalkeeper."""
        arena = self.arena
        p = self.params
        gx, gy = arena.W / 2, 0.0
        dx, dy = ox - gx, oy - gy
        dist = math.hypot(dx, dy) or 1.0
        near = max(0.0, dist - p["guard_distance"])
        deep = min(p["keeper_depth"], dist * 0.5)
        w = p["keeper_mix"]
        depth = (1 - w) * near + w * deep
        return gx + dx / dist * depth, gy + dy / dist * depth

    def loose(self, ctx) -> dict[str, Any]:
        arena = self.arena
        mx, my = ctx["me"]
        ox, oy = ctx["op"]
        bx, by = ctx["ball"]
        vx, vy = ctx["bv"]
        steps, outcome, end = arena.trace(bx, by, vx, vy, ctx["remaining"])
        if not steps:
            end = (bx, by)
        if outcome == "for" and arena.clearance(steps, [(x, y) for _, x, y in arena.neighbours(ox, oy)]) > 0:
            return {"move": self.stay_clear(mx, my, steps)}
        t_op = self.arrival(steps, end, ox, oy)[0]
        choice = self.chase_move(steps, end, mx, my, avoid_goal=(outcome == "for"))
        t_me = choice[0]
        if outcome == "against" or t_me <= t_op or t_me < 2:
            return {"move": choice[1]}
        # We lose the race: take the goal-side guard spot for where it is collected.
        _, px, py = self.arrival(steps, end, ox, oy)
        tx, ty = self.guard_point(px, py)
        return {"move": arena.move_toward(mx, my, tx, ty, (ox, oy))}

    def arrival(self, steps, end, px, py):
        """Optimistic earliest iteration a player could touch the ball."""
        arena = self.arena
        for k, points in enumerate(steps, start=1):
            reach = arena.speed * k
            for x, y in points:
                if octnorm(x - px, y - py) * COS - reach <= arena.catch:
                    return float(k), x, y
        ex, ey = end
        need = max(0.0, arena.path_len(px, py, ex, ey, None) - arena.poss_r) / arena.speed
        return max(float(len(steps)), math.ceil(need)), ex, ey

    def chase_move(self, steps, end, mx, my, avoid_goal=False):
        """Pick the first move that leads to the earliest touch of the ball."""
        arena = self.arena
        best = (INF, "STAY", INF)
        ex, ey = end
        field = None
        if arena.obstacles and not arena.clear((mx, my), (ex, ey)):
            field = arena.path_field(ex, ey)
        for move, nx, ny in arena.neighbours(mx, my):
            if move != "STAY" and (nx, ny) == (mx, my):
                continue
            t = INF
            slack = INF
            if steps:
                if any(math.hypot(x - nx, y - ny) <= arena.catch for x, y in steps[0]):
                    t = 1.0
                    slack = 0.0
                else:
                    second = [(sx, sy) for _, sx, sy in arena.neighbours(nx, ny)]
                    if len(steps) > 1 and any(math.hypot(x - sx, y - sy) <= arena.catch
                                              for x, y in steps[1] for sx, sy in second):
                        t = 2.0
                        slack = 0.0
                    else:
                        for k in range(3, len(steps) + 1):
                            reach = arena.speed * (k - 1)
                            for x, y in steps[k - 1]:
                                gap = octnorm(x - nx, y - ny) - reach
                                if gap <= arena.catch - 0.6:
                                    t = float(k)
                                    slack = gap
                                    break
                            if t < INF:
                                break
            if t == INF:
                d = arena.path_len(nx, ny, ex, ey, field)
                t = max(float(len(steps)), 1.0 + math.ceil(max(0.0, d - arena.poss_r + 0.3) / arena.speed))
                slack = d
            if (t, slack) < (best[0], best[2]):
                best = (t, move, slack)
        return best[0], best[1]

    def stay_clear(self, mx, my, steps) -> str:
        arena = self.arena
        best_move, best_value = "STAY", -INF
        for move, nx, ny in arena.neighbours(mx, my):
            clearance = min((math.hypot(x - nx, y - ny) for x, y in steps[0]), default=INF)
            value = min(clearance, 12.0) - 0.05 * octnorm(nx - arena.W / 2, ny - arena.H * 0.4)
            if clearance <= arena.catch + 0.5:
                value -= 100.0
            if value > best_value:
                best_value, best_move = value, move
        return best_move
