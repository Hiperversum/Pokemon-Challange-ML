"""Simulation features: take the snapshot of the battle at turn 30 and play out the rest.

The other groups describe the state at turn 30 column by column (HP, KOs, types, speed...)
and a linear model can only add those columns up. A player reasons differently: "my Snorlax
at 60% against their Zapdos at 30%: who moves first, how many hits does each one need,
and who comes in next?". This group does exactly that with simplified Gen 2 rules:

- real level-100 stats from the base stats (max DVs and stat experience);
- the Gen 2 damage formula: physical/special attack vs defense, STAB x1.5,
  type effectiveness from the supplied type chart, average random factor;
- each Pokemon attacks with the best of the moves seen in this battle or usually used by
  its species in the training battles;
- the faster Pokemon (boosts and paralysis included) hits first;
- sleep and freeze skip turns, poison/burn/toxic deal damage every turn, Leftovers heals;
- Player 2's Pokemon never seen are replaced by an average Pokemon;
- duel after duel: the loser faints and its player sends the best counter left,
  until one team is empty.

Everything is learned from train.jsonl only (species stats, types, usual moves and the
power of the attacks whose power is missing in the data); the only other input is the
supplied type chart. The game rules (formulas, status effects) are knowledge, not data.
"""

import collections
import math

import numpy as np

from ... import config

SIDES = ("p1", "p2")
# Attacks recorded with power 0 in the data: their power is estimated from the training battles
ESTIMATED_POWER = ("return", "frustration")
# Attacks that remove a fixed number of HP points: the amount is estimated from the training battles
FIXED_DAMAGE = ("seismictoss", "nightshade")
# Moves that heal the user: turns where the defender used one are not used for the estimates
HEALING_MOVES = {"rest", "recover", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis"}
# Attacks left out of the simulation: one-shot, conditional or too weak to matter
IGNORED_MOVES = {
    "explosion", "selfdestruct", "counter", "mirrorcoat", "bide", "present", "reversal", "flail",
    "superfang", "psywave", "magnitude", "dragonrage", "sonicboom", "rapidspin", "thief", "pursuit",
    # Hidden Power: its real type is hidden in the data, so its damage against a given target
    # cannot be known (a "neutral" estimate of 178 overestimates it and lowers the CV accuracy)
    "hiddenpower",
}
# Moves given to the "average" Pokemon that replaces the unrevealed Player 2 Pokemon
AVERAGE_MOVES = {"bodyslam", "thunderbolt", "earthquake", "icebeam"}
MAX_DUEL_TURNS = 40
MAX_DUELS = 12
# A move is "usual" for a species if it is at least this share of its attacking turns
USUAL_MOVE_SHARE = 0.05


####################################################################################
# GAME RULES
####################################################################################

def stage_multiplier(stage):
    """Multiplier of a stat boost stage (-6..+6)."""
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def real_stats(base, level=100):
    """Gen 2 stats at the given level with max DVs (15) and max stat experience."""
    stats = {s: math.floor(((base[s] + 15) * 2 + 63) * level / 100) + 5
             for s in config.STATS if s != "hp"}
    stats["hp"] = math.floor(((base["hp"] + 15) * 2 + 63) * level / 100) + level + 10
    return stats


def is_attack(move):
    """True for the attacks used in the simulation."""
    if move["category"] == "status" or move["name"] in IGNORED_MOVES:
        return False
    return bool(move["base_power"]) or move["name"] in ESTIMATED_POWER + FIXED_DAMAGE


def base_damage(level, power, attack, defense):
    """Gen 2 damage formula without STAB, type and random factor."""
    return (2 * level / 5 + 2) * power * attack / defense / 50 + 2


####################################################################################
# POKEMON IN THE SIMULATION
####################################################################################

class Fighter:
    """One Pokemon with its current HP fraction, status, boosts and attacks."""

    def __init__(self, name, base, types, hp, status, moves, level=100, boosts=None, leftovers=True):
        self.name = name
        self.types = types
        self.hp = hp
        self.status = status
        self.moves = moves
        self.level = level
        self.boosts = dict(boosts or {})
        self.leftovers = leftovers
        self.stats = real_stats(base, level)

    def stat(self, name):
        """Stat with boosts, paralysis (speed x0.25) and burn (attack x0.5)."""
        value = self.stats[name] * stage_multiplier(self.boosts.get(name, 0))
        if name == "spe" and self.status == "par":
            value *= 0.25
        if name == "atk" and self.status == "brn":
            value *= 0.5
        return value


def damage_per_hit(attacker, defender, type_chart, move_info):
    """Best expected damage of one hit, as a fraction of the defender's max HP."""
    best = 0.0
    for name in attacker.moves:
        move = move_info.get(name)
        if move is None or not move.get("base_power"):
            continue
        if "fixed_damage" in move:
            damage = move["fixed_damage"]
        else:
            power = move["base_power"]
            physical = move["category"] == "physical"
            attack = attacker.stat("atk" if physical else "spa")
            defense = defender.stat("def" if physical else "spd")
            move_type = move["type"]
            mult = type_chart.multiplier(move_type, defender.types) if defender.types else 1.0
            stab = 1.5 if move_type in attacker.types else 1.0
            damage = base_damage(attacker.level, power, attack, defense) * stab * mult * 0.925
            damage *= min(move.get("accuracy") or 1.0, 1.0)
        best = max(best, damage / defender.stats["hp"])
    # A paralysed Pokemon cannot move one turn out of four
    return best * 0.75 if attacker.status == "par" else best


def residual(fighter, toxic_counter):
    """HP change at the end of a turn: poison/burn/toxic damage and Leftovers."""
    chip = 0.0
    if fighter.status in ("psn", "brn"):
        chip = 1 / 8
    elif fighter.status == "tox":
        chip = toxic_counter / 16
    heal = 1 / 16 if fighter.leftovers else 0.0
    return heal - chip


def duel(a, b, type_chart, move_info):
    """Fight a against b. Return (a wins, a HP left, b HP left)."""
    hp = {"a": a.hp, "b": b.hp}
    hit = {"a": damage_per_hit(a, b, type_chart, move_info), "b": damage_per_hit(b, a, type_chart, move_info)}
    # Turns lost: sleep ~2 more turns, freeze ~10 turns (10% thaw chance per turn in Gen 2)
    lost = {key: 2 if f.status == "slp" else 10 if f.status == "frz" else 0
            for key, f in (("a", a), ("b", b))}
    order = ("a", "b") if a.stat("spe") >= b.stat("spe") else ("b", "a")
    toxic = {"a": 0, "b": 0}
    fighters = {"a": a, "b": b}

    for _ in range(MAX_DUEL_TURNS):
        for attacker in order:
            target = "b" if attacker == "a" else "a"
            if lost[attacker] > 0:
                lost[attacker] -= 1
            else:
                hp[target] -= hit[attacker]
            if hp[target] <= 0:
                return attacker == "a", max(hp["a"], 0.0), max(hp["b"], 0.0)
        for key in ("a", "b"):
            if fighters[key].status == "tox":
                toxic[key] += 1
            hp[key] = min(hp[key] + residual(fighters[key], toxic[key]), 1.0)
            if hp[key] <= 0:
                return key == "b", max(hp["a"], 0.0), max(hp["b"], 0.0)
    # Nobody fainted: the healthier one is counted as the winner
    return hp["a"] >= hp["b"], hp["a"], hp["b"]


####################################################################################
# SIMULATION GROUP
####################################################################################

class BattleSimulation:
    """Learn species knowledge on the training battles, then simulate each battle's end."""

    def __init__(self, type_chart):
        self.type_chart = type_chart
        self.base, self.types, self.usual_moves, self.move_info = {}, {}, {}, {}
        self.average_base = None

    #---------------------------
    # FIT: what we know about each species, from the training battles only
    #---------------------------

    def fit(self, battles):
        """Learn base stats, types and usual attacks of every species."""
        moves = collections.defaultdict(collections.Counter)
        p2_seen = collections.Counter()
        for battle in battles:
            for pokemon in battle["p1_team_details"] + [battle["p2_lead_details"]]:
                if pokemon["name"] not in self.base:
                    self.base[pokemon["name"]] = {s: pokemon[f"base_{s}"] for s in config.STATS}
                    self.types[pokemon["name"]] = [t for t in pokemon["types"] if t != "notype"]
            for turn in battle["battle_timeline"]:
                p2_seen[turn["p2_pokemon_state"]["name"]] += 1
                for side in SIDES:
                    move = turn.get(f"{side}_move_details")
                    if move and is_attack(move):
                        moves[turn[f"{side}_pokemon_state"]["name"]][move["name"]] += 1
                        self.move_info[move["name"]] = dict(move)
        for species, counts in moves.items():
            total = sum(counts.values())
            self.usual_moves[species] = {m for m, c in counts.items() if c >= USUAL_MOVE_SHARE * total}
        # The "average" Player 2 Pokemon: base stats weighted by how often each species appears
        known = [(n, c) for n, c in p2_seen.items() if n in self.base]
        weight = sum(c for _, c in known)
        self.average_base = {s: sum(self.base[n][s] * c for n, c in known) / weight for s in config.STATS}
        self.estimate_missing_powers(battles)
        return self

    def estimate_missing_powers(self, battles):
        """Estimate from the observed damage the power of the attacks recorded with power 0.

        For every use of the attack where the defender stayed on the field, its HP drop
        (plus the 1/16 of Leftovers) is turned into HP points and the damage formula is
        inverted to get the power. The median over all uses is kept. The same procedure
        recovers the known powers of the other attacks within 1-3% (Body Slam 85.8 vs 85,
        Earthquake 101.4 vs 100), so it can be trusted for the missing ones
        (Return 104, Frustration 102, Seismic Toss 101 HP).
        """
        samples = collections.defaultdict(list)
        for battle in battles:
            timeline = battle["battle_timeline"]
            for prev, turn in zip(timeline, timeline[1:]):
                for side, other in (("p1", "p2"), ("p2", "p1")):
                    move = turn.get(f"{side}_move_details")
                    if not move or move["name"] not in ESTIMATED_POWER + FIXED_DAMAGE:
                        continue
                    attacker = turn[f"{side}_pokemon_state"]
                    before, after = prev[f"{other}_pokemon_state"], turn[f"{other}_pokemon_state"]
                    other_move = turn.get(f"{other}_move_details")
                    # Only clean hits: same defender, not fainted, no healing, no boosts
                    if (before["name"] != after["name"] or after["status"] == "fnt"
                            or attacker["name"] not in self.base or after["name"] not in self.base
                            or (other_move and other_move["name"] in HEALING_MOVES)
                            or any(attacker["boosts"].values()) or any(after["boosts"].values())):
                        continue
                    drop = before["hp_pct"] - after["hp_pct"] + 1 / 16
                    if drop <= 0.07:  # missed or blocked
                        continue
                    defender_stats = real_stats(self.base[after["name"]])
                    hp_points = drop * defender_stats["hp"]
                    if move["name"] in FIXED_DAMAGE:
                        samples[move["name"]].append(hp_points)
                        continue
                    physical = move["category"] == "physical"
                    attack = real_stats(self.base[attacker["name"]])["atk" if physical else "spa"]
                    defense = defender_stats["def" if physical else "spd"]
                    mult = self.type_chart.multiplier(move["type"], self.types[after["name"]])
                    if mult == 0:
                        continue
                    stab = 1.5 if move["type"] in self.types[attacker["name"]] else 1.0
                    # Invert: hp_points = base_damage(100, P, A, D) * stab * mult * 0.925
                    samples[move["name"]].append((hp_points / (stab * mult * 0.925) - 2) * 50 * defense / (attack * 42))

        self.estimated = {name: float(np.median(values)) for name, values in samples.items() if values}
        for name, value in self.estimated.items():
            if name in self.move_info:
                if name in FIXED_DAMAGE:
                    self.move_info[name]["fixed_damage"] = value
                    self.move_info[name]["base_power"] = 1  # marks it as an attack
                else:
                    self.move_info[name]["base_power"] = value

    #---------------------------
    # SNAPSHOT AT TURN 30
    #---------------------------

    def snapshot(self, battle):
        """Return the alive Pokemon of each side and the two Pokemon on the field."""
        hp = {"p1": {p["name"]: 1.0 for p in battle["p1_team_details"]}, "p2": {}}
        status = {"p1": {p["name"]: "nostatus" for p in battle["p1_team_details"]}, "p2": {}}
        used = {side: collections.defaultdict(set) for side in SIDES}
        last = {}
        for turn in battle["battle_timeline"]:
            for side in SIDES:
                state = turn[f"{side}_pokemon_state"]
                hp[side][state["name"]] = state["hp_pct"]
                status[side][state["name"]] = state["status"]
                last[side] = state
                move = turn.get(f"{side}_move_details")
                if move and is_attack(move):
                    used[side][state["name"]].add(move["name"])

        items = {p["name"]: p.get("item") for p in battle["p1_team_details"]}
        levels = {p["name"]: p["level"] for p in battle["p1_team_details"]}
        teams = {}
        for side in SIDES:
            team = []
            for name in hp[side]:
                if status[side][name] == "fnt" or name not in self.base:
                    continue
                moves = used[side][name] | self.usual_moves.get(name, set()) or {"bodyslam"}
                boosts = last[side]["boosts"] if last[side]["name"] == name else {}
                # Player 1's items are known; almost every Player 2 Pokemon holds Leftovers
                leftovers = items.get(name) == "leftovers" if side == "p1" else True
                team.append(Fighter(name, self.base[name], self.types[name], hp[side][name],
                                    status[side][name], moves, levels.get(name, 100), boosts, leftovers))
            teams[side] = team
        for i in range(max(config.TEAM_SIZE - len(hp["p2"]), 0)):
            teams["p2"].append(Fighter(f"unknown{i}", self.average_base, [], 1.0, "nostatus", AVERAGE_MOVES))
        active = {side: next((f for f in teams[side] if f.name == last[side]["name"]), None) for side in SIDES}
        return teams, active

    #---------------------------
    # PLAY OUT THE REST OF THE BATTLE
    #---------------------------

    def play_out(self, teams, active):
        """Duel after duel until a team is empty. Return (P1 wins, P1 HP left, P2 HP left, P1 alive, P2 alive)."""
        alive = {side: list(teams[side]) for side in SIDES}
        current = {}
        for side in SIDES:
            current[side] = active[side] or (max(alive[side], key=lambda f: f.hp) if alive[side] else None)

        for _ in range(MAX_DUELS):
            if not alive["p1"] or not alive["p2"]:
                break
            p1, p2 = current["p1"], current["p2"]
            p1_wins, p1.hp, p2.hp = duel(p1, p2, self.type_chart, self.move_info)
            p1.boosts, p2.boosts = {}, {}
            loser, winner = ("p2", "p1") if p1_wins else ("p1", "p2")
            alive[loser].remove(current[loser])
            if not alive[loser]:
                break
            # The loser's player sends the Pokemon that beats the winner with the most HP left
            opponent = current[winner]

            def answer(fighter):
                wins, own_hp, opp_hp = duel(fighter, opponent, self.type_chart, self.move_info)
                return wins, own_hp - opp_hp

            current[loser] = max(alive[loser], key=answer)

        p1_left = sum(f.hp for f in alive["p1"])
        p2_left = sum(f.hp for f in alive["p2"])
        p1_wins = not alive["p2"] or (bool(alive["p1"]) and p1_left >= p2_left)
        return int(p1_wins), p1_left, p2_left, len(alive["p1"]), len(alive["p2"])

    #---------------------------
    # FEATURES
    #---------------------------

    def features(self, battle):
        """Return the 7 simulation features of one battle."""
        teams, active = self.snapshot(battle)
        feats = {}

        # Every alive P1 Pokemon against every alive P2 Pokemon, one duel each (from current HP)
        wins, margins = [], []
        for a in teams["p1"]:
            for b in teams["p2"]:
                won, hp_a, hp_b = duel(a, b, self.type_chart, self.move_info)
                wins.append(won)
                margins.append(hp_a - hp_b)
        feats["sim_pair_winrate"] = float(np.mean(wins)) if wins else 0.0
        feats["sim_pair_margin"] = float(np.mean(margins)) if margins else 0.0

        # The duel between the two Pokemon on the field
        if active["p1"] is not None and active["p2"] is not None:
            won, hp_a, hp_b = duel(active["p1"], active["p2"], self.type_chart, self.move_info)
            feats["sim_active_duel_win"] = int(won)
            feats["sim_active_duel_margin"] = hp_a - hp_b
        else:
            feats["sim_active_duel_win"] = 0.5
            feats["sim_active_duel_margin"] = 0.0

        # The whole rest of the battle
        p1_wins, p1_left, p2_left, p1_alive, p2_alive = self.play_out(teams, active)
        feats["sim_p1_wins"] = p1_wins
        feats["sim_hp_margin"] = p1_left - p2_left
        feats["sim_alive_margin"] = p1_alive - p2_alive
        return feats
