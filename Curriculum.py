import ffai
from ffai.core.model import Square, Action, Agent, D3, D6, D8, BBDie
from ffai.core.table import ActionType, Skill
import ffai.core.procedure as FFAI_procs
import random
from random import randint
from pdb import set_trace
from copy import deepcopy
from collections import Iterable
import numpy as np
from ffai.core.game import *
from ffai.core.load import *

from ffai.ai.bots.random_bot import RandomBot

from scipy.special import softmax

HISTORY_SIZE = 200


class Lecture:
    def __init__(self, name, max_level, delta_level=0.1):
        self.name = name
        self.level = 0
        self.max_level = max_level
        self.delta_level = delta_level
        self.exceptions_thrown = 0

        assert delta_level > 0

    def increase_diff(self):
        self.level = min(self.max_level, self.level + self.delta_level)

    def decrease_diff(self):
        self.level = max(0, self.level - self.delta_level)

    def increase_level(self):
        self.level += 1 * (self.level < self.max_level)

    def decrease_level(self):
        self.level -= 1 * (0 < self.level)

    def get_diff(self):
        return min(self.level, self.max_level) / self.max_level

    def get_level(self):
        return min(int(self.level), self.max_level)

    def reset_game(self, config):
        """ 
        :paran config: integer of pitch size, (currently 3,5,7,11)
        :return: return a fully initialized game object, with opp_agent initialized
        """
        raise NotImplementedError("Must be overridden by subclass")

    def evaluate(self, game, drive_over):
        """
        :param game: game object to be judged
        :param drive_over: is set to True last time this function is called. 
        :return: a LectureOutcome object  
        """
        raise NotImplementedError("Must be overridden by subclass")

    def allowed_fail_rate(self):
        """
        Not sure how to use this. TODO TBD 
        """
        return 0

    def get_name(self):
        return self.name


class LectureOutcome:
    def __init__(self, lecture, win, draw=None):

        self.lect_type = type(lecture)
        self.name = lecture.name
        self.steps = 0
        self.level = lecture.get_level()

        if win:
            self.result = 1
        elif draw is not None and draw:
            self.result = 0
        else:
            self.result = -1


class LectureHistory:
    def __init__(self, lecture):
        self.lecture = lecture
        self.latest_hundred = np.zeros((HISTORY_SIZE, 1))
        # self.rewards        = np.zeros( (HISTORY_SIZE,1) )
        self.latest_level = np.zeros((HISTORY_SIZE, 1))
        self.index = 0
        self.episodes = 0
        self.steps = 0
        self.max_acheived = -1
        self.history_filled = False

    def log(self, outcome):
        assert self.lecture.name == outcome.name
        assert type(self.lecture) == outcome.lect_type
        i = self.index

        self.latest_hundred[i] = outcome.result
        self.latest_level[i] = outcome.level
        self.episodes += 1
        self.steps += outcome.steps

        self.index += 1
        if self.index >= HISTORY_SIZE:
            self.history_filled = True
            self.index = 0

        # Increase difficulty?
        if self.lecture.get_level() == outcome.level:
            if outcome.result == 1:
                self.lecture.increase_diff()
            else:
                self.lecture.decrease_diff()

        if outcome.result == 1 and self.max_acheived < outcome.level:
            self.max_acheived = outcome.level

    def report(self, with_name=False):
        lvl = str(self.lecture.get_level())
        max_lvl = self.lecture.max_level
        avg = self.latest_hundred.mean()
        # prob        = self.lec_prob_soft[lec_index]
        # reward      = self.rewards[lec_index,:].mean()

        s = f"ep={self.episodes}, steps={self.steps}, lvl= {lvl} ({self.max_acheived})/{max_lvl}), avg={avg}"
        return s


class Academy:

    def __init__(self, lectures):
        self.lect_histo = []
        self.add_lecture(lectures)

    def _update_probs(self):
        try:
            self.lec_prob = np.array([lecture.episodes / lecture.steps for lecture in self.lect_histo])
            self.lec_prob /= self.lec_prob.sum()
        except:
            self.lec_prob = np.ones((self.num_lects,)) / self.num_lects
        assert round(sum(self.lec_prob), 3) == 1.0

    def get_next_lecture(self):
        return np.random.choice(self.lect_histo, 1, p=self.lec_prob)[0].lecture

    def add_lecture(self, lectures):
        if type(lectures) != list:
            lectures = [lectures]

        for l in lectures:
            self.lect_histo.append(LectureHistory(l))

        self.num_lects = len(self.lect_histo)
        self.lec_prob = np.zeros((self.num_lects,))
        self._update_probs()

        # Assert unique lectures
        self.lect_names = [l.name for l in lectures]
        for name in self.lect_names:
            assert self.lect_names.count(name) == 1

        self._update_probs()

    def log_training(self, outcome):
        name = outcome.name
        index = self.lect_names.index(name)
        self.lect_histo[index].log(outcome)
        self._update_probs()

    def report(self, filename=None):
        # render plots 

        max_name_len = max([len(l.lecture.name) for l in self.lect_histo])

        s = ""
        for l in self.lect_histo:
            name = l.lecture.name
            extra_spaces = max_name_len - len(name)

            s += l.lecture.name + ": " + " " * extra_spaces
            s += l.report() + "\n"

        return s


game_turn_memoized = {}


def get_empty_game_turn(config="ff-11", turn=0, clear_board=True, hometeam="human", awayteam="human", away_agent=None):
    if type(config) == str:
        config = load_config(config)
    config.competition_mode = False
    config.fast_mode = True

    pitch_size = config.pitch_max

    key = f"{hometeam} {awayteam} {pitch_size} {turn}"
    if key in game_turn_memoized:
        game = deepcopy(game_turn_memoized[key])

        if away_agent is not None:
            game.replace_away_agent(away_agent)
        return game

    D3.FixedRolls = []
    D6.FixedRolls = [3, 3, 3, 3, 3, 3, 3, 3, 3]  # No crazy kickoff or broken armors
    D8.FixedRolls = []
    BBDie.FixedRolls = []

    ruleset = load_rule_set(config.ruleset)

    size_suffix = f"-{pitch_size}" if pitch_size != 11 else ""
    home = load_team_by_filename(hometeam + size_suffix, ruleset, board_size=pitch_size)
    away = load_team_by_filename(awayteam + size_suffix, ruleset, board_size=pitch_size)
    game = Game(1, home, away, Agent("human1", human=True), Agent("human2", human=True), config)
    game.init()

    random_agent = RandomBot("home")

    if turn > 0:
        game.step(Action(ActionType.START_GAME))
        game.step(Action(ActionType.HEADS))
        game.step(Action(ActionType.KICK))
        game.step(Action(ActionType.SETUP_FORMATION_ZONE))
        game.step(Action(ActionType.END_SETUP))
        game.step(Action(ActionType.SETUP_FORMATION_WEDGE))
        game.step(Action(ActionType.END_SETUP))
        while type(game.get_procedure()) is not Turn or game.is_quick_snap() or game.is_blitz():
            action = random_agent.act(game)
            game.step(action)

        while game.state.home_team.state.turn != turn:
            game.step(Action(ActionType.END_TURN))

        if clear_board:
            game.clear_board()

    if away_agent is not None:
        game.replace_away_agent(away_agent)
    else:
        game.replace_away_agent(random_agent)
    D6.FixedRolls = []
    game_turn_memoized[key] = deepcopy(game)

    return game


def get_home_players(game):
    num = min(game.config.pitch_max, len(game.state.home_team.players))
    return random.sample(game.state.home_team.players, num)


def get_away_players(game):
    num = min(game.config.pitch_max, len(game.state.away_team.players))
    return random.sample(game.state.away_team.players, num)


def get_boundary_square(game, steps, from_position):
    steps = int(steps)

    if steps == 0:
        if game.state.pitch.board[from_position.y][from_position.x] is None:
            return from_position
        else:
            steps += 1

            # return a position that is 'steps' away from 'from_position'
    # checks are done so it's square is available 
    board_x_max = len(game.state.pitch.board[0]) - 2
    board_y_max = len(game.state.pitch.board) - 2

    assert steps > 0

    avail_squares = steps * 8

    squares_per_side = 2 * steps

    i = 0
    while True:
        i += 1
        assert i < 5000

        sq_index = randint(0, avail_squares - 1)
        steps_along_side = sq_index % squares_per_side

        # up, including left corner 
        if sq_index // squares_per_side == 0:
            dx = - steps + steps_along_side
            dy = - steps
        # right, including upper corner
        elif sq_index // squares_per_side == 1:
            dx = + steps
            dy = - steps + steps_along_side
        # down, including right corner
        elif sq_index // squares_per_side == 2:
            dx = + steps - steps_along_side
            dy = + steps
            # left, including lower corner
        elif sq_index // squares_per_side == 3:
            dx = - steps
            dy = + steps - steps_along_side
        else:
            assert False

        position = Square(from_position.x + dx, from_position.y + dy)
        x = position.x
        y = position.y

        if x < 1 or x > board_x_max or y < 1 or y > board_y_max:
            continue

        if game.state.pitch.board[y][x] is None:  # it should y first, don't ask.
            break

    return position


def scatter_ball(game, steps, from_position):
    # scatters ball a certain amount of steps away for original position 
    # checks are done so it's not out of bounds or on a player 
    if steps > 0:
        pos = get_boundary_square(game, steps, from_position)
    else:
        pos = from_position

    game.get_ball().move_to(pos)
    game.get_ball().is_carried = False


def set_player_state(player, p_used=None, p_down=None):
    if p_down is not None:
        player.state.up = not random.random() < p_down

    if p_used is not None and player.state.up:
        player.state.used = random.random() < p_used


def move_player_within_square(game, player, x, y, give_ball=False, p_used=None, p_down=None):
    # places the player at a random position within the given square. 

    assert isinstance(give_ball, bool)

    board_x_max = len(game.state.pitch.board[0]) - 2
    board_y_max = len(game.state.pitch.board) - 2

    xx = sorted(x) if isinstance(x, Iterable) else (x, x)
    yy = sorted(y) if isinstance(y, Iterable) else (y, y)

    x_min = max(xx[0], 1)
    x_max = min(xx[1], board_x_max)
    y_min = max(yy[0], 1)
    y_max = min(yy[1], board_y_max)

    assert x_min <= x_max
    assert y_min <= y_max

    i = 0

    while True:
        i += 1
        assert i < 5000

        x = randint(x_min, x_max)
        y = randint(y_min, y_max)

        # if x < 1 or x > board_x_max or y < 1 or y > board_y_max:
        #    continue 

        if game.state.pitch.board[y][x] is None:
            break

    game.put(player, Square(x, y))
    if give_ball == True:
        game.get_ball().move_to(player.position)
        game.get_ball().is_carried = True

    set_player_state(player, p_used=p_used, p_down=p_down)


def move_player_out_of_square(game, player, x, y, p_used=None, p_down=None):
    # places the player at a random position that is not in the given square. 

    xx = x if isinstance(x, Iterable) else (x, x)
    yy = y if isinstance(y, Iterable) else (y, y)

    x_min = xx[0]
    x_max = xx[1]
    y_min = yy[0]
    y_max = yy[1]

    board_x_max = len(game.state.pitch.board[0]) - 2
    board_y_max = len(game.state.pitch.board) - 2

    i = 0

    while True:
        x = randint(1, board_x_max)
        y = randint(1, board_y_max)

        if x_min <= x <= x_max and y_min <= y <= y_max:
            i += 1
            assert i < 5000
            continue

        if game.state.pitch.board[y][x] is None:
            break

    game.put(player, Square(x, y))
    set_player_state(player, p_used=p_used, p_down=p_down)


def move_players_out_of_square(game, players, x, y, p_used=None, p_down=None):
    for p in players:
        move_player_out_of_square(game, p, x, y, p_used=p_used, p_down=p_down)


def swap_game(game):
    moved_players = []
    board_x_max = len(game.state.pitch.board[0]) - 2

    player_to_move = get_home_players(game) + get_away_players(game)
    for p in player_to_move:

        if p in moved_players:
            continue

        old_x = p.position.x
        new_x = 27 - old_x

        potential_swap_p = game.state.pitch.board[p.position.y][new_x]
        if potential_swap_p is not None:
            game.move(potential_swap_p, Square(0, 0))

        game.move(p, Square(new_x, p.position.y))

        if potential_swap_p is not None:
            game.move(potential_swap_p, Square(old_x, p.position.y))
            moved_players.append(potential_swap_p)

            # ball_pos = game.get_ball().position

    # ball_new_x  = 27-ball_pos.x
    # ball_y      = ball_pos.y

    # game.get_ball().move_to( Square(ball_new_x, ball_y) )

    # assert game.get_ball().position.x ==  ball_new_x
    pass
