import math
import random
import time

from . import config
from .battlefield import BattleField
from .gamelog import get_module_logger

LOGGER = get_module_logger(__name__)


class JankenHockeyServer:
    SPEED_UP_RATE = 1.7

    def __init__(self, level):
        """Janken Hockey Game Server.

        Args:
            level (str): Game level
        """
        self.level = level

        self._bf = BattleField()
        self._start_time = None
        self._before_time = None
        self._p1_reflectable = True
        self._p2_reflectable = True
        self._starting = False

        # _v_puck_norm: "rock", "scissors", "paper", "first"
        # _mallet_half: wide (win), normal (draw), narrow (lose)
        if self.level == "easy":
            self._v_puck_norm = [0.4, 0.25, 0.15]
            self._mallet_half = [0.3, 0.2, 0.1]
        elif self.level == "normal":
            self._v_puck_norm = [0.7, 0.55, 0.4]
            self._mallet_half = [0.25, 0.14, 0.05]
        elif self.level == "hard":
            self._v_puck_norm = [0.8, 0.65, 0.5]
            self._mallet_half = [0.2, 0.08, 0.01]
        elif self.level == "god":
            self._v_puck_norm = [1.0, 0.8, 0.6]
            self._mallet_half = [0.1, 0.05, -1]
        else:
            raise ValueError(f"Unknown level {level}")

        self._v_puck_norm.append(self._v_puck_norm[-1])  # fist speed == paper's speed

        self._mallet_hands = [1, 1]  # initial size == normal
        self._unlock_time = 0

    def call(self, player_id, info_dict, player_ids):
        """Server executes this function every time.
        - update game flag
        - update puck
        - update player's mallet

        Args:
            player_id (str): User's UUID
            info_dict (dict): Dictionary containing mallet x-coordinate and
                hand type
            player_ids (list): List containing player ids

        Returns:
            dict: Dictionary containing battle field information
        """
        assert 0 <= info_dict["coord"] <= 1
        assert info_dict["type"] in (-1, 0, 1, 2)

        self._bf.player_ids = player_ids
        self._bf.num_players = len(player_ids)

        player = player_ids.index(player_id)
        hand = info_dict["type"]
        mallet_x = info_dict["coord"]

        current_time = time.time()

        self._update_mallet(player, hand, mallet_x)

        # set CPU mallet
        if self._bf.num_players == 1:
            self._update_mallet(
                player=1, hand=int(current_time) // 5 % 3, mallet_x=1 - self._bf.puck.x,
            )

        self._update_time(current_time)
        if self._starting:
            self._bf.time = self._unlock_time - current_time

        if self._before_time is None:
            self._before_time = current_time

        if self._unlock_time <= current_time and self._bf.game_flag:
            self._update_puck(current_time)

        if not self._bf.speedup_mode and not self._starting and self._bf.time < 30:
            self._bf.speedup_mode = True
            self._bf.puck.vx *= self.SPEED_UP_RATE
            self._bf.puck.vy *= self.SPEED_UP_RATE
            self._v_puck_norm[:3] = [v * self.SPEED_UP_RATE for v in self._v_puck_norm[:3]]

        self._before_time = current_time
        return self._get_bf_dict()

    def _update_time(self, current_time):
        """Update time and check time limit.

        Args:
            current_time (float): Current UNIX time
        """
        if self._start_time is None:
            if self._bf.num_players == 1 and self._bf.puck.vx == 0 and self._bf.puck.vy == 0:
                self._set_puck(player=0)
            elif self._bf.num_players == 2:
                if not self._starting:
                    # start PvP game
                    self._p1_reflectable = True
                    self._p2_reflectable = True
                    self._unlock_time = current_time + 5
                    self._starting = True
                    self._set_puck(player=0 if random.random() <= 0.5 else 1)
                    self._bf.scores = [0, 0]
                elif self._unlock_time <= current_time:
                    # end PvP game
                    self._starting = False
                    self._bf.time = config.LIMIT_TIME
                    self._start_time = current_time
        else:
            self._bf.time = config.LIMIT_TIME - (current_time - self._start_time)

        if self._bf.time < 0:
            self._bf.time = 0
            self._bf.game_flag = False
            self._bf.speedup_mode = False

    def _update_mallet(self, player, hand, mallet_x):
        """Updated mallet position, length and hand type.

        Args:
            player (int): Player index (0 or 1)
            hand (int): Hand type index (-1, 0, 1, or 2)
            mallet_x (float): Mallet x-coordinate.
        """
        assert player in (0, 1)
        assert hand in (-1, 0, 1, 2)
        assert 0 <= mallet_x <= 1

        # if hand not detected, replace it with last hand
        if hand != -1:
            self._bf.mallets[player].hand = hand

        # paper's length is -1 if god mode
        if self._mallet_half[2] == -1:
            self._bf.mallets[player].is_show = True

        # judge reflection
        if (player == 0 and self._p1_reflectable) or (player == 1 and self._p2_reflectable):
            self._mallet_hands[player] = self._update_mallet_hand(self._bf.mallets[player].hand, self._bf.puck.hand)
            if self._mallet_half[2] == -1 and self._mallet_hands[player] == 2:
                self._bf.mallets[player].is_show = False

        half = self._mallet_half[self._mallet_hands[player]]

        self._bf.mallets[player].left = max([0, mallet_x - half])
        self._bf.mallets[player].right = min([1, mallet_x + half])
        if player == 1:
            self._bf.mallets[player].left, self._bf.mallets[player].right = (
                1 - self._bf.mallets[player].right,
                1 - self._bf.mallets[player].left,
            )

    @staticmethod
    def _update_mallet_hand(player_hand, puck_hand):
        """Judge rock-paper-scissors.

        Args:
            player_hand (int): Hand type index (0, 1, or 2)
            puck_hand (int): Hand type index (0, 1, or 2)

        Returns:
            int: Result
        """
        if puck_hand == 3:  # any hand wins against white
            return 0
        elif player_hand == puck_hand:  # draw
            return 1
        elif player_hand == (puck_hand + 1) % 3:  # lose
            return 2
        return 0  # win

    def _update_puck(self, current_time):
        """Judge reflection and goal.

        Args:
            current_time (float): Current UNIX time
        """
        LOGGER.debug(
            "[puck] (x,y)=(%.2f,%.2f) (vx,vy)=(%2f,%2f)",
            self._bf.puck.x,
            self._bf.puck.y,
            self._bf.puck.vx,
            self._bf.puck.vy,
        )

        self._bf.puck.x += self._bf.puck.vx * (current_time - self._before_time)
        self._bf.puck.y += self._bf.puck.vy * (current_time - self._before_time)

        # setting the disappearing magic sphere
        self._bf.puck.is_hidden = self._bf.puck.hand == 2 and 0.3 < self._bf.puck.y < 0.7

        top = self._bf.puck.y - self._bf.puck.r
        bottom = self._bf.puck.y + self._bf.puck.r
        left = self._bf.puck.x - self._bf.puck.r
        right = self._bf.puck.x + self._bf.puck.r

        # check in order of goal, mallet, wall
        if self._p1_reflectable and self._hit_p1_mallet(bottom, bottom - self._bf.puck.vy):
            self._bf.puck.vx, self._bf.puck.vy = self._update_velocity(self._bf.mallets[0].hand)
            self._p1_reflectable = False
            self._p2_reflectable = True
            self._bf.puck.hand = self._bf.mallets[0].hand
            self._bf.puck.y = 2 * self._bf.mallets[0].y - bottom - self._bf.puck.r
        elif self._p2_reflectable and self._hit_p2_mallet(top, top - self._bf.puck.vy):
            self._bf.puck.vx, self._bf.puck.vy = self._update_velocity(self._bf.mallets[1].hand)
            self._p1_reflectable = True
            self._p2_reflectable = False
            self._bf.puck.hand = self._bf.mallets[1].hand
            self._bf.puck.y = 2 * self._bf.mallets[1].y - top + self._bf.puck.r
        elif self._p1_goal(bottom):
            self._goal(0, current_time)
        elif self._p2_goal(top):
            self._goal(1, current_time)

        if self._hit_left(left):
            self._bf.puck.vx *= -1
            self._bf.puck.x = -left + self._bf.puck.r

        elif self._hit_right(right):
            self._bf.puck.vx *= -1
            self._bf.puck.x = 2 - right - self._bf.puck.r

    def _hit_left(self, left):
        return left <= 0 < left - self._bf.puck.vx

    def _hit_right(self, right):
        return right - self._bf.puck.vx < 1 <= right

    @staticmethod
    def _p1_goal(bottom):
        return bottom < 0

    @staticmethod
    def _p2_goal(top):
        return top > 1

    def _hit_p1_mallet(self, bottom, before_bottom):
        LOGGER.debug(
            "[mallet1] (%.2f,%.2f) - (%.2f,%2f)",
            self._bf.mallets[0].left,
            self._bf.mallets[0].y,
            self._bf.mallets[0].right,
            self._bf.mallets[0].y,
        )

        if (
            before_bottom < self._bf.mallets[0].y <= bottom
            and self._bf.mallets[0].left <= self._bf.puck.x <= self._bf.mallets[0].right
        ):
            return True

        if (self._bf.puck.x - self._bf.mallets[0].left) ** 2 + (
            self._bf.puck.y - self._bf.mallets[0].y
        ) ** 2 <= self._bf.puck.r ** 2:
            return True
        if (self._bf.puck.x - self._bf.mallets[0].right) ** 2 + (
            self._bf.puck.y - self._bf.mallets[0].y
        ) ** 2 <= self._bf.puck.r ** 2:
            return True
        return False

    def _hit_p2_mallet(self, top, before_top):
        LOGGER.debug(
            "[mallet2] (%.2f,%.2f) - (%.2f,%2f)",
            self._bf.mallets[1].left,
            self._bf.mallets[1].y,
            self._bf.mallets[1].right,
            self._bf.mallets[1].y,
        )

        if (
            top <= self._bf.mallets[1].y < before_top
            and self._bf.mallets[1].left <= self._bf.puck.x <= self._bf.mallets[1].right
        ):
            return True

        if (self._bf.puck.x - self._bf.mallets[1].left) ** 2 + (
            self._bf.puck.y - self._bf.mallets[1].y
        ) ** 2 <= self._bf.puck.r ** 2:
            return True
        if (self._bf.puck.x - self._bf.mallets[1].right) ** 2 + (
            self._bf.puck.y - self._bf.mallets[1].y
        ) ** 2 <= self._bf.puck.r ** 2:
            return True
        return False

    def _goal(self, player, current_time):
        self._unlock_time = current_time + 1
        self._bf.scores[player] += 1
        self._p1_reflectable = True
        self._p2_reflectable = True
        self._set_puck(1 - player)

    def _set_puck(self, player):
        """Initialize puck.

        Args:
            player (int): Player index (0 or 1)
        """
        assert player in (0, 1)

        self._bf.puck.hand = 3  # white mallet
        self._bf.puck.x = 0.5
        self._bf.puck.y = 0.6 if player == 0 else 0.4
        self._bf.puck.vx = (self._bf.mallets[player].left + self._bf.mallets[player].right) / 2 - self._bf.puck.x
        self._bf.puck.vy = self._bf.mallets[player].y - self._bf.puck.y

        puck_norm = math.sqrt(self._bf.puck.vx ** 2 + self._bf.puck.vy ** 2)
        self._bf.puck.vx *= self._v_puck_norm[self._bf.puck.hand] / puck_norm
        self._bf.puck.vy *= self._v_puck_norm[self._bf.puck.hand] / puck_norm

    def _update_velocity(self, hand):
        """Updated pack speed by hand.

        Args:
            hand (int): Hand type index (0, 1, or 2)

        Returns:
            tuple: Tuple containing updated speed
        """
        assert hand in (0, 1, 2)

        old_norm = self._v_puck_norm[self._bf.puck.hand]
        new_norm = self._v_puck_norm[hand]

        if hand == 1:  # hand == scissors, middle speed and random reflection
            theta = random.uniform(math.pi / 9, math.pi * 8 / 9)  # 20 - 160
            return (
                new_norm * math.cos(theta),
                math.copysign(new_norm * math.sin(theta), -self._bf.puck.vy),
            )
        # hand == rock, high speed
        # hand == paper, slow speed (and vanish on client)
        return (
            self._bf.puck.vx * new_norm / old_norm,
            -self._bf.puck.vy * new_norm / old_norm,
        )

    def _get_bf_dict(self):
        """Get the battle field information to send to the client via socket
        communication.

        Returns:
            dict: Dictionary containing battle field information
        """
        bf_dict = {
            "p1_coord": [self._bf.mallets[0].left, self._bf.mallets[0].right],
            "p2_coord": [self._bf.mallets[1].left, self._bf.mallets[1].right],
            "p1_hand": self._bf.mallets[0].hand,
            "p2_hand": self._bf.mallets[1].hand,
            "p1_is_show": self._bf.mallets[0].is_show,
            "p2_is_show": self._bf.mallets[1].is_show,
            "puck_coord": [self._bf.puck.x, self._bf.puck.y],
            "puck_speed": [self._bf.puck.vx, self._bf.puck.vy],
            "puck_r": self._bf.puck.r,
            "puck_hand": self._bf.puck.hand,
            "scores": self._bf.scores,
            "time": self._bf.time,
            "game_flag": self._bf.game_flag,
            "num_players": self._bf.num_players,
            "p1_id": self._bf.player_ids[0] if self._bf.num_players == 1 else "",
            "p2_id": self._bf.player_ids[1] if self._bf.num_players == 2 else "",
            "puck_is_hidden": self._bf.puck.is_hidden,
            "speedup_mode": self._bf.speedup_mode,
        }
        return bf_dict
