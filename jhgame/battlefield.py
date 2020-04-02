from .config import LIMIT_TIME


class BattleField:
    """Hold battle field information."""

    def __init__(self):
        self.scores = [0, 0]
        self.player_ids = ["", ""]
        self.num_players = 0
        self.puck = _Puck(x=0.5, y=0.5, vx=0, vy=0, hand=1, r=0.02, is_hidden=False)
        self.mallets = [
            _Mallet(left=0.4, right=0.6, y=0.9, hand=0, is_show=True),
            _Mallet(left=0.4, right=0.6, y=0.1, hand=0, is_show=True),
        ]
        self.game_flag = True
        self.time = LIMIT_TIME
        self.speedup_mode = False

    def set_items(self, bf_dict):
        """Update variables based on information contained in dictionary
        received by socket communication.

        Args:
            bf_dict (dict): Dictionary containing battlefield information
        """
        self.puck.x, self.puck.y = bf_dict["puck_coord"]
        self.puck.vx, self.puck.vy = bf_dict["puck_speed"]
        self.puck.r = bf_dict["puck_r"]
        self.puck.hand = bf_dict["puck_hand"]
        self.puck.is_hidden = bf_dict["puck_is_hidden"]

        self.mallets[0].left, self.mallets[0].right = bf_dict["p1_coord"]
        self.mallets[0].hand = bf_dict["p1_hand"]
        self.mallets[0].is_show = bf_dict["p1_is_show"]
        self.mallets[1].left, self.mallets[1].right = bf_dict["p2_coord"]
        self.mallets[1].hand = bf_dict["p2_hand"]
        self.mallets[1].is_show = bf_dict["p2_is_show"]

        self.game_flag = bf_dict["game_flag"]
        self.num_players = bf_dict["num_players"]
        self.player_ids = [bf_dict["p1_id"], bf_dict["p2_id"]]
        self.scores = bf_dict["scores"]
        self.speedup_mode = bf_dict["speedup_mode"]
        self.time = bf_dict["time"]


class _Puck:
    def __init__(self, x, y, vx, vy, hand, r, is_hidden):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.hand = hand
        self.r = r
        self.is_hidden = is_hidden


class _Mallet:
    def __init__(self, left, right, y, hand, is_show):
        self.left = left
        self.right = right
        self.y = y
        self.hand = hand
        self.is_show = is_show
