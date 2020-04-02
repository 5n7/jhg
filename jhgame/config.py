import json
import os.path as osp

import torch.cuda
from screeninfo import get_monitors

with open(osp.join(osp.abspath(osp.dirname(__file__)), "../config.json")) as f:
    _user_config = json.load(f)

# --------------------------------------------------
# common
# --------------------------------------------------

FPS = _user_config["game.fps"]
IS_FULLSCREEN = _user_config["game.is_fullscreen"]
SERVER_HOST = _user_config["server.host"]

# if _resolution type is "auto", set game resolution automatically
_resolution_type = _user_config["game.resolution_type"]
if _resolution_type == "auto":
    m = get_monitors()[0]
    RESOLUTION_X, RESOLUTION_Y = m.width, m.height
elif _resolution_type == "manual":
    RESOLUTION_X, RESOLUTION_Y = (
        _user_config["game.resolution_x"],
        _user_config["game.resolution_y"],
    )
else:
    raise ValueError(f'In config.json, "game.resolution_type" must be auto/manual.')

# --------------------------------------------------
# color
# --------------------------------------------------

COLORS = {
    "black": (0, 0, 0),
    "blue": (3, 169, 244),
    "blue_gray": (96, 125, 139),
    "gray": (158, 158, 158),
    "green": (139, 195, 74),
    "orange": (255, 152, 0),
    "purple": (103, 58, 183),
    "red": (233, 30, 99),
    "white": (255, 255, 255),
    "yellow": (255, 193, 7),
}

HAND_COLORS = {
    "rock": COLORS["red"],
    "scissors": COLORS["yellow"],
    "paper": COLORS["blue"],
    "first": COLORS["white"],
}

# --------------------------------------------------
# game information
# --------------------------------------------------

GAME_INFO = {
    "title": "Janken Hockey Game",
    "icon_path": osp.join(osp.abspath(osp.dirname(__file__)), "../images/icon.png"),
}

# --------------------------------------------------
# menu
# --------------------------------------------------

MENU = {"font_size": int(RESOLUTION_Y / 60 + 12)}

HELP = {
    "title": "Janken Hockey Game",
    "overview": "This is a hockey game with rock-paper-scissors.",
    "usage": ["(1) launch server", "(2) launch client", "(3.a) create room", "(3.b) select room", "(4) start"],
}

ABOUT = {
    "title": "Janken Hockey Game",
    "organization": ["Tokyo University of Science", "Taniguchi Laboratory"],
    "developers": ["Kazuhiro Akizawa", "Hiroto Kobayashi", "Shunta Komatsu", "Hiroyuki Yokota"],
}

# --------------------------------------------------
# detector
# --------------------------------------------------

CAPTURE = {
    "device": _user_config["capture"],
    "left": int(RESOLUTION_X * 0.78125),
    "right": RESOLUTION_X,
    "top": RESOLUTION_Y * 0.70,
    "bottom": RESOLUTION_Y,
    "width": int(RESOLUTION_X * 0.21875),
    "height": int(RESOLUTION_Y * 0.3),
}

DETECTION = {
    "config_path": osp.join(osp.abspath(osp.dirname(__file__)), "..", _user_config["detection.config_path"],),
    "weights_path": osp.join(osp.abspath(osp.dirname(__file__)), "..", _user_config["detection.weights_path"],),
    "use_cuda": torch.cuda.is_available(),
    "yolo_resolution": 512,
}

# --------------------------------------------------
# battle field
# --------------------------------------------------

LIMIT_TIME = 120.0  # sec

BF_BOARD = {
    "left": (RESOLUTION_X - RESOLUTION_Y) // 2,
    "right": (RESOLUTION_X + RESOLUTION_Y) // 2,
    "top": 0,
    "bottom": RESOLUTION_Y,
    "width": RESOLUTION_Y,
    "height": RESOLUTION_Y,
    "center": RESOLUTION_Y // 2,
}

BF_SCORE = {
    "left": 0,
    "right": int(RESOLUTION_X * 0.21875),
    "top": 0,
    "bottom": int(RESOLUTION_Y * 0.3),
    "width": int(RESOLUTION_X * 0.21875),
    "height": int(RESOLUTION_Y * 0.3),
    "font_size": int(RESOLUTION_Y / 10.8),
}

BF_TIME = {
    "left": 0,
    "right": int(RESOLUTION_X * 0.21875),
    "top": int(RESOLUTION_Y * 0.3),
    "bottom": int(RESOLUTION_Y * 0.6),
    "width": int(RESOLUTION_X * 0.21875),
    "height": int(RESOLUTION_Y * 0.3),
    "font_size": int(RESOLUTION_Y / 10.8),
}

BF_SPEED = {
    "left": 0,
    "right": int(RESOLUTION_X * 0.21875),
    "top": int(RESOLUTION_Y * 0.6),
    "bottom": int(RESOLUTION_Y * 0.8),
    "width": int(RESOLUTION_X * 0.21875),
    "height": int(RESOLUTION_Y * 0.2),
    "font_size": int(RESOLUTION_Y / 21.6),
}

HAND_IMAGE_PATH = {
    "paper": osp.join(osp.abspath(osp.dirname(__file__)), "../images/paper.png"),
    "rock": osp.join(osp.abspath(osp.dirname(__file__)), "../images/rock.png"),
    "scissors": osp.join(osp.abspath(osp.dirname(__file__)), "../images/scissors.png"),
}

BF_HAND_1 = {
    "left": int(RESOLUTION_X * 0.78125),
    "right": RESOLUTION_X,
    "top": int(RESOLUTION_Y * 0.025),
    "bottom": int(RESOLUTION_Y * 0.325),
    "width": int(RESOLUTION_X * 0.21875),
    "height": int(RESOLUTION_Y * 0.3),
}

BF_HAND_2 = {
    "left": int(RESOLUTION_X * 0.78125),
    "right": RESOLUTION_X,
    "top": int(RESOLUTION_Y * 0.35),
    "bottom": int(RESOLUTION_Y * 0.65),
    "width": int(RESOLUTION_X * 0.21875),
    "height": int(RESOLUTION_Y * 0.3),
}

# --------------------------------------------------
# score screen
# --------------------------------------------------

SCORE_SCREEN = {
    "display_time": 10,
    "result_x": int(RESOLUTION_X * 0.5),
    "result_y": int(RESOLUTION_Y * 0.375),
    "score_x": int(RESOLUTION_X * 0.5),
    "score_y": int(RESOLUTION_Y * 0.625),
    "time_left": int(RESOLUTION_X * 0.3),
    "time_right": int(RESOLUTION_X * 0.7),
    "time_top": int(RESOLUTION_Y * 0.8),
    "time_bottom": int(RESOLUTION_Y * 0.82),
    "time_width": int(RESOLUTION_X * 0.4),
    "time_height": int(RESOLUTION_Y * 0.02),
}

# --------------------------------------------------
# error handling
# --------------------------------------------------

if RESOLUTION_X < 1280:
    raise ValueError(f"RESOLUTION_X must be 1280 or more, not {RESOLUTION_X}.")
if RESOLUTION_Y < 720:
    raise ValueError(f"RESOLUTION_Y must be 720 or more, not {RESOLUTION_Y}.")
if RESOLUTION_X * 9 != RESOLUTION_Y * 16:
    raise ValueError("The aspect ratio must be 16:9.")
