"""Configure the following GUI.

┌------------------┬------------------┬---------------------┐
| BattleFieldScore |                  | BattleFieldHand (1) |
├------------------┤                  |---------------------┤
| BattleFieldTime  | BattleFieldBoard | BattleFieldHand (2) |
├------------------┤                  |---------------------┤
| BattleFieldSpeed |                  | BattleFieldCapture  |
└------------------┴------------------┴---------------------┘
"""

import math

import cv2
import numpy as np
import pygame
from pygame.surface import Surface
import pygameMenu.font

from . import config
from .detector import HandDetector


class BattleFieldBoard:
    def __init__(self, width: int, height: int, jhclient):
        """
        Args:
            width (int): Battle field board width
            height (int): Battle filed board height
            jhclient (JankenHockeyClient): Instance for JankenHockeyClient
        """
        self._width = width
        self._height = height
        self._jhclient = jhclient

        self._hand_colors = list(config.HAND_COLORS.values())

    def draw(self):
        """Draw battle field board.

        Returns:
            Surface: Drawn surface
        """
        surface = Surface((self._width, self._height))

        pygame.draw.rect(
            surface, config.COLORS["white"], pygame.locals.Rect(0, 0, self._width, self._height), 5,
        )

        pygame.draw.line(
            surface, config.COLORS["white"], (0, self._height // 2), (self._width, self._height // 2), 5,
        )

        # puck (disappear when returned in paper)
        if not self._jhclient.bf.puck.is_hidden:
            pygame.draw.circle(
                surface,
                self._hand_colors[self._jhclient.bf.puck.hand],
                (int(self._jhclient.bf.puck.x * self._width), int(self._jhclient.bf.puck.y * self._height),),
                int(self._jhclient.bf.puck.r * self._width),
            )

        # mallet (disappear when losing to janken in god mode)
        # front mallet
        if self._jhclient.bf.mallets[0].is_show:
            pygame.draw.line(
                surface,
                self._hand_colors[self._jhclient.bf.mallets[0].hand],
                (
                    int(self._jhclient.bf.mallets[0].left * self._width),
                    int(self._jhclient.bf.mallets[0].y * self._height),
                ),
                (
                    int(self._jhclient.bf.mallets[0].right * self._width),
                    int(self._jhclient.bf.mallets[0].y * self._height),
                ),
                10,
            )

        # back mallet
        if self._jhclient.bf.mallets[1].is_show:
            pygame.draw.line(
                surface,
                self._hand_colors[self._jhclient.bf.mallets[1].hand],
                (
                    int(self._jhclient.bf.mallets[1].left * self._width),
                    int(self._jhclient.bf.mallets[1].y * self._height),
                ),
                (
                    int(self._jhclient.bf.mallets[1].right * self._width),
                    int(self._jhclient.bf.mallets[1].y * self._height),
                ),
                10,
            )

        if self._jhclient.my_id == self._jhclient.bf.player_ids[0]:
            return surface

        return pygame.transform.rotate(surface, 180)


class BattleFieldScore:
    def __init__(self, width: int, height: int, jhclient):
        """
        Args:
            width (int): Battle field score width
            height (int): Battle field score height
            jhclient (JankenHockeyClient): Instance for JankenHockeyClient
        """
        self.font = pygame.font.Font(pygameMenu.font.FONT_NEVIS, config.BF_SCORE["font_size"])
        self._width = width
        self._height = height
        self._jhclient = jhclient

        self.text_width, self.text_height = 0, 0

    def draw(self):
        """Draw battle field score.

        Returns:
            Surface: Drawn surface
        """
        surface = Surface((self._width, self._height))

        _scores = self._jhclient.bf.scores
        if self._jhclient.my_id == self._jhclient.bf.player_ids[0]:
            text = f"{_scores[0]} - {_scores[1]}"
        else:
            text = f"{_scores[1]} - {_scores[0]}"

        text = self.font.render(text, True, config.COLORS["white"])
        self.text_width, self.text_height = text.get_rect()[2:]

        _left = (self._width - self.text_width) // 2
        _top = (self._height - self.text_height) // 2
        surface.blit(text, (_left, _top))

        return surface


class BattleFieldTime:
    def __init__(self, width: int, height: int, jhclient):
        """
        Args:
            width (int): Battle field time width
            height (int): Battle filed time height
            jhclient (JankenHockeyClient): Instance for JankenHockeyClient
        """
        self.font = pygame.font.Font(pygameMenu.font.FONT_NEVIS, config.BF_TIME["font_size"])
        self._width = width
        self._height = height
        self._jhclient = jhclient

        self.text_width, self.text_height = 0, 0

    def draw(self):
        """Draw battle field time.

        Returns:
            Surface: Drawn surface
        """
        surface = Surface((self._width, self._height))

        _minute, _second = divmod(math.ceil(self._jhclient.bf.time), 60)
        text = f"{_minute:02d}:{_second:02d}"

        # change text color by remaining time
        if _minute >= 1 and _second > 0:
            text = self.font.render(text, True, config.COLORS["white"])
        elif _second > 30:
            text = self.font.render(text, True, config.COLORS["blue"])
        elif _second > 10:
            text = self.font.render(text, True, config.COLORS["orange"])
        else:
            text = self.font.render(text, True, config.COLORS["red"])

        self.text_width, self.text_height = text.get_rect()[2:]

        _left = (self._width - self.text_width) // 2
        _top = (self._height - self.text_height) // 2
        surface.blit(text, (_left, _top))

        return surface


class BattleFieldSpeed:
    def __init__(self, width: int, height: int):
        """
        Args:
            width (int): Battle field speed width
            height (int): Battle filed speed height
        """
        self.font = pygame.font.Font(pygameMenu.font.FONT_NEVIS, config.BF_SPEED["font_size"])
        self._width = width
        self._height = height

        self.text_width, self.text_height = 0, 0

    def draw(self):
        """Draw battle field speed.

        Returns:
            Surface: Drawn surface
        """
        surface = Surface((self._width, self._height))

        text = self.font.render("SPEED UP!", True, config.COLORS["orange"])
        self.text_width, self.text_height = text.get_rect()[2:]

        _left = (self._width - self.text_width) // 2
        _top = (self._height - self.text_height) // 2
        surface.blit(text, (_left, _top))

        return surface


class BattleFieldHand:
    def __init__(self, width: int, height: int):
        """
        Args:
            width (int): Battle field hand width
            height (int): Battle filed hand height
        """
        paper_image = pygame.transform.scale(
            pygame.image.load(config.HAND_IMAGE_PATH["paper"]).convert(),
            (config.BF_HAND_1["width"], config.BF_HAND_1["height"]),
        )
        rock_image = pygame.transform.scale(
            pygame.image.load(config.HAND_IMAGE_PATH["rock"]).convert(),
            (config.BF_HAND_1["width"], config.BF_HAND_1["height"]),
        )
        scissors_image = pygame.transform.scale(
            pygame.image.load(config.HAND_IMAGE_PATH["scissors"]).convert(),
            (config.BF_HAND_1["width"], config.BF_HAND_1["height"]),
        )

        self._hand_images = [rock_image, scissors_image, paper_image]
        self._surface = Surface((width, height))

    def draw(self, hand_type: int):
        """Draw battle field hand.

        Args:
            hand_type (int): Index of hand sign.
                0: rock, 1: scissors, 2: paper

        Returns:
            Surface: Drawn surface
        """
        if hand_type != 3:
            self._surface.blit(self._hand_images[hand_type], (0, 0))

        return self._surface


class BattleFieldCapture:
    def __init__(self, width: int, height: int, jhclient):
        """
        Args:
            width (int): Battle field capture width
            height (int): Battle field capture height
            jhclient (JankenHockeyClient): Instance for JankenHockeyClient
        """
        self._width = width
        self._height = height
        self._jhclient = jhclient

        self._capture = cv2.VideoCapture(config.CAPTURE["device"])
        assert self._capture.isOpened(), f"Cannot detect {config.CAPTURE['device']}."

        self._rect_colors = list(config.HAND_COLORS.values())
        self._hand_detector = HandDetector()

        self.hand_type = 0  # initialize as rock
        self._x_min = 0
        self._x_max = 0

        self._surface = Surface((self._width, self._height))

    def draw(self):
        """Draw battle field capture.

        Returns:
            Surface: Drawn surface
        """
        frame_bgr = self._pick_frame()
        hand_type, x_min, y_min, x_max, y_max = self._hand_detector.detect(frame_bgr)  # not normalized

        if hand_type != -1:
            # if not detected, make previous detection result
            self.hand_type = hand_type
            # if not detected, hide rectangle
            frame_bgr = self._draw_rectangle(frame_bgr, self.hand_type, x_min, y_min, x_max, y_max)

            self._x_min = x_min
            self._x_max = x_max

        self._jhclient.hand_detector = {
            "coord": (self._x_min + self._x_max) / (2 * frame_bgr.shape[1]),
            "type": hand_type,
        }

        self._jhclient.declient.send(self._jhclient.hand_detector)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb, k=1)  # for rotate image
        frame_rgb = frame_rgb[::-1, :, :]  # for horizontal flipping

        self._surface.blit(
            pygame.transform.scale(
                pygame.pixelcopy.make_surface(frame_rgb), (config.CAPTURE["width"], config.CAPTURE["height"]),
            ),
            (0, 0),
        )

        return self._surface

    def _pick_frame(self):
        """Pick current frame as np.ndarray (BGR).

        Returns:
            np.ndarray: BGR image
        """
        ret, frame = self._capture.read()
        assert ret, "Cannot pick frame."

        return frame

    def _draw_rectangle(self, frame, hand_type, x_min, y_min, x_max, y_max):
        """Draw rectangle to frame.

        Args:
            frame (np.ndarray): BGR image
            hand_type (int): 0, 1, or 2
            x_min (int): Upper left x-coordinate of the rectangle
            y_min (int): Upper left y-coordinate of the rectangle
            x_min (int): Lower right x-coordinate of the rectangle
            y_max (int): Lower right y-coordinate of the rectangle

            - Coordinates are not normalized.

        Returns:
            np.ndarray: BGR image
        """
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max >= frame.shape[1]:
            x_max = frame.shape[1] - 1
        if y_max >= frame.shape[0]:
            y_max = frame.shape[0] - 1

        left_top = (x_min, y_min)
        right_bottom = (x_max, y_max)

        color = self._rect_colors[hand_type]
        color = color[2], color[1], color[0]

        frame = cv2.rectangle(frame, left_top, right_bottom, color, 10)

        return frame
