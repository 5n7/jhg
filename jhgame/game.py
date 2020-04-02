import pygame
from pygame.surface import Surface

from . import config
from .battlefieldui import (
    BattleFieldBoard,
    BattleFieldCapture,
    BattleFieldHand,
    BattleFieldScore,
    BattleFieldSpeed,
    BattleFieldTime,
)
from .scorescreen import ScoreClock, ScoreScreen


class Game:
    def __init__(self, width: int, height: int, surface, jhclient):
        """Control game GUI.

        Args:
            width (int): Screen width, expected 1920
            height (int): Screen height, expected 1080
            surface (Surface): Instance for Surface
            jhclient (JankenHockeyClient): Instance for JankenHockeyClient
        """
        self._width = width
        self._height = height
        self._surface = surface
        self._jhclient = jhclient

        self._clock = pygame.time.Clock()

        self._score_screen = ScoreScreen(width=width, height=height, jhclient=jhclient)
        self.score_clock = ScoreClock()

        self._init_surface()

    def _init_surface(self):
        """Initialize surface."""
        self._background = Surface((self._width, self._height))
        self._background.fill(config.COLORS["black"])

        self._bf_score = BattleFieldScore(
            width=config.BF_SCORE["width"], height=config.BF_SCORE["height"], jhclient=self._jhclient,
        )

        self._bf_time = BattleFieldTime(
            width=config.BF_TIME["width"], height=config.BF_TIME["height"], jhclient=self._jhclient,
        )

        self._bf_board = BattleFieldBoard(
            width=config.BF_BOARD["width"], height=config.BF_BOARD["height"], jhclient=self._jhclient,
        )

        self._bf_speed = BattleFieldSpeed(width=config.BF_SPEED["width"], height=config.BF_SPEED["height"])

        self._hand_1 = BattleFieldHand(width=config.BF_HAND_1["width"], height=config.BF_HAND_1["height"])

        self._hand_2 = BattleFieldHand(width=config.BF_HAND_2["width"], height=config.BF_HAND_2["height"])

        self._capture = BattleFieldCapture(
            width=config.CAPTURE["width"], height=config.CAPTURE["height"], jhclient=self._jhclient,
        )

    def run(self):
        """Start game."""
        is_game_running = True  # including score indication
        while is_game_running:
            self._clock.tick(config.FPS)

            if self._jhclient.bf.game_flag:
                # show battlefield

                # get battlefield information from server and set them
                for message in [list(rec_dict.values())[0] for rec_dict in self._jhclient.declient.get_messages()]:
                    self._jhclient.bf.set_items(message)

                # GUI: background
                self._surface.blit(self._background, (0, 0))

                # GUI: score board
                self._surface.blit(
                    self._bf_score.draw(), (config.BF_SCORE["left"], config.BF_SCORE["top"]),
                )

                # GUI: time board
                self._surface.blit(
                    self._bf_time.draw(),
                    # for centering
                    (config.BF_TIME["left"], config.BF_TIME["top"]),
                )

                # GUI: battle speed
                if self._jhclient.bf.speedup_mode:
                    self._surface.blit(
                        self._bf_speed.draw(), (config.BF_SPEED["left"], config.BF_SPEED["top"]),
                    )

                # GUI: battle field
                self._surface.blit(
                    self._bf_board.draw(), (config.BF_BOARD["left"], config.BF_BOARD["top"]),
                )

                # GUI: hand image display
                # hand (1)
                self._surface.blit(
                    self._hand_1.draw(self._jhclient.bf.puck.hand), (config.BF_HAND_1["left"], config.BF_HAND_1["top"]),
                )

                # hand (2)
                if self._capture.hand_type != -1:
                    self._surface.blit(
                        self._hand_2.draw(self._capture.hand_type), (config.BF_HAND_2["left"], config.BF_HAND_2["top"]),
                    )

                # GUI: capture
                self._surface.blit(
                    self._capture.draw(), (config.CAPTURE["left"], config.CAPTURE["top"]),
                )

            else:
                # show score screen
                if self.score_clock.start_time is None:
                    self.score_clock.start_clock()

                elapsed_time = self.score_clock.calc_elapsed_time()
                if elapsed_time > config.SCORE_SCREEN["display_time"]:
                    is_game_running = False

                self._surface.blit(self._score_screen.draw(elapsed_time=elapsed_time), (0, 0))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.locals.KEYDOWN:
                    if event.key == pygame.locals.K_ESCAPE:
                        is_game_running = False
                    if event.key == pygame.locals.K_F11:
                        pygame.display.toggle_fullscreen()
