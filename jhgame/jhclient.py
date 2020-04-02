import sys

import pygame

from .battlefield import BattleField
from . import config
from .declient import DataExchangeClient
from .gamelog import get_module_logger
from .menu import MainMenu

LOGGER = get_module_logger(__name__)


class JankenHockeyClient:
    def __init__(self):
        self._setup_pygame()
        self._surface = pygame.display.set_mode((config.RESOLUTION_X, config.RESOLUTION_Y))
        self.clock = pygame.time.Clock()

        try:
            self.declient = DataExchangeClient(config.SERVER_HOST)
            self.my_id = self.declient.register()
        except ConnectionRefusedError as e:
            # FIXME Cannot kill task completely.
            LOGGER.critical(e)
            self._quit()

        self.bf = BattleField()
        self.hand_detector = {"coord": 0, "type": 0}
        self._main_menu = MainMenu(surface=self._surface, jhclient=self)

    @staticmethod
    def _setup_pygame():
        pygame.init()
        pygame.display.set_caption(config.GAME_INFO["title"])
        pygame.display.set_icon(pygame.image.load(config.GAME_INFO["icon_path"]))
        pygame.mouse.set_visible(False)

        # FIXME: Full screen mode is not working.
        if config.IS_FULLSCREEN:
            LOGGER.info("Display mode set to full screen.")
            pygame.display.set_mode(
                (config.RESOLUTION_X, config.RESOLUTION_Y), pygame.locals.FULLSCREEN,
            )
        else:
            LOGGER.info("Display mode set to window screen.")
            pygame.display.set_mode((config.RESOLUTION_X, config.RESOLUTION_Y))

    def run(self):
        """Run main menu."""
        self._main_menu.mainloop()
        pygame.display.flip()

        self._quit()

    @staticmethod
    def _quit():
        """Close client."""
        LOGGER.info("Client closed.")
        pygame.quit()
        sys.exit()
