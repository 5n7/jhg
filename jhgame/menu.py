import itertools

import pygame
import pygameMenu

from . import config
from .game import Game
from .gamelog import get_module_logger

LOGGER = get_module_logger(__name__)
SURFACE = pygame.display.set_mode((config.RESOLUTION_X, config.RESOLUTION_Y))


def main_background():
    """Surface for bgfun argument."""
    SURFACE.fill(config.COLORS["blue_gray"])


class MenuTemplate(pygameMenu.Menu):
    def __init__(self, surface, title):
        """Template for menu.

        Args:
            surface (Surface): Instance for Surface
            title (str): Menu title
        """
        super().__init__(
            surface=surface,
            window_width=config.RESOLUTION_X,
            window_height=config.RESOLUTION_Y,
            font=pygameMenu.font.FONT_NEVIS,
            title=title,
            bgfun=main_background,
            color_selected=config.COLORS["white"],
            font_color=config.COLORS["black"],
            font_size=config.MENU["font_size"],
            fps=config.FPS,
            menu_color=config.COLORS["green"],
            menu_color_title=config.COLORS["white"],
            menu_width=int(config.RESOLUTION_X * 0.7),
            menu_height=int(config.RESOLUTION_Y * 0.7),
            onclose=pygameMenu.events.DISABLE_CLOSE,
        )


class TextMenuTemplate(pygameMenu.TextMenu):
    def __init__(self, surface, title):
        """Template for text menu.

        Args:
            surface (Surface): Instance for Surface
            title (str): Menu title
        """
        super().__init__(
            surface=surface,
            window_width=config.RESOLUTION_X,
            window_height=config.RESOLUTION_Y,
            font=pygameMenu.font.FONT_NEVIS,
            title=title,
            bgfun=main_background,
            color_selected=config.COLORS["white"],
            text_color=config.COLORS["black"],
            text_fontsize=config.MENU["font_size"],
            font_color=config.COLORS["black"],
            menu_color=config.COLORS["green"],
            menu_color_title=config.COLORS["white"],
            menu_width=int(config.RESOLUTION_X * 0.7),
            menu_height=int(config.RESOLUTION_Y * 0.7),
        )


class MainMenu(MenuTemplate):
    def __init__(self, jhclient, surface, title="MAIN MENU"):
        super().__init__(surface=surface, title=title)

        play_menu = PlayMenu(surface=surface, jhclient=jhclient)
        help_menu = HelpMenu(surface=surface)
        about_menu = AboutMenu(surface=surface)

        self.add_option("PLAY", play_menu)
        self.add_option("HELP", help_menu)
        self.add_option("ABOUT", about_menu)
        self.add_option("QUIT", pygameMenu.events.EXIT)


class PlayMenu(MenuTemplate):
    def __init__(self, jhclient, surface, title="PLAY MENU"):
        super().__init__(surface=surface, title=title)
        self._jhclient = jhclient

        self.initial_rooms = [("NOT RELOADED", "")]
        self.rooms = self.initial_rooms
        self.levels = [
            ("EASY", "easy"),
            ("NORMAL", "normal"),
            ("HARD", "hard"),
        ]

        self.create_room = CreateRoomMenu(surface=surface, title="CREATE ROOM", jhclient=jhclient, play_menu=self,)
        self.select_room_menu = SelectRoomMenu(surface=surface, title="SELECT ROOM", jhclient=jhclient, play_menu=self,)

        # 1. create room button
        self.add_option("CREATE ROOM", self.create_room)

        # 2. start game button
        self.add_option("SELECT ROOM", self.select_room_menu)

        # 3. return main menu button
        self.add_option("RETURN TO MAIN MENU", pygameMenu.events.BACK)

        self.game = Game(config.RESOLUTION_X, config.RESOLUTION_Y, surface, self._jhclient)

    def finish_game(self):
        """Process at the end of the game."""
        LOGGER.info("Game finished.")
        self._jhclient.bf.__init__()
        self._jhclient.declient.get_messages()
        self.game.score_clock.__init__()


class CreateRoomMenu(MenuTemplate):
    def __init__(self, surface, title, jhclient, play_menu):
        super().__init__(surface=surface, title=title)

        self._jhclient = jhclient
        self.play_menu = play_menu

        # 1. room name input form
        self.add_text_input("INPUT ROOM NAME > ", maxchar=16, textinput_id="name_input")

        # 2. level selector
        self._levels = [
            ("EASY", "easy"),
            ("NORMAL", "normal"),
            ("HARD", "hard"),
        ]
        self.add_selector(
            "SELECT GAME LEVEL", self._levels, selector_id="level_selector", default=1,
        )

        # 3. game start button
        self.add_option("START GAME", self.create_room)

        # 4. back menu button
        self.add_option("RETURN TO MAIN MENU", pygameMenu.events.BACK)

        # 5. secret selector
        self.add_selector(
            "",
            [("", "") for _ in range(5)] + [(("GOD MODE", "god"))] + [("", "") for _ in range(5)],
            selector_id="secret_selector",
            default=0,
        )
        self.get_widget("secret_selector")._sformat = "{0} {1}"

    def create_room(self):
        """Create room and run game."""
        # get input data
        game_settings = self.get_input_data()
        room_name = game_settings["name_input"]
        level = self._levels[game_settings["level_selector"][1]][1]
        is_god_mode = game_settings["secret_selector"][0] == "GOD MODE"

        level = "god" if is_god_mode else level

        # check input
        if self._check_room_name(room_name):
            LOGGER.info("Room created `%s` (%s).", room_name, level)
            self._jhclient.declient.create_room(room_name=room_name, level=level)

            LOGGER.info("Game started `%s` (%s).", room_name, level)
            self.play_menu.game.run()

            self.play_menu.finish_game()

    @staticmethod
    def _check_room_name(room_name):
        """Check if the room name is the correct format.

        Args:
            room_name (str): Room name

        Returns:
            bool: Result
        """
        if not room_name or room_name is None:
            return False
        return True


class SelectRoomMenu(MenuTemplate):
    def __init__(self, surface, title, jhclient, play_menu):
        super().__init__(surface=surface, title=title)

        self._jhclient = jhclient
        self.play_menu = play_menu

        # 1. room selector
        self.add_selector(
            title="SELECT ROOM", values=self.play_menu.rooms, selector_id="room_selector", onreturn=None, default=0,
        )

        # 2. reload room button
        self.add_option("RELOAD ROOM INFO", self.reload_room)

        # 3. game start button
        self.add_option("START GAME", self.run_game)

        # 4. return main menu button
        self.add_option("RETURN TO MAIN MENU", pygameMenu.events.BACK)

    def reload_room(self):
        """Fetch server room infomation."""
        rooms = self._jhclient.declient.get_rooms()

        LOGGER.info("Room info reloaded.")

        if not rooms:
            self.play_menu.rooms = self.play_menu.initial_rooms

        else:
            room_ids = [room["id"] for room in rooms]
            room_names = [
                f"[{room['level']}] {room['name']} - " + f"{room['num_players']}/{room['capacity']}" for room in rooms
            ]

            self.play_menu.rooms = [(room_name, room_id) for room_name, room_id in zip(room_names, room_ids)]

        self.get_widget("room_selector").update_elements(self.play_menu.rooms)

    def run_game(self):
        """Start the game in the selected room. If failed, reload room list."""
        selected_room_index = self.get_input_data()["room_selector"][1]
        selected_room_id = self.play_menu.rooms[selected_room_index][1]

        if self._jhclient.declient.join_room(selected_room_id):
            LOGGER.info("Game started `%s`.", selected_room_id)
            self.play_menu.game.run()

            self.play_menu.finish_game()
        else:
            LOGGER.info("Could not start game.")
            self.reload_room()


class HelpMenu(TextMenuTemplate):
    def __init__(self, surface, title="HELP"):
        super().__init__(surface=surface, title=title)

        self.add_line(config.HELP["title"])
        self.add_line(pygameMenu.locals.TEXT_NEWLINE)

        self.add_line(config.HELP["overview"])
        self.add_line(pygameMenu.locals.TEXT_NEWLINE)

        for usage in config.HELP["usage"]:
            self.add_line(usage)
        self.add_line(pygameMenu.locals.TEXT_NEWLINE)

        self.add_option("RETURN TO MENU", pygameMenu.events.BACK)


class AboutMenu(TextMenuTemplate):
    def __init__(self, surface, title="ABOUT"):
        super().__init__(surface=surface, title=title)

        self.add_line(config.ABOUT["title"])
        self.add_line(pygameMenu.locals.TEXT_NEWLINE)

        for organization in config.ABOUT["organization"]:
            self.add_line(organization)
        self.add_line(pygameMenu.locals.TEXT_NEWLINE)

        developers = config.ABOUT["developers"]
        for developer in itertools.zip_longest(*[iter(developers)] * 2):
            # indicated side by side two people
            self.add_line(f"{developer[0]} / {developer[1]}")
        self.add_line(pygameMenu.locals.TEXT_NEWLINE)

        self.add_option("RETURN TO MENU", pygameMenu.events.BACK)
