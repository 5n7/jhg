import time

import pygame
from pygame.surface import Surface
import pygameMenu.font

from . import config


class ScoreScreen:
    def __init__(self, width: int, height: int, jhclient):
        """Game end screen.

        Args:
            width (int): Score screen width
            height (int): Score screen height
            jhclient (JankenHockeyClient): Instance for JankenHockeyClient
        """
        self.width = width
        self.height = height
        self.jhclient = jhclient

        self.is_leave_room = True

        self.font = pygame.font.Font(pygameMenu.font.FONT_NEVIS, config.BF_SCORE["font_size"])

    def draw(self, elapsed_time):
        """Draw score screen.

        Args:
            elapsed_time (float): Elapsed time

        Returns:
            Surface: Drawn surface
        """
        if self.is_leave_room:
            self.jhclient.declient.leave_room()
            self.is_leave_room = False

        remaining_time = config.SCORE_SCREEN["display_time"] - elapsed_time
        p1_score, p2_score = self.jhclient.bf.scores

        surface = Surface((self.width, self.height))

        if (self.jhclient.my_id == self.jhclient.bf.player_ids[0] and p1_score > p2_score) or (
            self.jhclient.my_id == self.jhclient.bf.player_ids[1] and p1_score < p2_score
        ):
            surface = self._win(surface)

        elif (self.jhclient.my_id == self.jhclient.bf.player_ids[0] and p1_score < p2_score) or (
            self.jhclient.my_id == self.jhclient.bf.player_ids[1] and p1_score > p2_score
        ):
            surface = self._lose(surface)

        else:
            surface = self._draw(surface)

        pygame.draw.rect(
            surface,
            config.COLORS["white"],
            pygame.locals.Rect(
                config.SCORE_SCREEN["time_left"],
                config.SCORE_SCREEN["time_top"],
                int(config.SCORE_SCREEN["time_width"] * remaining_time / config.SCORE_SCREEN["display_time"]),
                config.SCORE_SCREEN["time_height"],
            ),
        )

        return surface

    def _win(self, surface):
        win_text = self.font.render("YOU WIN!", True, config.COLORS["red"])
        score_text = self.font.render(
            f"{max(self.jhclient.bf.scores)} - " + f"{min(self.jhclient.bf.scores)}", True, config.COLORS["white"],
        )

        win_rect = win_text.get_rect(center=(config.SCORE_SCREEN["result_x"], config.SCORE_SCREEN["result_y"],))
        score_rect = score_text.get_rect(center=(config.SCORE_SCREEN["score_x"], config.SCORE_SCREEN["score_y"],))

        surface.blit(win_text, win_rect)
        surface.blit(score_text, score_rect)
        return surface

    def _lose(self, surface):
        lose_text = self.font.render("YOU LOSE!", True, config.COLORS["blue"])
        score_text = self.font.render(
            f"{min(self.jhclient.bf.scores)} - " + f"{max(self.jhclient.bf.scores)}", True, config.COLORS["white"],
        )

        lose_rect = lose_text.get_rect(center=(config.SCORE_SCREEN["result_x"], config.SCORE_SCREEN["result_y"],))
        score_rect = score_text.get_rect(center=(config.SCORE_SCREEN["score_x"], config.SCORE_SCREEN["score_y"],))

        surface.blit(lose_text, lose_rect)
        surface.blit(score_text, score_rect)
        return surface

    def _draw(self, surface):
        draw_text = self.font.render("DRAW!", True, config.COLORS["yellow"])
        score_text = self.font.render(
            f"{self.jhclient.bf.scores[0]} - " + f"{self.jhclient.bf.scores[1]}", True, config.COLORS["white"],
        )

        lose_rect = draw_text.get_rect(center=(config.SCORE_SCREEN["result_x"], config.SCORE_SCREEN["result_y"],))
        score_rect = score_text.get_rect(center=(config.SCORE_SCREEN["score_x"], config.SCORE_SCREEN["score_y"],))

        surface.blit(draw_text, lose_rect)
        surface.blit(score_text, score_rect)
        return surface


class ScoreClock:
    def __init__(self):
        self.start_time = None

    def start_clock(self):
        self.start_time = time.time()

    def calc_elapsed_time(self):
        return time.time() - self.start_time
