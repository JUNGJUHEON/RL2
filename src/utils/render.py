import numpy as np


class ParallelScreen:
    """An auxiliary class for envs with displayable screen-like states.
    Supports parallel rendering of multiple active envs."""

    def __init__(self, num_screens, screen_shape):
        """
        :param num_screens: Usually the number of parallel environments
        :param screen_shape: (x, y) for grayscale and (x, y, 3) for RGB
        """
        self.num_screens = num_screens
        self.screen_shape = screen_shape
        self.screens = np.zeros(shape=(self.num_screens, *self.screen_shape), dtype="uint8")

    def update_screens(self, screens, scr_ids=None):
        """Updates the screens of all envs. Values are expected to be int
        in range 0 to 256."""
        if scr_ids is None:
            self.screens[:] = screens
        else:
            self.screens[scr_ids] = screens

    def render(self):
        pass  # headless: no display output
