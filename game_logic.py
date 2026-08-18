"""
game_logic.py
-----------------------------------------
Headless (no graphics) Flappy Bird simulation.

This module contains the CORE game rules only. It is used by ai.py to
run thousands of fast simulations while training the Genetic Algorithm.
The actual visual game the user sees is drawn in JavaScript
(static/script.js), but it follows the SAME rules defined here so that
an AI trained on this simulation behaves the same way in the browser.

Keeping the rules in one place (this file) keeps the project simple:
Python = "brain" (learns to play), JavaScript = "eyes" (shows it playing).
"""

import random

# ---------------------------------------------------------------
# Game constants (kept small & simple, tuned to feel like Flappy Bird)
# ---------------------------------------------------------------
GAME_WIDTH = 480
GAME_HEIGHT = 640

BIRD_X = 80            # Bird stays at a fixed horizontal position
BIRD_RADIUS = 14

GRAVITY = 0.5           # Downward acceleration applied every frame
FLAP_STRENGTH = -8.0    # Upward velocity applied when the bird flaps

PIPE_WIDTH = 60
PIPE_GAP = 160          # Vertical opening the bird must fly through
PIPE_SPEED = 3.5
PIPE_SPACING = 220      # Horizontal distance between two pipes


class Bird:
    """A simple bird with a vertical position and velocity."""

    def __init__(self):
        self.y = GAME_HEIGHT / 2
        self.velocity = 0.0
        self.alive = True

    def flap(self):
        """Give the bird an upward boost."""
        self.velocity = FLAP_STRENGTH

    def update(self):
        """Apply gravity each frame."""
        self.velocity += GRAVITY
        self.y += self.velocity


class Pipe:
    """A pair of top/bottom pipes with a gap the bird must pass through."""

    def __init__(self, x):
        self.x = x
        # Random gap center, leaving room so the gap never touches the edges
        margin = 80
        self.gap_y = random.randint(margin, GAME_HEIGHT - margin)
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED

    @property
    def top(self):
        return self.gap_y - PIPE_GAP / 2

    @property
    def bottom(self):
        return self.gap_y + PIPE_GAP / 2


class Game:
    """
    A full headless game session.

    Usage:
        game = Game()
        while game.alive:
            action = 1 or 0      # 1 = flap, 0 = do nothing
            game.step(action)
        final_score = game.score
    """

    def __init__(self):
        self.bird = Bird()
        self.pipes = [Pipe(GAME_WIDTH + 100)]
        self.score = 0
        self.alive = True
        self.frames = 0

    def get_state(self):
        """
        Return the 4 simple inputs the AI uses to make a decision.
        All values are normalized to roughly the 0-1 (or -1 to 1) range,
        which helps the tiny neural-net-like perceptron learn faster.
        """
        next_pipe = self._next_pipe()

        bird_y_norm = self.bird.y / GAME_HEIGHT
        velocity_norm = max(-1.0, min(1.0, self.bird.velocity / 10.0))
        dist_to_pipe_norm = max(0.0, (next_pipe.x - BIRD_X) / GAME_WIDTH)
        gap_y_norm = next_pipe.gap_y / GAME_HEIGHT

        return [bird_y_norm, velocity_norm, dist_to_pipe_norm, gap_y_norm]

    def _next_pipe(self):
        """Return the closest pipe that the bird has not passed yet."""
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > BIRD_X:
                return pipe
        return self.pipes[-1]

    def step(self, action):
        """
        Advance the game by a single frame.
        action: 1 -> flap, 0 -> do nothing
        Returns: (alive: bool, score: int)
        """
        if not self.alive:
            return self.alive, self.score

        if action == 1:
            self.bird.flap()

        self.bird.update()
        self.frames += 1

        # Move pipes, add new ones, remove old ones
        for pipe in self.pipes:
            pipe.update()

        if self.pipes[-1].x < GAME_WIDTH - PIPE_SPACING:
            self.pipes.append(Pipe(GAME_WIDTH + 20))

        self.pipes = [p for p in self.pipes if p.x > -PIPE_WIDTH]

        # Score a point whenever the bird passes a pipe's center
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.passed = True
                self.score += 1

        # Collision checks: ground, ceiling, pipes
        if self.bird.y > GAME_HEIGHT - BIRD_RADIUS or self.bird.y < BIRD_RADIUS:
            self.alive = False

        next_pipe = self._next_pipe()
        if (next_pipe.x < BIRD_X + BIRD_RADIUS and
                next_pipe.x + PIPE_WIDTH > BIRD_X - BIRD_RADIUS):
            if self.bird.y - BIRD_RADIUS < next_pipe.top or \
               self.bird.y + BIRD_RADIUS > next_pipe.bottom:
                self.alive = False

        return self.alive, self.score
