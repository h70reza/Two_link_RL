import pygame
import numpy as np
import random
import matplotlib.pyplot as plt

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600  # Screen size
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLOCK_SIZE = 5  # Scaling for visualization
SPEED = 40  # Game speed

# font for diplaying text
FONT = pygame.font.Font(None, 24)
# Robot Parameters
λ = 50  # Link length
m, J = 1, 1  # System parameters (Modify as needed)

def compute_next_state(state, dα, dt=0.1):
    """ Compute next state based on given equations. """
    x, y, θ, α, p = state
    
    Iα = 4 * J + m * λ**2 + (m * λ**2 - 4 * J) * np.cos(α)
    
    # Compute state derivatives
    dx = (λ / 4) * np.cos(θ) * np.tan(α / 2) * ((4 * p * (1 + np.cos(α)) / Iα) + dα)
    dy = (λ / 4) * np.sin(θ) * np.tan(α / 2) * ((4 * p * (1 + np.cos(α)) / Iα) + dα)
    dθ = (2 * p * (1 - np.cos(α)) / Iα) - (dα / 2)
    dp = - (m * λ**2 * p * dα) / ((m * λ**2 + 4 * J + (m * λ**2 - 4 * J) * np.cos(α)) * np.tan(α / 2))
    
    # Update state
    x_new = x + dx * dt
    y_new = y + dy * dt
    θ_new = θ + dθ * dt
    α_new = α + dα * dt
    p_new = p + dp * dt
    
    return x_new, y_new, θ_new, α_new, p_new

class SnakeRobotGame:
    def __init__(self):
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Two-Link Snake Robot")
        self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self):
        """ Initialize the game state. """
        self.state = (WIDTH / 2, HEIGHT / 2, np.pi / 4, np.pi / 4, 0.02)  # (x, y, θ, α, p)
        self.target = (random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100))
        self.dα = random.randint(-1,1)  # Initial input

    def play_step(self):
        """ Perform one step in the game. """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.dα += 0.1  # Increase angular velocity
                elif event.key == pygame.K_DOWN:
                    self.dα -= 0.1  # Decrease angular velocity

        # Update state
        self.state = compute_next_state(self.state, self.dα)
        x, y, θ, α, p = self.state
        
        # Check if reached target
        if np.linalg.norm(np.array([x, y]) - np.array(self.target)) < 10:
            self.target = (random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100))

        # Update UI
        self._update_ui()
        self.clock.tick(SPEED)

    def _update_ui(self):
        """ Draws the game state. """
        self.display.fill(BLACK)
        
        # Draw target
        pygame.draw.circle(self.display, RED, self.target, 5)
        
        # Draw robot
        x, y, θ, α, p = self.state
        x0 = x - λ/2 * np.cos(θ)
        y0 = y - λ/2 * np.sin(θ)
        x1 = x + λ/2 * np.cos(θ)
        y1 = y + λ/2 * np.sin(θ)
        x2 = x1 + λ * np.cos(θ + α)
        y2 = y1 + λ * np.sin(θ + α)
        
        pygame.draw.line(self.display, BLUE, (x0, y0), (x1, y1), 5)
        pygame.draw.line(self.display, BLUE, (x1, y1), (x2, y2), 5)

        # Display angular velocity
        text_surface = FONT.render(f"dα: {self.dα: .2f}", True, WHITE)
        self.display.blit(text_surface, (10,10))
        
        pygame.display.flip()

if __name__ == "__main__":
    game = SnakeRobotGame()
    while True:
        game.play_step()