import pygame
import numpy as np
import matplotlib.pyplot as plt
pygame.init()

class Canvas:
    def __init__(self, width, height, grid: np.ndarray):
        self.width = width
        self.height = height
        self.grid = grid
        self.screen = pygame.Surface((width, height))
        self.brush_size = 1.0
        self.updated = False

    def save_plot(self, filename="digit.png", show=False):
        """
        Save the current grid as a matplotlib plot.
        Args:
            filename: Name of the file to save (default: 'digit.png')
            show: Whether to display the plot (default: False)
        """
        # Transpose the array to match the display orientation
        plt.imsave(filename, self.grid.T, cmap='gray', vmin=0, vmax=1, dpi=300)

    def gamma_correct(self, value):
        return np.power(value, 1.0 / 2.2) * 255

    def get_relative_mouse_pos(self, mouse_pos, window):
        mouse_x, mouse_y = mouse_pos
        canvas_x = window.get_width() - self.width
        canvas_y = window.get_height() - self.height
        relative_x = mouse_x - canvas_x
        relative_y = mouse_y - canvas_y
        return relative_x, relative_y

    def get_grid_coordinates(self, rel_x, rel_y):
        cols, rows = self.grid.shape
        cell_width = int(self.width / cols)
        cell_height = int(self.height / rows)
        grid_x = int(rel_x / cell_width)
        grid_y = int(rel_y / cell_height)
        # Ensure coordinates are within grid bounds
        grid_x = max(0, min(grid_x, cols - 1))
        grid_y = max(0, min(grid_y, rows - 1))
        return grid_x, grid_y

    def iterate_indices(self):
        it = np.nditer(self.grid, flags=['multi_index'])
        while not it.finished:
            x, y = it.multi_index
            yield x, y
            it.iternext()

    def draw(self, window):
        cols, rows = self.grid.shape
        width = int(self.width / cols)
        height = int(self.height / rows)

        for x, y in self.iterate_indices():
            I = self.gamma_correct(self.grid[x, y])
            x_pos = int(x * width)
            y_pos = int(y * height)
            pygame.draw.rect(self.screen, (I, I, I), (x_pos, y_pos, width, height), width=0)

        canvas_x = window.get_width() - self.width
        canvas_y = window.get_height() - self.height
        window.blit(self.screen, (canvas_x, canvas_y))

    def add(self, mouse_pos, window):
        rel_x, rel_y = self.get_relative_mouse_pos(mouse_pos, window)
        grid_x, grid_y = self.get_grid_coordinates(rel_x, rel_y)
        
        # Get the exact position in grid coordinates (not rounded)
        cols, rows = self.grid.shape
        cell_width = self.width / cols
        cell_height = self.height / rows
        exact_x = rel_x / cell_width
        exact_y = rel_y / cell_height
        
        # Calculate distance from grid point
        dx = exact_x - grid_x
        dy = exact_y - grid_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Adjust amount based on distance (closer = more)
        amount = 1.5 * (1 - min(distance, 1))

        # Apply to all cells within brush size
        radius = int(np.ceil(self.brush_size))
        for x in range(max(0, grid_x - radius), min(cols, grid_x + radius + 1)):
            for y in range(max(0, grid_y - radius), min(rows, grid_y + radius + 1)):
                if self.brush_size <= 1.0:
                    # For size <= 1, only affect the exact cell
                    if x == grid_x and y == grid_y:
                        self.grid[x, y] = min(self.grid[x, y] + amount, 1)
                        self.updated = True
                else:
                    # For larger sizes, use circular brush
                    brush_dx = x - exact_x
                    brush_dy = y - exact_y
                    brush_distance = np.sqrt(brush_dx*brush_dx + brush_dy*brush_dy)
                    if brush_distance <= self.brush_size:
                        brush_amount = amount * (1 - brush_distance/self.brush_size)
                        self.grid[x, y] = min(self.grid[x, y] + brush_amount, 1)
                        self.updated = True

    def erase(self, mouse_pos, window):
        rel_x, rel_y = self.get_relative_mouse_pos(mouse_pos, window)
        grid_x, grid_y = self.get_grid_coordinates(rel_x, rel_y)
        cols, rows = self.grid.shape
        
        # Erase all cells within brush size
        radius = int(np.ceil(self.brush_size))
        for x in range(max(0, grid_x - radius), min(cols, grid_x + radius + 1)):
            for y in range(max(0, grid_y - radius), min(rows, grid_y + radius + 1)):
                if self.brush_size <= 1.0:
                    # For size <= 1, only affect the exact cell
                    if x == grid_x and y == grid_y:
                        self.grid[x, y] = 0
                        self.updated = True
                else:
                    # For larger sizes, use circular brush
                    brush_dx = x - grid_x
                    brush_dy = y - grid_y
                    brush_distance = np.sqrt(brush_dx*brush_dx + brush_dy*brush_dy)
                    if brush_distance <= self.brush_size:
                        self.grid[x, y] = 0
                        self.updated = True

    def clear(self):
        self.grid = np.zeros_like(self.grid)
        self.updated = True

    def adjust_brush_size(self, amount):
        self.brush_size = max(0.5, min(1.3, self.brush_size + amount*0.5))



