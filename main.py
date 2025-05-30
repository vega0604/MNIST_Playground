from keras.models import load_model
import numpy as np
import pygame
from canvas import Canvas
import threading
import queue
import colorsys
pygame.init()

class Window:
    def __init__(self):
        self.model = load_model('mnist_model.keras')
        self.screen = pygame.display.set_mode((850, 560))
        pygame.display.set_caption('MNIST Playground')
        self.clock = pygame.time.Clock()
        self.running = True
        self.canvas = Canvas(560, 560, np.zeros((28, 28)))
        self.mouse_down = False
        self.mouse_button = None
        self.bold_font = pygame.font.Font('fonts/SpaceGrotesk-Bold.ttf', 18)  # Load local TTF font
        self.medium_font = pygame.font.Font('fonts/SpaceGrotesk-Medium.ttf', 24)  # Load local TTF font
        self.small_font = pygame.font.Font('fonts/SpaceGrotesk-Medium.ttf', 14)  # Load local TTF font
        self.prediction_queue = queue.Queue()
        self.prediction_thread = None
        # self.labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.prediction = np.zeros((1, len(self.labels)))
        self.continuous_prediction = True
        self.title = self.medium_font.render('MNIST Playground', True, (245, 245, 245))


    def predict_thread(self, grid):
        try:
            input_data = grid.T.reshape(1, 28, 28, 1)
            if np.all(input_data == 0):  # Check if input is all zeros
                self.prediction_queue.put(np.zeros((1, len(self.labels))))  # Return zero probabilities
                return
            prediction = self.model.predict(input_data, verbose=0)

            self.prediction_queue.put(prediction)
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()

    def check_events(self):
        self.canvas.updated = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
                self.mouse_button = event.button
                if event.button == 1:
                    self.canvas.add(event.pos, self.screen)
                if event.button == 3:
                    self.canvas.erase(event.pos, self.screen)
                if event.button == 2:
                    self.canvas.clear()
                    self.prediction = None
            if event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False
                self.mouse_button = None
            if event.type == pygame.MOUSEMOTION and self.mouse_down:
                if self.mouse_button == 1:
                    self.canvas.add(event.pos, self.screen)
                if self.mouse_button == 3:
                    self.canvas.erase(event.pos, self.screen)
            if event.type == pygame.MOUSEWHEEL:
                self.canvas.adjust_brush_size(event.y)  # event.y is 1 for scroll up, -1 for scroll down
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and not self.continuous_prediction:
                    if self.prediction_thread is None or not self.prediction_thread.is_alive():
                        self.prediction_thread = threading.Thread(target=self.predict_thread, args=(self.canvas.grid,))
                        self.prediction_thread.start()
                if event.key == pygame.K_SPACE:
                    self.continuous_prediction = not self.continuous_prediction
                if event.key == pygame.K_s:  # Save when 'S' is pressed
                    self.canvas.save_plot()
                    print("Saved digit plot to 'digit.png'")

    def update(self):
        self.screen.fill((30, 30, 30))
        self.canvas.draw(self.screen)
        
        if self.canvas.updated:
            if self.continuous_prediction:
                if self.prediction_thread is None or not self.prediction_thread.is_alive():
                    self.prediction_thread = threading.Thread(target=self.predict_thread, args=(self.canvas.grid,))
                    self.prediction_thread.start()
        
        # Check for new predictions
        try:
            while not self.prediction_queue.empty():
                self.prediction = self.prediction_queue.get_nowait()
        except queue.Empty:
            pass
        
        x_margin = 10
        y_margin = 50
        row_gap = 10

        self.screen.blit(self.title, (x_margin, x_margin))
        
        # digit = np.argmax(self.prediction)
        # print(self.prediction)
        for idx, confidence in enumerate(self.prediction[0]):
            # Create a gradient from red (low confidence) to green (high confidence)
            # passing through orange and yellow
            h = (confidence * 126)/360
            # print(confidence)
            s = 1
            l = 0.84
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            color = (int(r * 255), int(g * 255), int(b * 255))
            text = f"{self.labels[idx]}"
            text_surface = self.bold_font.render(text, True, (245, 245, 245))
            text_height = text_surface.get_height()

            bar_width = int(confidence * 150)
            bar_height = 14
            bar_x = x_margin + 10 + x_margin
            bar_y = y_margin + text_height * idx + row_gap * idx + 1 + (text_height - bar_height) / 2

            bar_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
            
            pygame.draw.rect(self.screen, color, bar_rect)
            self.screen.blit(text_surface, (x_margin, y_margin + text_height * idx + row_gap * idx))
            if confidence > 0.009:
                small_text = self.small_font.render(f"{confidence*100:.2f}%", True, (180, 180, 180))
                small_text_height = small_text.get_height()
                small_text_x = bar_x + bar_width + x_margin
                small_text_y = bar_y + (bar_height - small_text_height) / 2
                self.screen.blit(small_text, (small_text_x, small_text_y))

        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(60)
            self.check_events()
            self.update()
        pygame.quit()
        quit()

if __name__ == '__main__':
    window = Window()
    window.run()
