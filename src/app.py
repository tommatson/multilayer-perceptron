import pygame
import sys
import subprocess

# Constants
GRID_SIZE = 28
CELL_SIZE = 20
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 100  # Extra space for UI
BUTTON_HEIGHT = 40

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_COLOR = (200, 200, 200)
BUTTON_COLOR = (180, 180, 255)
BUTTON_HOVER = (150, 150, 255)
TEXT_COLOR = BLACK

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Drawing Window")
font = pygame.font.SysFont(None, 24)
big_font = pygame.font.SysFont(None, 48)
clock = pygame.time.Clock()

# Grid data
grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
prediction = None

def draw_grid():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + 50, CELL_SIZE, CELL_SIZE)
            color = BLACK if grid[y][x] == 1 else WHITE
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

def draw_buttons():
    clear_button_rect = pygame.Rect(10, GRID_SIZE * CELL_SIZE + 60, 100, BUTTON_HEIGHT)
    predict_button_rect = pygame.Rect(120, GRID_SIZE * CELL_SIZE + 60, 100, BUTTON_HEIGHT)

    for rect, label in [(clear_button_rect, "Clear"), (predict_button_rect, "Predict")]:
        color = BUTTON_HOVER if rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)
        text = font.render(label, True, TEXT_COLOR)
        screen.blit(text, (rect.x + 15, rect.y + 10))

    return clear_button_rect, predict_button_rect

def draw_prediction():
    if prediction is not None:
        text = big_font.render(f"Prediction: {prediction}", True, BLACK)
        screen.blit(text, (10, 5))

def get_grid_as_1d_array():
    return [cell for row in grid for cell in row]

def clear_grid():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            grid[y][x] = 0

# Main loop
running = True
drawing = False
while running:
    screen.fill(WHITE)
    draw_prediction()
    draw_grid()
    clear_button_rect, predict_button_rect = draw_buttons()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Final grid as 1D array:")
            print(get_grid_as_1d_array())
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if clear_button_rect.collidepoint(event.pos):
                clear_grid()
                prediction = None
            elif predict_button_rect.collidepoint(event.pos):
                # Script for when the prediction button is pressed
                colourList = get_grid_as_1d_array()
                with open("data/input.txt", "w") as f:
                    f.write(' '.join(str(x) for x in colourList))
                
                result = subprocess.run(['./build/main'], capture_output=True, text=True)

                print("Return code:", result.returncode)
                print("Stdout:", result.stdout.strip())
                print("Stderr:", result.stderr.strip())

                with open("data/output.txt", "r") as f:
                    prediction = int(f.read().strip())
                
            else:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

    if drawing:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if 50 <= mouse_y < 50 + GRID_SIZE * CELL_SIZE:
            grid_x = mouse_x // CELL_SIZE
            grid_y = (mouse_y - 50) // CELL_SIZE
            if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                grid[grid_y][grid_x] = 1

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
