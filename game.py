import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Route(Enum):
    STRAIGHT = 1
    LEFT = 2
    RIGHT = 3

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 80
MAX_DUMMY_STEPS = 100  # max amount of steps w/o food feed
DEFAULT_REWARD = 10

# DONE: record >= 60 == WIN
# DONE: time measurements of game round execution
# DONE: total time spent on training
# DONE: store trained model
# TODO: calculations over the field with simple dimensions (coordinates): 31x31 (not just screen resolution)

# Snake is nothing but an array of coordinates it takes over the field
# Snake initial coordinates: [center, center and left shift, center and double left shift
#           [‾‾‾‾‾‾‾‾‾‾]
#           [          ]
#           [ X X O    ]
#           [          ]
#           [__________]


class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.record_text = font.render("Record: 0", True, (255, 255, 255, 0.3))
        self.game_text = font.render("Game: 0", True, (255, 255, 255, 0.3))
        self.time_text = font.render("Time: 0 s", True, (255, 255, 255, 0.3))

        self.reset()
    
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _rand_coord(self, horizontal: bool = True) -> int:
        return random.randint(0, ((self.w if horizontal else self.h) - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

    def _place_food(self):
        x = self._rand_coord()
        y = self._rand_coord(horizontal=False)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action=None):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > MAX_DUMMY_STEPS * len(self.snake):
            print('Game over:', self.frame_iteration, MAX_DUMMY_STEPS * len(self.snake), 'Collision:', self.is_collision())
            game_over = True
            reward = -DEFAULT_REWARD
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = DEFAULT_REWARD
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False

    def will_collide(self, cur_dir: Direction, future_dir: Direction) -> bool:
        future_pt: Point
        match future_dir:
            case Direction.UP:
                future_pt = Point(self.head.x, self.head.y - BLOCK_SIZE)
            case Direction.DOWN:
                future_pt = Point(self.head.x, self.head.y + BLOCK_SIZE)
            case Direction.LEFT:
                future_pt = Point(self.head.x - BLOCK_SIZE, self.head.y)
            case Direction.RIGHT:
                future_pt = Point(self.head.x + BLOCK_SIZE, self.head.y)
            case _:
                future_pt = None

        if future_pt is None:
            return False

        return self.direction == cur_dir and self.is_collision(future_pt)

    def will_collide_route(self, future_route: Route):
        match future_route:
            case Route.STRAIGHT:
                return (self.will_collide(Direction.RIGHT, Direction.RIGHT) or
                        self.will_collide(Direction.LEFT, Direction.LEFT) or
                        self.will_collide(Direction.UP, Direction.UP) or
                        self.will_collide(Direction.DOWN, Direction.DOWN))
            case Route.LEFT:
                return (self.will_collide(Direction.DOWN, Direction.RIGHT) or
                        self.will_collide(Direction.UP, Direction.LEFT) or
                        self.will_collide(Direction.RIGHT, Direction.UP) or
                        self.will_collide(Direction.LEFT, Direction.DOWN))
            case Route.RIGHT:
                return (self.will_collide(Direction.UP, Direction.RIGHT) or
                        self.will_collide(Direction.DOWN, Direction.LEFT) or
                        self.will_collide(Direction.LEFT, Direction.UP) or
                        self.will_collide(Direction.RIGHT, Direction.DOWN))
            case _:
                return False

    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, (255, 255, 255, 0.3))
        self.display.blit(text, [0, 0])

        self.display.blit(self.record_text, [150, 0])
        self.display.blit(self.game_text, [300, 0])
        self.display.blit(self.time_text, [450, 0])

        pygame.display.flip()

    def set_record_text(self, value):
        self.record_text = font.render(f'Record: {value}', True, (255, 255, 255, 0.3))

    def set_game_text(self, value):
        self.game_text = font.render(f'Game: {value}', True, (255, 255, 255, 0.3))

    def set_time_text(self, value):
        self.time_text = font.render(f'Time: {value} s', True, (255, 255, 255, 0.3))

    def _move(self, action=None):
        if action is not None:
            # action: [straight, right, left]

            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]  # no change
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
            else:  # [0, 0, 1]
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

            self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)


if __name__ == "__main__":
    game = SnakeGameAI()

    while True:
        reward, game_over, score = game.play_step()

        if game_over:
            break

    print('Final score:', score)

    pygame.quit()
