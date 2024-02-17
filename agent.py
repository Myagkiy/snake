import torch
import random
import numpy as np
from collections import deque
from time import time
from game import SnakeGameAI, Direction, Route
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
WIN_SCORE = 60

# train layer:
# will collide straight: [0, 1]
# will collide right: [0, 1]
# will collide left: [0, 1]
# is current direction == LEFT: [0, 1]
# is current direction == RIGHT: [0, 1]
# is current direction == UP: [0, 1]
# is current direction == DOWN: [0, 1]
# food is on the LEFT: [0, 1]
# food is on the RIGHT: [0, 1]
# food is above: [0, 1]
# food is below: [0, 1]


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0                                                        # randomness
        self.gamma = 0.9                                                        # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)                                  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.time_total = time()

    @staticmethod
    def get_state(game):
        state = [
            # Danger straight
            game.will_collide_route(Route.STRAIGHT),

            # Danger right
            game.will_collide_route(Route.RIGHT),

            # Danger left
            game.will_collide_route(Route.LEFT),

            # Move direction
            game.direction == Direction.LEFT,
            game.direction == Direction.RIGHT,
            game.direction == Direction.UP,
            game.direction == Direction.DOWN,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))   # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)        # list of tuples
        else:
            mini_sample = self.memory

        # print('Train long memory, mini_sample:', mini_sample)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []        # scores per each game step
    plot_mean_scores = []   # average scores per each step (current score / total score)
    total_score = 0
    record = 0              # maximum score
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = Agent.get_state(game)
        # print('Old state:', state_old)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = Agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            if record >= WIN_SCORE:
                agent.time_total = time() - agent.time_total
                print('Game WIN! Record:', record, f'Total time: {agent.time_total} s')
                agent.model.save(agent.n_games, record, agent.time_total)
                break
            
            game.set_record_text(record)
            game.set_game_text(agent.n_games)
            # game.set_time_text(f'{(time() - agent.time_total):{0}{6}}')
            game.set_time_text('%06d' % (time() - agent.time_total))

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            print('Scores:', plot_scores)
            print('Mean scores:', plot_mean_scores)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
