from agent import Agent, SnakeGameAI
from model import Linear_QNet
import torch
import pygame
import os, sys

def get_action(model, state):
    final_move = [0, 0, 0]
    state0 = torch.tensor(state, dtype=torch.float)
    prediction = model(state0)
    move = torch.argmax(prediction).item()
    final_move[move] = 1

    return final_move


def play():
    agent = Agent()
    game = SnakeGameAI()
    model = Linear_QNet(11, 256, 3)
    model.load_state_dict(torch.load("model/model.pth"))
    model.eval()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = get_action(model, state_old)

        # perform move and get new state
        _, done, _ = game.play_step(final_move)

        if done:
            pygame.quit()
            # sys.quit()


if __name__ == "__main__":
    play()