#!/usr/bin/env python

# This file is your main submission that will be graded against. Only copy-paste
# code on the relevant classes included here from the IPython notebook. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.

# Submission Class 1
class OpenMoveEvalFn():
    """Evaluation function that outputs a 
    score equal to how many moves are open
    for AI player on the board minus
    the moves open for opponent player."""
    def score(self, game, maximizing_player_turn=True):
        # TODO: finish this function!
        return eval_func
        
# Submission Class 2
class CustomEvalFn():
    """Custom evaluation function that acts
    however you think it should. This is not
    required but highly encouraged if you
    want to build the best AI possible."""
    def score(self, game, maximizing_player_turn=True):
        # TODO: finish this function!
        return eval_func

class CustomPlayer():
    # TODO: finish this class!
    """Player that chooses a move using 
    your evaluation function and 
    a depth-limited minimax algorithm 
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move
    in less than 5 seconds."""
    def __init__(self,  search_depth=3, eval_fn=OpenMoveEvalFn()):
        # if you find yourself with a superior eval function, update the
        # default value of `eval_fn` to `CustomEvalFn()`
        self.eval_fn = eval_fn
        self.search_depth = search_depth

    
    def move(self, game, legal_moves, time_left):
        best_move,best_queen, utility = self.minimax(game,time_left, depth=self.search_depth)   
        #change minimax to alphabeta after completing alphabeta part of assignment 
        return best_move, best_queen 


    def utility(self, game):
        """TODO: Update this function to calculate the utility of a game state"""
        return self.eval_fn.score(game)


    def minimax(self, game, time_left, depth=float("inf"), maximizing_player=True):
        # TODO: finish this function!
        return best_move,best_queen, best_val

    def alphabeta(self, game, time_left, depth=float("inf"), alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        # TODO: finish this function!
        return best_move, best_queen, val
