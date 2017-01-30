#!/usr/bin/env python
from operator import itemgetter

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
        # get unique moves using set then get the number of moves left
        p1_num_moves = len(set(sum(game.get_legal_moves().values(), [])))
        p2_num_moves = len(set(sum(game.get_opponent_moves().values(), [])))

        if maximizing_player_turn:
            return p1_num_moves - p2_num_moves
        else:
            return p2_num_moves - p1_num_moves


# Submission Class 2
class CustomEvalFn():
    """Custom evaluation function that acts
    however you think it should. This is not
    required but highly encouraged if you
    want to build the best AI possible."""
    def score(self, game, maximizing_player_turn=True):
        # get unique moves using set then get the number of moves left
        p1_num_moves = len(set(sum(game.get_legal_moves().values(), [])))
        p2_num_moves = len(set(sum(game.get_opponent_moves().values(), [])))

        if maximizing_player_turn:
            return p1_num_moves - (3 * p2_num_moves)
        else:
            return p2_num_moves - (3 * p1_num_moves)


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
    def __init__(self, search_depth=3, eval_fn=CustomEvalFn(), algo='alphabeta'):
        # if you find yourself with a superior eval function, update the
        # default value of `eval_fn` to `CustomEvalFn()`
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.algo = algo
        self.time_limit = 2000

    def move(self, game, legal_moves, time_left):
        if self.algo == 'minimax':
            best_move, best_queen, utility = self.minimax(game, time_left, depth=self.search_depth, maximizing_player=True)
        elif self.algo == 'alphabeta':
            best_move, best_queen, utility = self.alphabeta(game, time_left, depth=self.search_depth, maximizing_player=True)
        else:
            # iterative deepening and alpha beta pruning
            move_dict = {}
            # use array because minimax/alpha-beta pruning select the left most path
            move_array = []
            depth = 2
            while time_left() > self.time_limit:
                move, queen, utility = self.alphabeta(game, time_left, depth=depth, maximizing_player=True)
                key = (move, queen)
                if key in move_dict:
                    move_dict[key] += 1
                else:
                    move_dict[key] = 1
                    move_array.append(key)

                if move_dict[key] >= 4:
                    break

                depth += 1

            best_move = None
            best_queen = None
            select_count = 0
            # move thru the array selecting the one with the highest count and the left most position
            for key in move_array:
                count = move_dict[key]
                if select_count < count:
                    select_count = count
                    best_move, best_queen = key

        return best_move, best_queen 

    def utility(self, game):
        """TODO: Update this function to calculate the utility of a game state"""
        return self.eval_fn.score(game)

    def minimax(self, game, time_left, depth=float("inf"), maximizing_player=True):
        best_move = None
        best_queen = None

        moves = [(queen, move) for queen, legal_moves in game.get_legal_moves().iteritems() for move in legal_moves]
        value_list = []

        if depth == 1 or len(moves) == 0:
            return best_move, best_queen, self.utility(game)

        for queen, move in moves:
            forecasted_game = game.forecast_move(move, queen)
            _, _, val = self.minimax(forecasted_game, time_left, depth=(depth - 1), maximizing_player=(not maximizing_player))
            value_list.append(val)

        if maximizing_player:
            index, best_val = max(enumerate(value_list), key=itemgetter(1))
        else:
            index, best_val = min(enumerate(value_list), key=itemgetter(1))

        best_queen, best_move = moves[index]

        return best_move, best_queen, best_val

    def alphabeta(self, game, time_left, depth=float("inf"), alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        best_move = (-1, -1)
        best_queen = -1

        move_dict = game.get_legal_moves()
        moves = set(sum(move_dict.values(), []))

        if depth == 1 or len(moves) == 0 or time_left() < self.time_limit:
            return best_move, best_queen, self.utility(game)

        if maximizing_player:
            val = float('-inf')
            for queen in move_dict:
                for move in move_dict[queen]:
                    forecasted_game = game.forecast_move(move, queen)
                    _, _, next_val = self.alphabeta(forecasted_game, time_left, depth=(depth - 1),
                                                    alpha=alpha, beta=beta, maximizing_player=(not maximizing_player))
                    val = max(val, next_val)
                    if alpha < val:
                        alpha = val
                        best_move = move
                        best_queen = queen
                    if beta <= alpha:
                        break
            return best_move, best_queen, val
        else:
            val = float('inf')
            for queen in move_dict:
                for move in move_dict[queen]:
                    forecasted_game = game.forecast_move(move, queen)
                    _, _, next_val = self.alphabeta(forecasted_game, time_left, depth=(depth - 1),
                                                    alpha=alpha, beta=beta, maximizing_player=(not maximizing_player))
                    val = min(val, next_val)
                    if beta > val:
                        beta = val
                        best_move = move
                        best_queen = queen
                    if beta <= alpha:
                        break
            return best_move, best_queen, val
