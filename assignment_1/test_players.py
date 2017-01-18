from random import randint


class RandomPlayer():
    """Player that chooses a move randomly."""    

    def move(self, game, legal_moves, time_left):
        if not legal_moves: return (-1,-1)        
        num=randint(game.__active_players_queen1__,game.__active_players_queen2__)
        if not len(legal_moves[num]):
            num = game.__active_players_queen1__ if num == game.__active_players_queen2__ else game.__active_players_queen2__
            if not len(legal_moves[num]):
                return (-1,-1),num
        
        moves=legal_moves[num][randint(0,len(legal_moves[num])-1)]
        return moves,num
    


class HumanPlayer():
    """Player that chooses a move according to
    user's input."""
    def move(self, game, legal_moves, time_left):
        i=0
        choice = {}
        if not len(legal_moves[game.__active_players_queen1__]) and not len(legal_moves[game.__active_players_queen2__]):
            return None, None
        for queen in legal_moves:
                for move in legal_moves[queen]:        
                    choice.update({i:(queen,move)})
                    print('\t'.join(['[%d] q%d: (%d,%d)'%(i,queen,move[0],move[1])] ))
                    i=i+1
        
        
        valid_choice = False
        while not valid_choice:
            try:
                index = int(input('Select move index:'))
                valid_choice = 0 <= index < i

                if not valid_choice:
                    print('Illegal move! Try again.')
            
            except ValueError:
                print('Invalid index! Try again.')
        
        return choice[index][1],choice[index][0]