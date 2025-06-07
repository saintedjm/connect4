from typing import List
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Tuple, MultiBinary
from colorama import Fore


class Connect4Env(gym.Env):
    """
        GameState for the Connect 4 game.
        The board is represented as a 2D array (rows and columns).
        Each entry on the array can be:
            0 = empty    (.)
            1 = player 1 (X)
            2 = player 2 (O)

        Winner can be:
             None = No winner (yet)
            -1 = Draw
             1 = player 1 (X)
             2 = player 2 (O)
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
    }

    def __init__(self, width=7, height=6, connect=4, render_mode=None):
        self.num_players = 2

        self.width = width
        self.height = height
        self.connect = connect

        self.render_mode = render_mode

        # 3: Channels. Empty cells, p1 chips, p2 chips
        player_observation_space = MultiBinary([self.num_players + 1,
                                               self.width, self.height])
        self.observation_space = Tuple([player_observation_space
                                        for _ in range(self.num_players)])
        self.action_space = Discrete(self.width)

        # Naive calculation. There are height * width individual cells
        # and each one can have 3 values. This is also encapsulates
        # invalid cases where a chip rests on top of an empy cell.
        self.state_space_size = 3**(self.height * self.width)

        self.reset()

    def reset(self, seed=None, options=None) -> List[np.ndarray]:
        """
        Initialises the Connect 4 gameboard.
        """
        super().reset(seed=seed)
        self.board = np.full((self.width, self.height), -1)

        self.current_player = 0 # Player 1 (represented by value 0) will move now
        self.winner = None
        self.winning_positions = None  # Track winning positions for rendering

        info = self._get_info()

        return (self.get_player_observations(), info)

    def filter_observation_player_perspective(self, player: int) -> List[np.ndarray]:
        opponent = 0 if player == 1 else 1
        # One hot channel encoding of the board
        empty_positions = np.where(self.board == -1, 1, 0)
        player_chips   = np.where(self.board == player, 1, 0)
        opponent_chips = np.where(self.board == opponent, 1, 0)
        return np.array([empty_positions, player_chips, opponent_chips])

    def get_player_observations(self) -> List[np.ndarray]:
        p1_state = self.filter_observation_player_perspective(0)
        p2_state = np.array([np.copy(p1_state[0]),
                             np.copy(p1_state[-1]), np.copy(p1_state[-2])])
        return (p1_state, p2_state)

    def clone(self):
        """
        Creates a deep copy of the game state.
        NOTE: it is _really_ important that a copy is used during simulations
              Because otherwise MCTS would be operating on the real game board.
        :returns: deep copy of this GameState
        """
        st = Connect4Env(width=self.width, height=self.height)
        st.current_player = self.current_player
        st.winner = self.winner
        st.board = np.array([self.board[col][:] for col in range(self.width)])
        return st

    def step(self, action):
        """
        Changes this GameState by "dropping" a chip in the column
        specified by param movecol.
        :param movecol: column over which a chip will be dropped
        """
        movecol = action
        if not(movecol >= 0 and movecol <= self.width and self.board[movecol][self.height - 1] == -1):
            raise IndexError(f'Invalid move. tried to place a chip on column {movecol} which is already full. Valid moves are: {self.get_moves()}')
        row = self.height - 1
        while row >= 0 and self.board[movecol][row] == -1:
            row -= 1

        row += 1

        self.board[movecol][row] = self.current_player
        self.current_player = 1 - self.current_player

        self.winner, reward_vector = self.check_for_episode_termination(movecol, row)
        # Store winning positions only if render_mode is 'rgb_array'
        if self.render_mode == 'rgb_array' and self.winner is not None and self.winner != -1:
            self.winning_positions = self.get_winning_positions(movecol, row)
        else:
            self.winning_positions = None

        info = self._get_info()

        return self.get_player_observations(), reward_vector, \
               self.winner is not None, False, info
    
    def _get_info(self):
        return {
            'legal_actions': self.get_moves(),
            'current_player': self.current_player,
        }

    def check_for_episode_termination(self, movecol, row):
        winner, reward_vector = self.winner, [0, 0]
        if self.does_move_win(movecol, row):
            winner = 1 - self.current_player
            if winner == 0: reward_vector = [1, -1]
            elif winner == 1: reward_vector = [-1, 1]
        elif self.get_moves() == []:  # A draw has happened
            winner = -1
            reward_vector = [0, 0]
        return winner, reward_vector

    def get_moves(self):
        """
        :returns: array with all possible moves, index of columns which aren't full
        """
        if self.winner is not None:
            return []
        return [col for col in range(self.width) if self.board[col][self.height - 1] == -1]

    def does_move_win(self, x, y):
        """
        Checks whether a newly dropped chip at position param x, param y
        wins the game.
        :param x: column index
        :param y: row index
        :returns: (boolean) True if the previous move has won the game
        """
        me = self.board[x][y]
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
            p = 1
            while self.is_on_board(x+p*dx, y+p*dy) and self.board[x+p*dx][y+p*dy] == me:
                p += 1
            n = 1
            while self.is_on_board(x-n*dx, y-n*dy) and self.board[x-n*dx][y-n*dy] == me:
                n += 1

            if p + n >= (self.connect + 1): # want (p-1) + (n-1) + 1 >= 4, or more simply p + n >= 5
                return True

        return False

    def is_on_board(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def get_result(self, player):
        """
        :param player: (int) player which we want to see if he / she is a winner
        :returns: winner from the perspective of the param player
        """
        if self.winner == -1: return 0  # A draw occurred
        return +1 if player == self.winner else -1

    def render(self):
        mode = self.render_mode
        if mode is None:
            return None
        if mode == 'human':
            s = ""
            for x in range(self.height - 1, -1, -1):
                for y in range(self.width):
                    s += {-1: Fore.WHITE + '.', 0: Fore.RED + 'X', 1: Fore.YELLOW + 'O'}[self.board[y][x]]
                    s += Fore.RESET
                s += "\n"
            print(s)
        elif mode == 'rgb_array':
            return self._render_rgb_array()
        else:
            raise NotImplementedError(f'Rendering mode {mode} has not been coded yet')

    def get_winning_positions(self, x, y):
        """
        Returns the list of positions [(col, row), ...] that form the winning line.
        """
        me = self.board[x][y]
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
            positions = [(x, y)]
            # Forward direction
            p = 1
            while self.is_on_board(x+p*dx, y+p*dy) and self.board[x+p*dx][y+p*dy] == me:
                positions.append((x+p*dx, y+p*dy))
                p += 1
            # Backward direction
            n = 1
            while self.is_on_board(x-n*dx, y-n*dy) and self.board[x-n*dx][y-n*dy] == me:
                positions.append((x-n*dx, y-n*dy))
                n += 1
            if len(positions) >= self.connect:
                return positions
        return []

    def _render_rgb_array(self):
        """
        Returns an RGB array representation of the board.
        Board shape: (height, width, 3)
        Colors:
            Empty: white (255,255,255)
            Player 1: red (255,0,0)
            Player 2: yellow (255,255,0)
        """
        cell_size = 40  # pixels per cell
        board_rgb = np.ones((self.height * cell_size, self.width * cell_size, 3), dtype=np.uint8) * 255

        color_map = {
            -1: (255, 255, 255),  # empty
             0: (255, 0, 0),      # player 1 (red)
             1: (0, 0, 0),    # player 2 (black)
        }

        for x in range(self.width):
            for y in range(self.height):
                color = color_map[self.board[x][y]]
                y_start = (self.height - 1 - y) * cell_size
                y_end = y_start + cell_size
                x_start = x * cell_size
                x_end = x_start + cell_size
                # Draw filled circle in the cell
                cy, cx = (y_start + cell_size // 2, x_start + cell_size // 2)
                for i in range(y_start, y_end):
                    for j in range(x_start, x_end):
                        if (i - cy) ** 2 + (j - cx) ** 2 < (cell_size // 2 - 2) ** 2:
                            board_rgb[i, j] = color

        # Draw winning line if game is over and there is a winning line
        if self.winning_positions and len(self.winning_positions) >= self.connect:
            # Sort positions to draw line from start to end
            sorted_positions = sorted(self.winning_positions, key=lambda pos: (pos[0], pos[1]))
            start = sorted_positions[0]
            end = sorted_positions[-1]
            cell_size = 40
            def center(col, row):
                x = col * cell_size + cell_size // 2
                y = (self.height - 1 - row) * cell_size + cell_size // 2
                return (x, y)
            x0, y0 = center(*start)
            x1, y1 = center(*end)
            # Draw a thick green line over the winning positions
            self._draw_line(board_rgb, x0, y0, x1, y1, color=(0,255,0), thickness=6)
        return board_rgb

    def _draw_line(self, img, x0, y0, x1, y1, color=(0,255,0), thickness=4):
        """
        Draw a line on a numpy RGB image using Bresenham's algorithm.
        """
        import math
        dx = x1 - x0
        dy = y1 - y0
        dist = int(math.hypot(dx, dy))
        for t in range(dist+1):
            x = int(x0 + dx * t / dist)
            y = int(y0 + dy * t / dist)
            for tx in range(-thickness//2, thickness//2+1):
                for ty in range(-thickness//2, thickness//2+1):
                    xi = x + tx
                    yi = y + ty
                    if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                        img[yi, xi] = color