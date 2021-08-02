import numpy as np
import gym
from gym import spaces


class sokoban(gym.Env):
    _characters = {
        " ": 0,  # free space
        "@": 1,  # player
        "#": 2,  # wall
        "$": 3,  # box
        ".": 4,  # goal
        "*": 5,  # box in goal
        "+": 6,  # player in goal
    }

    _num_to_char = {v: k for k, v in _characters.items()}
    _directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Top, right, down, left

    TOP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, text, step_rewards=True):
        super(sokoban, self).__init__()
        self.rewards = step_rewards
        self._lines = text.split('\n')
        self.height = len(self._lines)
        self.width = max(map(lambda x: len(x), self._lines))

        # OpenAI gym specifics
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=6, shape=(1, self.height, self.width), dtype=np.uint8)

        self.board = np.zeros(shape=(1, self.height, self.width), dtype=np.uint8)
        self.reset()
        self.legal_moves = self.get_moves()

    def reset(self):
        """
        A method to reset the environment to the initial state of the puzzle described in the text file passed as
        argument while initializing the sokoban environment.

        :returns: The board representation of the text file as a numpy array.

        """
        self.board = np.zeros(shape=(1, self.height, self.width), dtype=np.uint8)
        for i, line in enumerate(self._lines):
            for j, character in enumerate(line):
                self.board[0][i, j] = self._characters[character]
        return self.board

    def step(self, action):
        """
        :param action ::
        :returns

        """
        move = self._directions[action]
        reward = self.execute_move(move)
        is_done = self.check_if_solved()
        reward = 10 if is_done else reward
        info = {}
        return self.board, reward, is_done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        str_board = np.vectorize(self._num_to_char.get)(self.board[0])
        text = '\n'.join([''.join(row) for row in str_board])
        print(text)

    def __str__(self):
        str_board = np.vectorize(self._num_to_char.get)(self.board[0])
        text = '\n'.join([''.join(row) for row in str_board])
        return text

    def check_if_solved(self):
        return False if self._find_unattended_boxes() else True

    def get_moves(self):
        player_coords = self._find_player()
        possible_moves = set()

        for move in self._directions:
            next_x, next_y = player_coords[0] + move[0], player_coords[1] + move[1]

            if self.board[0][next_x, next_y] != self._characters["#"]:

                # Check if the next move is an empty space or a empty goal space
                if self.board[0][next_x, next_y] in [self._characters[" "], self._characters["."]]:
                    possible_moves.add(move)

                # Check if the next move location has a box
                if self.board[0][next_x, next_y] in [self._characters["$"], self._characters["*"]]:
                    beyond_x, beyond_y = self._get_beyond_coords(action=self._directions.index(move))

                    # Check if the beyond location has an empty space or empty goal
                    if self.board[0][beyond_x, beyond_y] in [self._characters[" "], self._characters["."]]:
                        possible_moves.add(move)

        return possible_moves

    def execute_move(self, move):

        step_reward = -0.1

        if move in self.legal_moves:
            player_x, player_y = self._find_player()
            target_x, target_y = player_x + move[0], player_y + move[1]
            target_char = self.board[0][target_x, target_y]
            beyond_x, beyond_y = target_x + move[0], target_y + move[1]
            beyond_char = self.board[0][beyond_x, beyond_y]

            # Empty space
            if target_char not in [self._characters["$"], self._characters["*"]]:
                # The new location
                if target_char == self._characters["."]:
                    self.board[0][target_x, target_y] = self._characters["+"]
                else:
                    self.board[0][target_x, target_y] = self._characters["@"]
            # Box case
            else:
                # box pushing
                if beyond_char == self._characters["."]:
                    self.board[0][beyond_x, beyond_y] = self._characters["*"]
                    step_reward = 1.0
                else:
                    self.board[0][beyond_x, beyond_y] = self._characters["$"]

                # player move
                if target_char == self._characters["*"]:
                    self.board[0][target_x, target_y] = self._characters["+"]
                    step_reward = -1.0
                else:
                    self.board[0][target_x, target_y] = self._characters["@"]

            # the old location
            if self.board[0][player_x, player_y] == self._characters["+"]:
                self.board[0][player_x, player_y] = self._characters["."]
            else:
                self.board[0][player_x, player_y] = self._characters[" "]

            self.legal_moves = self.get_moves()
        return step_reward

    def _count_boxes_in_goals(self):
        _x, _y = np.where(self.board[0] == self._characters["*"])
        return len(list(zip(_x, _y)))

    def median_distance(self):

        manhattan = lambda c1, c2: sum([abs(a - b) for a, b in zip(c1, c2)])
        boxes = self._find_unattended_boxes()
        goals = self._find_unattended_goals()

        assert len(boxes) == len(goals)
        if len(goals) == 0:
            return 0

        else:
            distances = [manhattan(c1, c2) for c1, c2 in zip(boxes, goals)]
            return np.median(distances)

    def _find_unattended_boxes(self):

        _x, _y = np.where(self.board[0] == self._characters["$"])
        return list(zip(_x, _y))

    def _find_unattended_goals(self):

        _x, _y = np.where(self.board[0] == self._characters["."])
        return list(zip(_x, _y))

    def _find_player(self):
        _x, _y = np.where((self.board[0] == self._characters["@"]) | (self.board[0] == self._characters["+"]))
        return _x, _y

    def _get_beyond_coords(self, action):

        move = self._directions[action]
        player_coords = self._find_player()
        return tuple(int(b + 2 * a) for a, b in zip(move, player_coords))

if __name__ == '__main__':
    puzzle = """
     #####
    ##.@ #
    # $$ #
    #.  ##
    #####
    """

    env = sokoban(text=puzzle)
    move_dict = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
    moves = ['left', 'down', 'up', 'right', 'right', 'down', 'left', 'down', 'left', 'up']

    env.reset()
    for mve in moves:
        action = move_dict[mve]
        obs, reward, is_done, _ = env.step(action)
        print("MOVE: {} :::: REWARD: {}".format(mve, reward))
        if is_done:
            print("Solved")
