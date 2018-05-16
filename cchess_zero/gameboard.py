import numpy as np


def get_pieces_count(state):
    count = 0
    for s in state:
        if s.isalpha():
            count += 1
    return count

def is_kill_move(state_prev, state_next):
    return get_pieces_count(state_prev) - get_pieces_count(state_next)

def create_position_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[8 - l1] + numbers[n1]
            labels_array.append(move)
#     labels_array.reverse()
    return labels_array

def create_position_labels_reverse():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[l1] + numbers[n1]
            labels_array.append(move)
    labels_array.reverse()
    return labels_array

class GameBoard(object):
    board_pos_name = np.array(create_position_labels()).reshape(9,10).transpose()
    Ny = 10
    Nx = 9

    def __init__(self):
        self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"    #
        self.round = 1
        # self.players = ["w", "b"]
        self.current_player = "w"
        self.restrict_round = 0

# 小写表示黑方，大写表示红方
# [
#     "rheakaehr",
#     "         ",
#     " c     c ",
#     "p p p p p",
#     "         ",
#     "         ",
#     "P P P P P",
#     " C     C ",
#     "         ",
#     "RHEAKAEHR"
# ]
    def reload(self):
        self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"    #
        self.round = 1
        self.current_player = "w"
        self.restrict_round = 0

    @staticmethod
    def print_borad(board, action = None):
        def string_reverse(string):
            # return ''.join(string[len(string) - i] for i in range(1, len(string)+1))
            return ''.join(string[i] for i in range(len(string) - 1, -1, -1))

        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

        if(action != None):
            src = action[0:2]

            src_x = int(x_trans[src[0]])
            src_y = int(src[1])

        # board = string_reverse(board)
        board = board.replace("1", " ")
        board = board.replace("2", "  ")
        board = board.replace("3", "   ")
        board = board.replace("4", "    ")
        board = board.replace("5", "     ")
        board = board.replace("6", "      ")
        board = board.replace("7", "       ")
        board = board.replace("8", "        ")
        board = board.replace("9", "         ")
        board = board.split('/')
        # board = board.replace("/", "\n")
        print("  abcdefghi")
        for i,line in enumerate(board):
            if (action != None):
                if(i == src_y):
                    s = list(line)
                    s[src_x] = 'x'
                    line = ''.join(s)
            print(i,line)
        # print(board)

    @staticmethod
    def sim_do_action(in_action, in_state):
        x_trans = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8}

        src = in_action[0:2]
        dst = in_action[2:4]

        src_x = int(x_trans[src[0]])
        src_y = int(src[1])

        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        # GameBoard.print_borad(in_state)
        # print("sim_do_action : ", in_action)
        # print(dst_y, dst_x, src_y, src_x)
        board_positions = GameBoard.board_to_pos_name(in_state)
        line_lst = []
        for line in board_positions:
            line_lst.append(list(line))
        lines = np.array(line_lst)
        # print(lines.shape)
        # print(board_positions[src_y])
        # print("before board_positions[dst_y] = ",board_positions[dst_y])

        lines[dst_y][dst_x] = lines[src_y][src_x]
        lines[src_y][src_x] = '1'

        board_positions[dst_y] = ''.join(lines[dst_y])
        board_positions[src_y] = ''.join(lines[src_y])

        # src_str = list(board_positions[src_y])
        # dst_str = list(board_positions[dst_y])
        # print("src_str[src_x] = ", src_str[src_x])
        # print("dst_str[dst_x] = ", dst_str[dst_x])
        # c = copy.deepcopy(src_str[src_x])
        # dst_str[dst_x] = c
        # src_str[src_x] = '1'
        # board_positions[dst_y] = ''.join(dst_str)
        # board_positions[src_y] = ''.join(src_str)
        # print("after board_positions[dst_y] = ", board_positions[dst_y])

        # board_positions[dst_y][dst_x] = board_positions[src_y][src_x]
        # board_positions[src_y][src_x] = '1'

        board = "/".join(board_positions)
        board = board.replace("111111111", "9")
        board = board.replace("11111111", "8")
        board = board.replace("1111111", "7")
        board = board.replace("111111", "6")
        board = board.replace("11111", "5")
        board = board.replace("1111", "4")
        board = board.replace("111", "3")
        board = board.replace("11", "2")

        # GameBoard.print_borad(board)
        return board

    @staticmethod
    def board_to_pos_name(board):
        board = board.replace("2", "11")
        board = board.replace("3", "111")
        board = board.replace("4", "1111")
        board = board.replace("5", "11111")
        board = board.replace("6", "111111")
        board = board.replace("7", "1111111")
        board = board.replace("8", "11111111")
        board = board.replace("9", "111111111")
        return board.split("/")

    @staticmethod
    def check_bounds(toY, toX):
        if toY < 0 or toX < 0:
            return False

        if toY >= GameBoard.Ny or toX >= GameBoard.Nx:
            return False

        return True

    @staticmethod
    def validate_move(c, upper=True):
        if (c.isalpha()):
            if (upper == True):
                if (c.islower()):
                    return True
                else:
                    return False
            else:
                if (c.isupper()):
                    return True
                else:
                    return False
        else:
            return True

    @staticmethod
    def get_legal_moves(state, current_player):
        moves = []
        k_x = None
        k_y = None

        K_x = None
        K_y = None

        face_to_face = False

        board_positions = np.array(GameBoard.board_to_pos_name(state))
        for y in range(board_positions.shape[0]):
            for x in range(len(board_positions[y])):
                if(board_positions[y][x].isalpha()):
                    if(board_positions[y][x] == 'r' and current_player == 'b'):
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif(board_positions[y][x] == 'R' and current_player == 'w'):
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif ((board_positions[y][x] == 'n' or board_positions[y][x] == 'h') and current_player == 'b'):
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False) and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False) and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'N' or board_positions[y][x] == 'H') and current_player == 'w'):
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True) and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True) and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'b' or board_positions[y][x] == 'e') and current_player == 'b'):
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 5 and \
                                            board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 5 and \
                                            board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'B' or board_positions[y][x] == 'E') and current_player == 'w'):
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 4 and \
                                            board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 4 and \
                                            board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'a' and current_player == 'b'):
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'A' and current_player == 'w'):
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'k'):
                        k_x = x
                        k_y = y

                        if(current_player == 'b'):
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign

                                    if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                                upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                        moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'K'):
                        K_x = x
                        K_y = y

                        if(current_player == 'w'):
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign

                                    if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                                upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                        moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'c' and current_player == 'b'):
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break
                    elif (board_positions[y][x] == 'C' and current_player == 'w'):
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break
                    elif (board_positions[y][x] == 'p' and current_player == 'b'):
                        toY = y - 1
                        toX = x

                        if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False)):
                            moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                        if y < 5:
                            toY = y
                            toX = x + 1
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False)):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                            toX = x - 1
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False)):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    elif (board_positions[y][x] == 'P' and current_player == 'w'):
                        toY = y + 1
                        toX = x

                        if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True)):
                            moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                        if y > 4:
                            toY = y
                            toX = x + 1
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True)):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                            toX = x - 1
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True)):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

        if(K_x != None and k_x != None and K_x == k_x):
            face_to_face = True
            for i in range(K_y + 1, k_y, 1):
                if(board_positions[i][K_x].isalpha()):
                    face_to_face = False

        if(face_to_face == True):
            if(current_player == 'b'):
                moves.append(GameBoard.board_pos_name[k_y][k_x] + GameBoard.board_pos_name[K_y][K_x])
            else:
                moves.append(GameBoard.board_pos_name[K_y][K_x] + GameBoard.board_pos_name[k_y][k_x])

        return moves