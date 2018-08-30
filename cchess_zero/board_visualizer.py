import matplotlib.pyplot as plt  
import matplotlib.cbook as cbook  
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2

currentfile = '/'.join(__file__.split('/')[:-1])
img_path = os.path.join(currentfile,'images')


def put_chess(l_img,s_img,cord):
    l_img.flags.writeable = True
    x_offset, y_offset = cord
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c])

def piece2img(piece):
    if piece in 'RNBAKRPC'.lower():
        return "B{}.GIF".format(piece.upper())
    else:
        return "R{}.GIF".format(piece.upper())
    
imgdic = {}
SEL = plt.imread(os.path.join(img_path,'OOS.GIF'))
for i in 'RNBAKRPC':
    picname = piece2img(i)
    picurl = os.path.join(img_path,picname)
    imgdic[i] = plt.imread(picurl)
for i in 'RNBAKRPC'.lower():
    picname = piece2img(i)
    picurl = os.path.join(img_path,picname)
    imgdic[i] = plt.imread(picurl)

def get_board_img(board,action=None):
    board_im = plt.imread(os.path.join(img_path,'WHITE.GIF'))
    def string_reverse(string):
        # return ''.join(string[len(string) - i] for i in range(1, len(string)+1))
        return ''.join(string[i] for i in range(len(string) - 1, -1, -1))

    x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

    if(action != None):
        src = action[0:2]
        to = action[2:]

        src_x = int(x_trans[src[0]])
        src_y = int(src[1])
        to_x = int(x_trans[to[0]])
        to_y = int(to[1])
    
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
    for i in range(9):
        cv2.putText(board_im,'abcdefghi'[i],(20 + 40 * i,17),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0,255),1)
        cv2.putText(board_im,'abcdefghi'[i],(20 + 40 * i,410),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0,255),1)
    for i in range(10):
        cv2.putText(board_im,'9876543210'[i],(5,33 + 40 * i),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0,255),1)
        cv2.putText(board_im,'9876543210'[i],(355,33 + 40 * i),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0,255),1)
    for i in range(9):
        #put_chess(board,king,(10 + 40 * i,10))
        for j in range(10):
            piece = board[9 - j][i]
            if piece.strip():
                put_chess(board_im,imgdic[piece],(8 + 40 * i,10 + 40 * j))
    if(action != None):
        put_chess(board_im,SEL,(8 + 40 * src_x,10 + 40 * (9 - src_y)))
        put_chess(board_im,SEL,(8 + 40 * to_x,10 + 40 * (9 - to_y)))
    return board_im
    #print("  abcdefghi")
    #for i,line in enumerate(board):
    #    if (action != None):
    #        if(i == src_y):
    #            s = list(line)
    #            s[src_x] = 'x'
    #            line = ''.join(s)
    #    print(i,line)