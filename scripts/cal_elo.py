import os
import numpy as np
import time
#evalue_dir = 'data/validate/2018-05-25'
#stamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))
stamp = sorted([i for i in os.listdir('data/validate/') if '_blank' not in i])[-1]
play_save_root = 'data/validate'
evalue_dir = os.path.join(play_save_root,stamp)

chessplays = os.listdir(evalue_dir)

new_score = 0
old_score = 0

nw_win,nw_lose,ow_win,ow_lose = 0,0,0,0
peaces = 0
for one_play in chessplays:
    if 'mcts_peace' in one_play or 'mcts_-1' in one_play:
        new_score += 0.5
        old_score += 0.5
        peaces += 1
        continue
    if 'new-old' in one_play:
        if 'mcts_w' in one_play:
            new_score += 1
            nw_win += 1
        elif 'mcts_b' in one_play:
            old_score += 1
            nw_lose += 1
    elif 'old-new' in one_play:
        if 'mcts_b' in one_play:
            new_score += 1
            ow_lose += 1
        elif 'mcts_w' in one_play:
            old_score += 1
            ow_win += 1
            
new_score = new_score / len(chessplays)
old_score = old_score / len(chessplays)

elo = np.log10(1 / old_score - 1) * 400

with open('data/log_update','a') as whdl:
    whdl.write("{} {} {} {} {} {} {}\n".format(evalue_dir,elo,nw_win,nw_lose,ow_win,ow_lose,peaces))
