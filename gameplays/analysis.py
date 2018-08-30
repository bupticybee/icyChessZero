import os
import sys

project_basedir = '..'
sys.path.append(project_basedir)
from cchess_zero.gameboard import *
extra = {"font-weight":"bold","background":"#73a1bf","resource":[]}
def re_get_km_json(tree,statestr,is_root=True,c_puct=5,move="",depth=100):
    retval = {'data':{}}
    retval['data']['text'] = "Q {:.4f} u {:.4f} vloss {:.4f} visit {} move {}".format(tree._Q,tree._u,tree.virtual_loss,tree._n_visits,move)
    visits = []
    chind_keys = []
    
    childern_mks = []
    if depth > 0:
        for one_chind_key in tree._children:
            if tree._children[one_chind_key]._n_visits == 0 and tree._children[one_chind_key].virtual_loss == 0:
                continue
            visits.append(tree._children[one_chind_key]._n_visits)
            chind_keys.append(one_chind_key)
            childern_mks.append(re_get_km_json(tree._children[one_chind_key]
                                                ,GameBoard.sim_do_action(one_chind_key,statestr)
                                                ,move=one_chind_key,depth=depth - 1
                                              )
                               )
        if visits:
            visits = np.asarray(visits)
            childern_mks[np.argmax(visits)]['data'].update(extra)
    retval['children'] = childern_mks
        
    return retval

def get_km_json(mstc_policy,statestr,c_puct=5,depth=100):
    rootdic = re_get_km_json(mstc_policy._root,statestr,c_puct=5,depth=depth)
    retdata = {
            "root":rootdic,
            "template": "default",
            "theme": "fresh-blue",
            "version": "1.4.43"
        }
    return retdata