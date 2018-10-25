import asyncio
import time
import numpy as np
import copy
from tqdm import tqdm

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p,noise=False):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self.virtual_loss = 0
        self.noise = noise

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        # dirichlet noise should be applied when every select action 
        if False and self.noise == True and self._parent == None:
            #print("noise")
            noise_d =  np.random.dirichlet([0.3] * len(action_priors))
            for (action, prob),one_noise in zip(action_priors,noise_d):
                if action not in self._children:
                    prob = (1 - 0.25) * prob + 0.25 * one_noise
                    self._children[action] = TreeNode(self, prob, noise=self.noise)
        else:
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        if self.noise == False:
            return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))
        elif self.noise == True and self._parent != None:
            return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))
        else:
            noise_d =  np.random.dirichlet([0.3] * len(self._children))
            return max(list(zip(noise_d,self._children.items())),
                   key=lambda act_node: act_node[1][1].get_value(c_puct,noise_p=act_node[0]))[1]

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct,noise_p=None):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        if noise_p is None:
            self._u = (c_puct * self._P *
                       np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
            return self._Q + self._u + self.virtual_loss
        else:
            self._u = (c_puct * (self._P * 0.75 + noise_p * 0.25) *
                       np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
            return self._Q + self._u + self.virtual_loss

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000,search_threads=32,virtual_loss=3,policy_loop_arg=False,dnoise=False,):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0, noise=dnoise)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.virtual_loss = virtual_loss
        self.loop = asyncio.get_event_loop()
        self.policy_loop_arg = policy_loop_arg
        self.sem = asyncio.Semaphore(search_threads)
        self.now_expanding = set()
        
        self.select_time = 0
        self.policy_time = 0
        self.update_time = 0
        
        self.num_proceed = 0
        self.dnoise = dnoise

    async def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        with await self.sem:
            node = self._root
            road = []
            while(1):
                while node in self.now_expanding:
                    await asyncio.sleep(1e-4)
                start = time.time()
                if node.is_leaf():
                    break
                # Greedily select next move.
                action, node = node.select(self._c_puct)
                road.append(node)
                node.virtual_loss -= self.virtual_loss
                state.do_move(action)
                self.select_time += (time.time() - start)
            
            # at leave node if long check or long catch then cut off the node
            if state.should_cutoff():
                # cut off node
                for one_node in road:
                    one_node.virtual_loss += self.virtual_loss
                # now at this time, we do not update the entire tree branch, the accuracy loss is supposed to be small
                # node.update_recursive(-leaf_value)
                
                # set virtual loss to -inf so that other threads would not visit the same node again(so the node is cut off)
                node.virtual_loss = - np.inf
                #node.update_recursive(leaf_value)
                self.update_time += (time.time() - start)
                # however the proceed number still goes up 1
                self.num_proceed += 1
                return 
                
            
            start = time.time()
            self.now_expanding.add(node)
            # Evaluate the leaf using a network which outputs a list of
            # (action, probability) tuples p and also a score v in [-1, 1]
            # for the current player
            if self.policy_loop_arg == False:
                action_probs, leaf_value = await self._policy(state)
            else:
                action_probs, leaf_value = await self._policy(state,self.loop)
            self.policy_time += (time.time() - start)
            
            
            start = time.time()
            # Check for end of game.
            end, winner = state.game_end()
            if not end:
                node.expand(action_probs)
            else:
                # for end state，return the "true" leaf_value
                if winner == -1:  # tie
                    leaf_value = 0.0
                else:
                    leaf_value = (
                        1.0 if winner == state.get_current_player() else -1.0
                    )

            # Update value and visit count of nodes in this traversal.
            for one_node in road:
                one_node.virtual_loss += self.virtual_loss
            node.update_recursive(-leaf_value)
            self.now_expanding.remove(node)
            #node.update_recursive(leaf_value)
            self.update_time += (time.time() - start)
            self.num_proceed += 1

    def get_move_probs(self, state, temp=1e-3,verbose=False,predict_workers = [],can_apply_dnoise=False):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        if can_apply_dnoise == False:
            self._root.noise = False
        coroutine_list = []
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            coroutine_list.append(self._playout(state_copy))
        coroutine_list += predict_workers
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move,allow_legacy=True):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        self.num_proceed = 0
        if last_move in self._root._children and allow_legacy:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0, noise=self.dnoise)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)