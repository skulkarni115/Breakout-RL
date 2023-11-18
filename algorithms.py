import gym
from typing import Optional
from collections import defaultdict
import numpy as np
from typing import Optional, Sequence
import random
from typing import Callable, Tuple
from tqdm import trange
import time

def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO (Done?)
    #use n = 4
    n = 4
    
    #Initialize Q Arbitrarily
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    steps = []
    total_re = []
    
    #Loop for each episode
    for i in trange(num_steps):
        #Initialize and store S0 and make sure its not terminal
        S,_ = env.reset()
        S = S.tobytes()
        #Select and store an action A0
        A = e_greedy(Q,S,epsilon)
        #T <- inf
        T = float('inf')
        t = 0
        states = [S]
        actions = [A]
        rewards = [0]
        #Loop for t = 0,1,2...
        while True:
            #if t < T:
            if t < T:
                #Take action At
                next_S,R,done, _,_ = env.step(A)
                next_S = next_S.tobytes()
                #observe R t+1 and S t+1
                states.append(next_S)
                rewards.append(R)
                
                #if St+1 is terminal then:
                if done:
                    #T <- t+1
                    T = t+1
                    total_re.append(sum(rewards))
                #else:
                else:
                    #Select and store an action At+1
                    next_A = e_greedy(Q,S,epsilon)
                    actions.append(next_A)
                    
            #tau <- t-n+1
            tau = t - n + 1
            #if tau >= 0:
            if tau >= 0:
                #update G
                G = sum([gamma**(i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T))])
                
                #if tau +n < T:
                if (tau+n) < T:
                    #update G
                    G += gamma**n * Q[states[tau + n]][actions[tau + n]]
                #update Q
                Q[states[tau]][actions[tau]] += step_size * (G - Q[states[tau]][actions[tau]])
                
            #if tau  = T-1 then break
            if tau == (T-1):
                steps.append(t)
                total_re.append(sum(rewards))
                break
                
            #state = next state, action = next action
            S = next_S
            A = next_A
            
            t+=1
            

    return steps,rewards


def argmax(arr: Sequence[float]) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values
    """
    max = arr.max()
    arr_idx = []
    for i in range(len(arr)):
        if arr[i] == max:
            arr_idx.append(i)
    
    if len(arr_idx) > 1:
        num = random.randint(0,(len(arr_idx)-1))
        idx = arr_idx[num]
    else:
        idx = arr_idx[0]

    return idx

def e_greedy(Q: defaultdict, state, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    if np.random.random() < epsilon:
        action = random.randint(0,num_actions-1)
    else:
        action = argmax(Q[state])


    return action

def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO (Done?)
    #initialize Q arbitrarily
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    steps = []
    tot_re = []
    #loop for each episode
    for x in trange(num_steps):
        rewards = []
        S,_ = env.reset()
        S = S.tobytes()
        #Choose A from S using policy derived from Q (e greedy)
        A = e_greedy(Q,S,epsilon)
        i = 0
        while True:
            i +=1

        #for each step in episode:
            #take action A and observe R and S'
            next_S,R,done, _,_ = env.step(A)
            next_S = next_S.tobytes()
            rewards.append(R)
            #Choose A' From S' using policy derived from Q (e-greedy)
            next_A = e_greedy(Q,next_S,epsilon)
            #update Q
            target = R + gamma*Q[next_S][next_A]
            Q[S][A] += step_size*(target-Q[S][A])
            
            #s = S'; A = A'
            S = next_S
            A = next_A
            #until S is terminal (check if episode is done); if done then break loop
            if done:
                steps.append(i)
                tot_re.append(sum(rewards))
                break
                                                              
    return steps,tot_re