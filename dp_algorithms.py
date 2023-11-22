import env

def get_max_v_argmax_a(world: env.Gridworld5x5, s, gamma):
    a_max = world.policy[s]
    max_exp_rew = 0
    for a in (env.Action.LEFT, env.Action.RIGHT, env.Action.UP, env.Action.DOWN):
        new_v = world.expected_return(world.values, world.state_space[s], a, gamma)
        if new_v > max_exp_rew:
            max_exp_rew = new_v
            a_max = a
            
    return max_exp_rew, a_max


# Doesn't quite give the correct output, but is decent after 25 iterations 
def iter_policy_eval(world: env.Gridworld5x5, gamma=0.9, theta=0.0001): # a removed as param
    delta = 0
    while delta < theta:
        delta = 0
        for st in range(len(world.state_space)):
            v = world.values[st]
            #print(f'passing a: {world.policy[st]}, st: {st}, pi[st]: {world.policy[st]}')
            world.values[st] = (world.expected_return(world.values, world.state_space[st], env.Action.LEFT, gamma) + 
                                   world.expected_return(world.values, world.state_space[st], env.Action.RIGHT, gamma) + 
                                   world.expected_return(world.values, world.state_space[st], env.Action.DOWN, gamma) + 
                                   world.expected_return(world.values, world.state_space[st], env.Action.UP, gamma)) / 4 # CHANGED
            delta = max(delta, abs(v-world.values[st]))
    return world.values
    
# Gives very close to correct output with 50 iterations
def value_iter(world: env.Gridworld5x5, gamma=0.9, theta=0.0001):
    delta = 0
    while delta < theta:
        delta = 0
        for st in range(len(world.state_space)):
            a_max = None
            v = world.values[st]
            max_exp_rew, a_max = get_max_v_argmax_a(world, st, gamma)
            world.values[st] = max_exp_rew
            world.policy[st] = a_max
            delta = max(delta, abs(v-world.values[st]))
    
    return world.values, world.policy
                        
def policy_improvement(world: env.Gridworld5x5, gamma):
    for st in range(len(world.state_space)):
        max_exp_rew, a_max = get_max_v_argmax_a(world, st, gamma)
        if world.values[st] != max_exp_rew:
            old_a = world.policy[s]
            world.policy[st] = a_max
            if old_a != world.policy[st]:
                policy_stable = False

# Gives very close to correct output with 1 iteration.
def policy_iter(world: env.Gridworld5x5, gamma=0.9, theta=0.001):
    # Initialization
    #world.values = list(np.zeros(len(self.state_space)))
    #for s in range(len(world.state_space)):
    #    world.policy[s] = random.choice([env.Action.LEFT, env.Action.RIGHT, env.Action.UP, env.Action.DOWN])
             
    while True:
        #policy_improvement(world, gamma)
        # Policy Evaluation
        #world.values = iter_policy_eval(world, gamma, theta)
        delta = 0
        while delta < theta:
            delta = 0
            for st in range(len(world.state_space)):
                v = world.values[st]
                #print(f'passing a: {world.policy[st]}, st: {st}, pi[st]: {world.policy[st]}')
                world.values[st] = world.expected_return(world.values, world.state_space[st], world.policy[st], gamma)
                delta = max(delta, abs(v-world.values[st]))

        # Policy Improvement
        policy_stable = True
        for st in range(len(world.state_space)):
            max_exp_rew, a_max = get_max_v_argmax_a(world, st, gamma)
            if world.values[st] != max_exp_rew:
                old_a = world.policy[st]
                world.policy[st] = a_max
                if old_a != world.policy[st]:
                    policy_stable = False

        if policy_stable:
            return world.values, world.policy
        else:
            continue