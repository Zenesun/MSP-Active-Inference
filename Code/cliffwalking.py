import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from pymdp import utils
from pymdp.maths import softmax, spm_log_single as log_stable
from pymdp.control import construct_policies


def plot_likelihood(matrix, xlabels=None, ylabels=None, title_str="Likelihood distribution (A)"):
    """
    Plots a 2-D likelihood matrix as a square heatmap with a controlled plot size.
    """

    if not np.isclose(matrix.sum(axis=0), 1.0).all():
        raise ValueError("Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)")

    if xlabels is None:
        xlabels = list(range(matrix.shape[1]))
    if ylabels is None:
        ylabels = list(range(matrix.shape[0]))

    # Adjust the figure size manually for more controlled sizing
    max_size = 8  # Maximum figure size (in inches)
    size = max(matrix.shape)  # Get the larger of (rows, cols)

    # Use a fixed figure size to prevent the plot from being too big
    # Scale the figure size based on the matrix shape but constrain the size
    fig_size = (min(size, max_size), min(size, max_size))

    plt.figure(figsize=fig_size)  # Set the figure size
    sns.heatmap(matrix, xticklabels=xlabels, yticklabels=ylabels, cmap='gray', cbar=False, vmin=0.0, vmax=1.0, square=True)
    plt.title(title_str)
    plt.show()

def plot_grid(grid_locations, num_x=4, num_y=12):
    """
    Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate 
    labeled with its linear index (its `state id`).
    """
    grid_heatmap = np.zeros((num_x, num_y))
    for linear_idx, location in enumerate(grid_locations):
      y, x = location
      grid_heatmap[y, x] = linear_idx
    sns.set(font_scale=1.5)
    sns.heatmap(grid_heatmap, annot=True, cbar = False, fmt='.0f', cmap='crest')
    plt.show()


def plot_point_on_grid(state_vector, grid_locations):
    """
    Plots the current location of the agent on the grid world
    """
    state_index = np.where(state_vector)[0][0]
    y, x = grid_locations[state_index]
    grid_heatmap = np.zeros((4,12))
    grid_heatmap[y,x] = 1.0
    sns.heatmap(grid_heatmap, cbar = False, fmt='.0f')
    plt.show()

def plot_beliefs(belief_dist, title_str="Belief Distribution"):
    """
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`.
    """

    if not np.isclose(belief_dist.sum(), 1.0):
        raise ValueError("Distribution not normalized! Please normalize")

    plt.figure(figsize=(12, 4))
    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title_str)
    plt.show()


env = gym.make("CliffWalking-v0", render_mode="rgb_array")
obs, info = env.reset()

def display_frame(frame, title_str=""):
  plt.figure(figsize=(10,4))
  plt.imshow(frame)
  plt.axis("off")
  plt.title(title_str)
  plt.show(block=True)


grid_locations = list(itertools.product(range(4), range(12)))

n_states = len(grid_locations)
n_observations = len(grid_locations)

actions = ["UP", "RIGHT", "DOWN", "LEFT"]


A = np.eye(n_observations, n_states)

plot_likelihood(A)

def create_B_matrix():
  B = np.zeros( (len(grid_locations), len(grid_locations), len(actions)) )

  for action_id, action_label in enumerate(actions):

    for curr_state, grid_location in enumerate(grid_locations):

      y, x = grid_location

      if (y == 3 and 1 <= x <= 10):
        x = 0
        y = 3

      if action_label == "UP":
        next_y = y - 1 if y > 0 else y 
        next_x = x
      elif action_label == "DOWN":
        next_y = y + 1 if y < 3 else y 
        next_x = x
      elif action_label == "LEFT":
        next_x = x - 1 if x > 0 else x 
        next_y = y
      elif action_label == "RIGHT":
        next_x = x + 1 if x < 11 else x 
        next_y = y

      new_location = (next_y, next_x)
      next_state = grid_locations.index(new_location)
      B[next_state, curr_state, action_id] = 1.0
  return B

B = create_B_matrix()


C = np.full(n_states, -1.0)

for idx in range(n_states):
    i, j = grid_locations[idx]
    distance = abs(3 - i) + abs(11 - j)
    c_value = -distance / 3
    
    if i == 3 and 1 <= j <= 10:
        c_value = -100
    C[idx] = c_value

C = softmax(C)


plot_beliefs(C)

D = utils.onehot(grid_locations.index((3,0)), n_states) # start the agent with the prior belief that it starts in location (3,0)

plot_beliefs(D)



""" Create an infer states function that implements the math we just discussed"""

def infer_states(observation_index, A, prior):

  """ Implement inference here -- NOTE: prior is already passed in, so you don't need to do anything with the B matrix. """
  """ This function has already been given P(s_t). The conditional expectation that creates "today's prior", using "yesterday's posterior", will happen *before calling* this function"""
  
  log_likelihood = log_stable(A[observation_index,:])

  log_prior = log_stable(prior)

  qs = softmax(log_likelihood + log_prior)
   
  return qs


""" define component functions for computing expected free energy """

def get_expected_states(B, qs_current, action):
  """ Compute the expected states one step into the future, given a particular action """
  qs_u = B[:,:,action].dot(qs_current)

  return qs_u

def get_expected_observations(A, qs_u):
  """ Compute the expected observations one step into the future, given a particular action """

  qo_u = A.dot(qs_u)

  return qo_u

def entropy(A):
  """ Compute the entropy of a set of conditional distributions, i.e. one entropy value per column """

  H_A = - (A * log_stable(A)).sum(axis=0)

  return H_A

def kl_divergence(qo_u, C):
  """ Compute the Kullback-Leibler divergence between two 1-D categorical distributions"""
  
  return (log_stable(qo_u) - log_stable(C)).dot(qo_u)


def calculate_G_policies(A, B, C, qs_current, policies):

  G = np.zeros(len(policies)) # initialize the vector of expected free energies, one per policy
  H_A = entropy(A)            # can calculate the entropy of the A matrix beforehand, since it'll be the same for all policies

  for policy_id, policy in enumerate(policies): # loop over policies - policy_id will be the linear index of the policy (0, 1, 2, ...) and `policy` will be a column vector where `policy[t,0]` indexes the action entailed by that policy at time `t`

    t_horizon = policy.shape[0] # temporal depth of the policy

    G_pi = 0.0 # initialize expected free energy for this policy

    for t in range(t_horizon): # loop over temporal depth of the policy

      action = policy[t,0] # action entailed by this particular policy, at time `t`

      # get the past predictive posterior - which is either your current posterior at the current time (not the policy time) or the predictive posterior entailed by this policy, one timstep ago (in policy time)
      if t == 0:
        qs_prev = qs_current 
      else:
        qs_prev = qs_pi_t
        
      qs_pi_t = get_expected_states(B, qs_prev, action) # expected states, under the action entailed by the policy at this particular time
      qo_pi_t = get_expected_observations(A, qs_pi_t)   # expected observations, under the action entailed by the policy at this particular time

      kld = kl_divergence(qo_pi_t, C) # Kullback-Leibler divergence between expected observations and the prior preferences C

      G_pi_t = H_A.dot(qs_pi_t) + kld # predicted uncertainty + predicted divergence, for this policy & timepoint

      G_pi += G_pi_t # accumulate the expected free energy for each timepoint into the overall EFE for the policy

    G[policy_id] += G_pi
  
  return G

def compute_prob_actions(actions, policies, Q_pi):
  P_u = np.zeros(len(actions)) # initialize the vector of probabilities of each action

  for policy_id, policy in enumerate(policies):
    #P_u[int(policy[0,0])] += Q_pi[policy_id] # get the marginal probability for the given action, entailed by this policy at the first timestep
    P_u[int(policy[0,0])] = max(P_u[int(policy[0,0])], Q_pi[policy_id])
  
  P_u = utils.norm_dist(P_u) # normalize the action probabilities
  
  return P_u

def active_inference_with_planning(A, B, C, D, n_actions, env, policy_len=2, T=5):
    """ Initialize prior, first observation, and policies """
    prior = D
    obs, _ = env.reset() 
    cumulative_reward = 0
    
    policies = construct_policies([n_states], [n_actions], policy_len=policy_len)


    for t in range(T):
        print(f'\nTime {t}: Agent observes itself in location: {obs} (Grid: {grid_locations[obs]})')

        if obs == 47:
            print(f"Agent reached the target at {grid_locations[obs]}! Stopping.")
            return qs_current, cumulative_reward  # Stop and return

        # Inference (use the integer observation directly)
        qs_current = infer_states(obs, A, prior)
        #plot_beliefs(qs_current, title_str=f"Beliefs about location at time {t}")

        # Calculate expected free energy
        G = calculate_G_policies(A, B, C, qs_current, policies)
        Q_pi = softmax(-G)
        P_u = compute_prob_actions(actions, policies, Q_pi)
        chosen_action = np.argmax(P_u)
        #print(P_u, chosen_action)
        #chosen_action = utils.sample(P_u)
        

        # Update prior
        prior = B[:, :, chosen_action].dot(qs_current)

        next_obs, reward, terminated, truncated, info = env.step(chosen_action)
        cumulative_reward += reward
        
        
        print(f"Chosen action: {actions[chosen_action]} | Reward: {reward}")
        frame = env.render()  # returns an RGB array since env was created with render_mode="rgb_array"
        display_frame(frame, title_str=f"Time {t + 1}: Action {actions[chosen_action]}")
        
        if terminated or truncated:
            print(f"Episode terminated! Total reward: {cumulative_reward}")
            return qs_current, cumulative_reward 
        obs = next_obs

    print(f"\nFinal cumulative reward after {T} timesteps: {cumulative_reward}")
    return qs_current, cumulative_reward

n_actions = len(actions)

frame = env.render()
display_frame(frame, title_str=f"Time {0}")

qs_final, reward = active_inference_with_planning(A, B, C, D, n_actions, env, policy_len = 7, T = 20)
