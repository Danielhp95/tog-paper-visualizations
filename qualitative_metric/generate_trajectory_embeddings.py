import copy
import os
import sys
import pickle

import numpy as np
import pandas as pd
import gym_rock_paper_scissors
from tqdm import tqdm
from sklearn.manifold import TSNE

from regym.environments import generate_task, EnvType
from regym.networks.preprocessing import flatten_and_turn_into_single_element_batch
from regym.rl_algorithms import rockAgent, paperAgent, scissorsAgent, randomAgent, build_PPO_Agent
from regym.rl_loops.multiagent_loops import simultaneous_action_rl_loop
from regym.rl_loops.multiagent_loops.self_play_loop import self_play_training
from regym.training_schemes import NaiveSelfPlay


def RPSTask():
    return generate_task('RockPaperScissors-v0', EnvType.MULTIAGENT_SIMULTANEOUS_ACTION)


def collect_basis_trajectories_for(env, agents, fixed_opponents, nbr_episodes_matchup):
    trajs = {'agent':[],
             'opponent':[],
             'trajectory':[]
            }
    progress_bar = tqdm(range(len(fixed_opponents)))
    for e in progress_bar:
        fixed_opponent = fixed_opponents[e]
        for agent in agents:
            basis_trajectories = simulate(env, agent, fixed_opponent, episodes=nbr_episodes_matchup, training=False)
            process_trajectories(basis_trajectories, trajs, fixed_opponent, agent)
        progress_bar.set_description(f'Collecting trajectories: {agent.name} against {fixed_opponent.name}.')
    return trajs


def process_trajectories(trajectories, trajs_dict, fixed_opponent, agent):
    for t in trajectories:
        trajs_dict['agent'].append(fixed_opponent.name)
        trajs_dict['opponent'].append(agent.name)
        trajs_dict['trajectory'].append(t)


def merge_basis_and_trained_trajectories(basis_trajectories, training_trajectories):
    for k in training_trajectories.keys():
        for idx in range(len(training_trajectories[k])):
            basis_trajectories[k].append( training_trajectories[k][idx])
    return basis_trajectories


def simulate(env, agent, fixed_opponent, episodes, training):
    agent_vector = [agent, fixed_opponent]
    trajectories = list()
    mode = 'Training' if training else 'Inference'
    progress_bar = tqdm(range(episodes))
    for e in progress_bar:
        trajectory = simultaneous_action_rl_loop.run_episode(env, agent_vector, training=training)
        trajectories.append(trajectory)
        progress_bar.set_description(f'{mode} {agent.name} against {fixed_opponent.name}')
    return trajectories


def compute_basis_trajectories(task):
    num_basis_trajectories_per_action = 1000
    trajectories = collect_basis_trajectories_for(
        task.env,
        [randomAgent],  # Fixed agent
        [rockAgent, paperAgent, scissorsAgent],  # Agents that fixed agent will play against
        nbr_episodes_matchup=num_basis_trajectories_per_action
    )
    return trajectories


def compute_sp_training_trajectories(task, agent, sp_scheme):
    trajs = {'agent':[],
             'opponent':[],
             'trajectory':[]
            }

    #target_episodes = 2000
    #menagerie, agent, trajectories = self_play_training(
    #    task = task,
    #    training_agent=agent,
    #    self_play_scheme=sp_scheme,
    #    show_progress=True,
    #    target_episodes=target_episodes
    #)

    #pickle.dump(trajectories, open('trained_agent_traj.pickle', 'wb'))

    trajectories = pickle.load(open('trained_agent_traj.pickle', 'rb'))

    process_trajectories(trajectories, trajs, agent, agent)
    return trajs


def initialize_agent(task):
    # Config defined in paper
    config = dict()
    config['discount'] = 0.99
    config['use_gae'] = True
    config['use_cuda'] = False
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 10
    config['mini_batch_size'] = 64
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['horizon'] = 2048
    config['phi_arch'] = 'MLP'
    config['actor_arch'] = 'None'
    config['critic_arch'] = 'None'
    config['state_preprocess'] = flatten_and_turn_into_single_element_batch

    agent = build_PPO_Agent(task, config, 'ppo_agent')
    return agent


def generate_t_sne_embedding(traj_actions, trajs):
    traj_actions = np.asarray(traj_actions)
    x_actions = copy.deepcopy(traj_actions)
    y_agents = np.asarray( copy.deepcopy(trajs['agent']) )

    X_sample_flat = np.reshape(x_actions, [x_actions.shape[0], -1])
    perplexities = [5, 50, 100,200,300,500]
    embeddings = []
    for perplexity in perplexities:
        embeddings.append(
            TSNE(n_components=2,  # We want only 2 dims, to have a plot!
                 init='pca',
                 random_state=17,
                 verbose=2,
                 learning_rate=300,
                 n_iter=1000,
                 perplexity=perplexity
                 ).fit_transform(X_sample_flat)
        )

    print(f'Moving to save embeddings ({len(embeddings)} in total)')
    embeddings_dir = 't_sne_embeddings'
    os.makedirs(embeddings_dir, exist_ok=True)
    for e, perplexity in zip(embeddings, perplexities):
        pickle.dump(e, open(f'{embeddings_dir}/embedding_perplexity_{perplexity}.pickle', 'wb'))

    return embeddings


def main():
    task = generate_task('RockPaperScissors-v0', EnvType.MULTIAGENT_SIMULTANEOUS_ACTION)
    print('Initializing agent')
    agent = initialize_agent(task)

    print('Computing SP-induced trajectories')
    training_trajectories = compute_sp_training_trajectories(
        task=task,
        agent=agent,
        sp_scheme=NaiveSelfPlay,
    )

    print('Computing basis trajectories')
    basis_trajectories = compute_basis_trajectories(task)

    print('Merging trajectories')
    import ipdb; ipdb.set_trace()
    all_trajectories = merge_basis_and_trained_trajectories(basis_trajectories, training_trajectories)
    print('Number basis trajectories:', len(basis_trajectories['trajectory']))

    # Compute trajectories from training agent
    ts = copy.deepcopy(all_trajectories['trajectory'])
    print(f'Number trajectories: {len(ts)} // Steps per trajectories: {len(ts[0])}')

    actions = [
        # We get the last observation of the first agent,
        # This contains the last joint action by both agents.
        [step.observation[0][-1]
         for idx, step in enumerate(t) if idx < 10 and idx > 0
        ]
        for t in ts
    ]

    embeddings = generate_t_sne_embedding(actions, all_trajectories)


if __name__ == "__main__":
    main()
