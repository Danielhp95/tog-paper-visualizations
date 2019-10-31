import os
import time
from typing import List, Tuple
import logging 
import pickle

import torch
import numpy as np
import pandas as pd

from regym.environments import generate_task
from regym.environments.task import Task
from regym.rl_loops.multiagent_loops.simultaneous_action_rl_loop import self_play_training
from regym.training_schemes import SelfPlayTrainingScheme, NaiveSelfPlay, FullHistorySelfPlay
from regym.rl_algorithms import build_PPO_Agent, AgentHook

from regym.game_theory import compute_winrate_matrix_metagame, compute_nash_averaging

def experiment(task: Task, training_agent, self_play_scheme: SelfPlayTrainingScheme,
                       checkpoint_at_iterations: List[int], base_path: str, seed: int):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f'Experiment: Task: {task.name}. SP: {self_play_scheme.name}. Agent: {training_agent.name}')
    logger.setLevel(logging.INFO)

    # Because we care for reproducibility!
    np.random.seed(seed)
    torch.manual_seed(seed)

    population = training_phase(task, training_agent, self_play_scheme, checkpoint_at_iterations, base_path, seed)
    # TODO: refactor into single function which returns winrate_submatrices,
    #       and evolution_maxent_nash_and_nash_averaging
    winrate_matrix = compute_winrate_matrix_metagame(population, env=task.env,
                                                     episodes_per_matchup=10)
    logger.info('Computing winrate matrix')
    winrate_submatrices = [np.array([[0.5]])] + [winrate_matrix[:i,:i] for i in range(2, len(winrate_matrix) + 1)] # TODO explain why we are appending
    logger.info('Computing nash averagings for all submatrices')
    evolution_maxent_nash_and_nash_averaging = [(np.array([1.]), np.array([0.]))] + \
                                               [compute_nash_averaging(m, perform_logodds_transformation=True)
                                                for m in winrate_submatrices[1:]]
    logger.info('FINISHED computation! Moving to saving')
    save_results(winrate_submatrices,
                 evolution_maxent_nash_and_nash_averaging,
                 checkpoint_at_iterations,
                 save_path=f'{base_path}/results')
    logger.info('FINISHED saving')
    logger.info('DONE')


def save_results(winrate_submatrices: List[np.ndarray],
                 evolution_maxent_nash_and_nash_averaging: List[Tuple[np.ndarray]],
                 checkpoint_at_iterations: List[int],
                 save_path: str):
    if not os.path.exists(save_path): os.mkdir(save_path)
    save_winrate_matrices(winrate_submatrices, checkpoint_at_iterations, save_path)
    save_evolution_maxent_nash_and_nash_averaging(evolution_maxent_nash_and_nash_averaging,
                                                  checkpoint_at_iterations, save_path)


def save_winrate_matrices(winrate_submatrices, checkpoint_at_iterations, save_path):
    checkpoints_winrate_submatrices = {checkpoint: m
                                       for checkpoint, m in 
                                       zip(checkpoint_at_iterations, winrate_submatrices)}
    pickle.dump(checkpoints_winrate_submatrices,
                open(f'{save_path}/winrate_matrices.pickle', 'wb'))


def save_evolution_maxent_nash_and_nash_averaging(evolution_maxent_nash_and_nash_averaging, checkpoint_at_iterations, save_path):
    maxent_nash_list, nash_averaging_list = zip(*evolution_maxent_nash_and_nash_averaging)
    nash_progression_df = pd.DataFrame(maxent_nash_list, index=checkpoint_at_iterations,
                                       columns=list(range(len(checkpoint_at_iterations))))
    nash_progression_df.to_csv(path_or_buf=f'{save_path}/evolution_maxent_nash.csv')


def training_phase(task: Task, training_agent, self_play_scheme: SelfPlayTrainingScheme,
               checkpoint_at_iterations: List[int], base_path: str, seed: int):
    """
    :param task: Task on which agents will be trained
    :param training_agent: agent representation + training algorithm which will be trained in this process
    :param self_play_scheme: self play scheme used to meta train the param training_agent.
    :param checkpoint_at_iterations: array containing the episodes at which the agents will be cloned for benchmarking against one another
    :param agent_queue: queue shared among processes to submit agents that will be benchmarked
    :param process_name: String name identifier
    :param base_path: Base directory from where subdirectories will be accessed to reach menageries, save episodic rewards and save checkpoints of agents.
    """
    
    logger = logging.getLogger(f'TRAINING: Task: {task.name}. SP: {self_play_scheme.name}. Agent: {training_agent.name}')
    logger.info('Started')

    menagerie, menagerie_path = [], f'{base_path}/menagerie'
    agents_to_benchmark = [] # Come up with better name

    if not os.path.exists(base_path):
        os.mkdir(base_path)
        os.mkdir(menagerie_path)

    completed_iterations, start_time = 0, time.time()

    trained_policy_save_directory = base_path

    for target_iteration in sorted(checkpoint_at_iterations):
        next_training_iterations = target_iteration - completed_iterations
        (menagerie, trained_agent,
         trajectories) = train_for_given_iterations(task.env, training_agent, self_play_scheme,
                                                    menagerie, menagerie_path,
                                                    next_training_iterations, completed_iterations, logger)
        del trajectories # we are not using them here
        completed_iterations += next_training_iterations
        save_trained_policy(trained_agent,
                            save_path=f'{trained_policy_save_directory}/{target_iteration}_iterations.pt',
                            logger=logger)
        
        agents_to_benchmark += [trained_agent.clone()]
        training_agent = trained_agent # Updating

    logger.info('FINISHED training. Total duration: {} seconds'.format(time.time() - start_time))
    return agents_to_benchmark


def train_for_given_iterations(env, training_agent, self_play_scheme,
                               menagerie, menagerie_path,
                               next_training_iterations, completed_iterations, logger):

    training_start = time.time()
    (menagerie, trained_agent,
     trajectories) = self_play_training(env=env, training_agent=training_agent, self_play_scheme=self_play_scheme,
                                        target_episodes=next_training_iterations, iteration=completed_iterations,
                                        menagerie=menagerie, menagerie_path=menagerie_path)
    training_duration = time.time() - training_start
    logger.info('Training between iterations [{}, {}]: {} seconds'.format(
                completed_iterations, completed_iterations + next_training_iterations,
                training_duration))
    return menagerie, training_agent, trajectories


def save_trained_policy(trained_agent, save_path: str, logger):
    logger.info(f'Saving agent \'{training_agent.name}\' in \'{save_path}\'')
    hooked_agent = AgentHook(trained_agent.clone(training=False), save_path=save_path)


def ppo_config_dict():
    config = dict()
    config['discount'] = 0.99
    config['use_gae'] = False
    config['use_cuda'] = False
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 10
    config['mini_batch_size'] = 256
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['horizon'] = 1024
    config['phi_arch'] = 'MLP'
    config['actor_arch'] = 'None'
    config['critic_arch'] = 'None'
    return config


if __name__ == '__main__':
    task = generate_task('RockPaperScissors-v0')
    sp_scheme = FullHistorySelfPlay
    checkpoint_at_iterations = [10, 20, 30]
    base_path= 'experiment-test'
    seed = 69
    training_agent = build_PPO_Agent(task, ppo_config_dict(), agent_name='Test')
    experiment(task=task, self_play_scheme=sp_scheme,
               training_agent=training_agent,
               checkpoint_at_iterations=checkpoint_at_iterations,
               base_path=base_path, seed=seed)
