from functools import reduce
import logging
import pickle
import os
import time
from typing import List, Tuple, Dict

import yaml
import torch
import numpy as np
import pandas as pd

from regym.util.experiment_parsing import initialize_agents
from regym.util.experiment_parsing import initialize_training_schemes
from regym.util.experiment_parsing import filter_relevant_agent_configurations
from regym.environments import generate_task
from regym.environments.task import Task
from regym.rl_loops.multiagent_loops.simultaneous_action_rl_loop import self_play_training
from regym.training_schemes import SelfPlayTrainingScheme, NaiveSelfPlay, FullHistorySelfPlay
from regym.rl_algorithms import build_PPO_Agent, AgentHook
from regym.rl_algorithms import load_population_from_path

from regym.game_theory import compute_winrate_matrix_metagame, compute_nash_averaging


def experiment(task: Task, agents: List, selfplay_schemes: List[SelfPlayTrainingScheme],
               checkpoint_at_iterations: List[int], base_path: str, seed: int,
               benchmarking_episodes: int):
    trained_agent_paths = []
    for sp_scheme in sp_schemes:
        for agent in agents:
            training_agent = agent.clone(training=True)
            path = f'{base_path}/{sp_scheme.name}-{agent.name}'
            trained_agent_paths += [path]
            train_and_evaluate(task=task, self_play_scheme=sp_scheme,
                               training_agent=training_agent,
                               checkpoint_at_iterations=checkpoint_at_iterations,
                               benchmarking_episodes=experiment_config['benchmarking_episodes'],
                               base_path=path, seed=seed)
    logging.info('Loading all trained agents')
    joint_trained_population = reduce(lambda succ, path: succ + load_population_from_path(path),
                                      trained_agent_paths, [])
    logging.info('START winrate matrix computation of all trained policies')
    final_winrate_matrix = compute_winrate_matrix_metagame(joint_trained_population,
                                                           episodes_per_matchup=5,
                                                           env=task.env)
    logging.info('START Nash averaging computation of all trained policies')
    maxent_nash, nash_avg = compute_nash_averaging(final_winrate_matrix,
                                                   perform_logodds_transformation=True)
    logging.info('Experiment FINISHED!')
    pickle.dump(final_winrate_matrix,
                open(f'{base_path}/final_winrate_matrix.pickle', 'wb'))
    pickle.dump(maxent_nash,
                open(f'{base_path}/final_maxent_nash.pickle', 'wb'))


def train_and_evaluate(task: Task, training_agent, self_play_scheme: SelfPlayTrainingScheme,
                       checkpoint_at_iterations: List[int], base_path: str, seed: int,
                       benchmarking_episodes: int):
    logger = logging.getLogger(f'Experiment: Task: {task.name}. SP: {self_play_scheme.name}. Agent: {training_agent.name}')

    np.random.seed(seed)
    torch.manual_seed(seed)

    population = training_phase(task, training_agent, self_play_scheme,
                                checkpoint_at_iterations, base_path)
    logger.info('FINISHED training! Moving to saving')
    winrate_submatrices, evolution_maxent_nash_and_nash_averaging = compute_optimality_metrics(population, task,
                                                                                               benchmarking_episodes,
                                                                                               logger)
    save_results(winrate_submatrices,
                 evolution_maxent_nash_and_nash_averaging,
                 checkpoint_at_iterations,
                 save_path=f'{base_path}/results')
    logger.info('FINISHED saving')


def compute_optimality_metrics(population, task, benchmarking_episodes, logger):
    logger.info('Computing winrate matrix')
    winrate_matrix = compute_winrate_matrix_metagame(population, env=task.env,
                                                     episodes_per_matchup=benchmarking_episodes)
    winrate_submatrices = [winrate_matrix[:i, :i] for i in range(1, len(winrate_matrix) + 1)]
    logger.info('Computing nash averagings for all submatrices')
    evolution_maxent_nash_and_nash_averaging = [compute_nash_averaging(m, perform_logodds_transformation=True)
                                                for m in winrate_submatrices]
    return winrate_submatrices, evolution_maxent_nash_and_nash_averaging


def save_results(winrate_submatrices: List[np.ndarray],
                 evolution_maxent_nash_and_nash_averaging: List[Tuple[np.ndarray]],
                 checkpoint_at_iterations: List[int],
                 save_path: str):
    if not os.path.exists(save_path): os.makedirs(save_path)
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
                   checkpoint_at_iterations: List[int], base_path: str):
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
        os.makedirs(base_path)
        os.mkdir(menagerie_path)

    completed_iterations, start_time = 0, time.time()

    trained_policy_save_directory = base_path
    final_iteration = max(checkpoint_at_iterations)

    for target_iteration in sorted(checkpoint_at_iterations):
        next_training_iterations = target_iteration - completed_iterations
        (menagerie, trained_agent,
         trajectories) = train_for_given_iterations(task.env, training_agent, self_play_scheme,
                                                    menagerie, menagerie_path,
                                                    next_training_iterations, completed_iterations, logger)
        logger.info('Training completion: {}%'.format(100 * target_iteration / final_iteration)) 
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
    logger.info('Training between iterations [{}, {}]: {:.2} seconds'.format(
                completed_iterations, completed_iterations + next_training_iterations,
                training_duration))
    return menagerie, training_agent, trajectories


def save_trained_policy(trained_agent, save_path: str, logger):
    logger.info(f'Saving agent \'{trained_agent.name}\' in \'{save_path}\'')
    AgentHook(trained_agent.clone(training=False), save_path=save_path)


def initialize_experiment(experiment_config, agents_config):
    task = generate_task(experiment_config['environment'])
    sp_schemes = initialize_training_schemes(experiment_config['self_play_training_schemes'])
    agents = initialize_agents(experiment_config['environment'], agents_config)

    seed = experiment_config['seed']
    return task, sp_schemes, agents, seed


def load_configs(config_file_path: str):
    all_configs = yaml.load(open(config_file_path), Loader=yaml.FullLoader)
    experiment_config = all_configs['experiment']
    agents_config = filter_relevant_agent_configurations(experiment_config,
                                                         all_configs['agents'])
    return experiment_config, agents_config


def save_used_configs(experiment_config: Dict, agents_config: Dict, save_path: str):
    all_relevant_config = {'experiment': experiment_config, 'agents': agents_config}
    with open(f'{save_path}/experiment_parameters.yml', 'w') as outfile:
        yaml.dump(all_relevant_config, outfile, default_flow_style=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Nash experiment')

    config_file_path = './config.yaml'
    experiment_config, agents_config = load_configs(config_file_path)
    base_path = experiment_config['experiment_id']
    if not os.path.exists(base_path): os.mkdir(base_path)
    number_of_runs = experiment_config['number_of_runs']
    save_used_configs(experiment_config, agents_config, save_path=base_path)

    task, sp_schemes, agents, seed = initialize_experiment(experiment_config, agents_config)
    checkpoint_at_iterations = list(range(0, 20, 10))

    for run_id in range(number_of_runs):
        logger.info(f'Starting run: {run_id}')
        start_time = time.time()
        experiment(task=task, selfplay_schemes=sp_schemes,
                   agents=agents,
                   checkpoint_at_iterations=checkpoint_at_iterations,
                   benchmarking_episodes=experiment_config['benchmarking_episodes'],
                   base_path=f'{base_path}/run-{run_id}', seed=seed)
        experiment_duration = time.time() - start_time
        logger.info(f'Finished run: {run_id}. Duration: {experiment_duration} (seconds)\n')
