import sys
from functools import reduce
import logging
import dill
import os
import time
from typing import List, Tuple, Dict

import yaml
import torch
import numpy as np
import pandas as pd

import gym_rock_paper_scissors
import gym_kuhn_poker

from regym.util.experiment_parsing import initialize_agents
from regym.util.experiment_parsing import initialize_training_schemes
from regym.util.experiment_parsing import filter_relevant_configurations
from regym.environments import generate_task, EnvType, Task
from regym.rl_loops.multiagent_loops import self_play_training
from regym.training_schemes import SelfPlayTrainingScheme, NaiveSelfPlay, FullHistorySelfPlay
from regym.rl_algorithms import build_PPO_Agent, AgentHook
from regym.rl_algorithms import load_population_from_path

from regym.game_theory import compute_winrate_matrix_metagame, compute_nash_averaging

from relative_population_performance_experiment import compute_relative_pop_performance_all_populations


def run_multiple_experiments(task, agents, sp_schemes,
                             experiment_config: Dict, seeds: List[int],
                             checkpoint_at_iterations: List[int],
                             base_path: str,
                             number_of_runs: int,
                             logger: logging.Logger):
    experiment_durations = list()
    for run_id in range(number_of_runs):
        logger.info(f'Starting run: {run_id}')
        start_time = time.time()
        single_experiment(task=task, selfplay_schemes=sp_schemes,
                          agents=agents,
                          checkpoint_at_iterations=checkpoint_at_iterations,
                          benchmarking_episodes=experiment_config['benchmarking_episodes'],
                          base_path=f'{base_path}/run-{run_id}', seed=seeds[run_id])
        experiment_durations.append(time.time() - start_time)
        logger.info(f'Finished run: {run_id}. Duration: {experiment_durations[-1]} (seconds)\n')
    logger.info('ALL DONE: total duration: {}'.format(sum(experiment_durations)))


def single_experiment(task: Task, agents: List, selfplay_schemes: List[SelfPlayTrainingScheme],
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
            # Self-play schemes like PSRO contain useful information
            dill.dump(sp_scheme, open(f'{path}/{sp_scheme.name}.pickle', 'wb'))

    logging.info('Computing relative performances')
    relative_performances_path = f'{base_path}/relative_performances/'
    if not os.path.exists(relative_performances_path): os.mkdir(relative_performances_path)
    compute_relative_pop_performance_all_populations(trained_agent_paths, task,
                                                     benchmarking_episodes,
                                                     base_path=relative_performances_path)

    logging.info('Loading all trained agents')
    joint_trained_population = reduce(lambda succ, path: succ + load_population_from_path(path),
                                      trained_agent_paths, [])
    logging.info('START winrate matrix computation of all trained policies')
    final_winrate_matrix = compute_winrate_matrix_metagame(joint_trained_population,
                                                           episodes_per_matchup=5,
                                                           task=task)
    logging.info('START Nash averaging computation of all trained policies')
    maxent_nash, nash_avg = compute_nash_averaging(final_winrate_matrix,
                                                   perform_logodds_transformation=True)
    logging.info('Experiment FINISHED!')
    dill.dump(final_winrate_matrix,
                open(f'{base_path}/final_winrate_matrix.pickle', 'wb'))
    dill.dump(maxent_nash,
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
    winrate_matrix = compute_winrate_matrix_metagame(population, task=task,
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
    dill.dump(checkpoints_winrate_submatrices,
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
         trajectories) = train_for_given_iterations(task, training_agent, self_play_scheme,
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


def train_for_given_iterations(task, training_agent, self_play_scheme,
                               menagerie, menagerie_path,
                               next_training_iterations, completed_iterations, logger):

    training_start = time.time()
    (menagerie, trained_agent,
     trajectories) = self_play_training(task=task, training_agent=training_agent, self_play_scheme=self_play_scheme,
                                        target_episodes=next_training_iterations, initial_episode=completed_iterations,
                                        menagerie=menagerie, menagerie_path=menagerie_path)
    training_duration = time.time() - training_start
    logger.info('Training between iterations [{}, {}]: {:.2} seconds'.format(
                completed_iterations, completed_iterations + next_training_iterations,
                training_duration))
    return menagerie, training_agent, trajectories


def save_trained_policy(trained_agent, save_path: str, logger):
    logger.info(f'Saving agent \'{trained_agent.name}\' in \'{save_path}\'')
    AgentHook(trained_agent.clone(training=False), save_path=save_path)


def initialize_experiment(experiment_config, agents_config, self_play_configs):
    env, env_type = experiment_config['environment']
    task = generate_task(env, EnvType(env_type))
    sp_schemes = initialize_training_schemes(self_play_configs, task)
    agents = initialize_agents(task, agents_config)

    seeds = list(map(int, experiment_config['seeds']))

    number_of_runs = experiment_config['number_of_runs']
    if len(seeds) < number_of_runs:
        print(f'Number of random seeds does not match "number of runs" config value. Genereting new seeds"')
        seeds = np.random.randint(0, 10000, number_of_runs).tolist()

    return task, sp_schemes, agents, seeds


def load_configs(config_file_path: str):
    all_configs = yaml.load(open(config_file_path), Loader=yaml.FullLoader)
    experiment_config = all_configs['experiment']
    self_play_configs = filter_relevant_configurations(experiment_config,
                                                       target_configs=all_configs['self_play_training_schemes'],
                                                       target_key='self_play_training_schemes')
    agents_config = filter_relevant_configurations(experiment_config,
                                                   target_configs=all_configs['agents'],
                                                   target_key='algorithms')
    return experiment_config, agents_config, self_play_configs


def save_used_configs(experiment_config: Dict, agents_config: Dict,
                      self_play_configs: Dict, save_path: str):
    all_relevant_config = {'experiment': experiment_config,
                           'agents': agents_config,
                           'self_play_training_schemes': self_play_configs}
    with open(f'{save_path}/experiment_parameters.yml', 'w') as outfile:
        yaml.dump(all_relevant_config, outfile, default_flow_style=False)



def setup_loggers(base_path: str):
    log_format = logging.Formatter(fmt='[%(asctime)s]:%(levelname)s:%(name)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    logging.basicConfig(format='[%(asctime)s]:%(levelname)s:%(name)s: %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        level=logging.INFO)
    file_handler = logging.FileHandler(filename=f'{base_path}/log.logs')
    file_handler.setFormatter(log_format)
    logging.getLogger().addHandler(file_handler)


if __name__ == '__main__':
    config_file_path = sys.argv[1]
    experiment_config, agents_config, self_play_configs = load_configs(config_file_path)
    base_path = experiment_config['experiment_id']
    if not os.path.exists(base_path): os.mkdir(base_path)
    number_of_runs = experiment_config['number_of_runs']

    task, sp_schemes, agents, seeds = initialize_experiment(experiment_config,
                                                            agents_config,
                                                            self_play_configs)
    experiment_config['seeds'] = seeds

    save_used_configs(experiment_config, agents_config,
                      self_play_configs, save_path=base_path)

    step_size = agents_config['ppo']['horizon'] * 1
    number_checkpoints = experiment_config['number_checkpoints']
    checkpoint_at_iterations = list(range(step_size,
                                          step_size * (number_checkpoints + 1),
                                          step_size))

    setup_loggers(base_path)
    logger = logging.getLogger('Nash experiment')

    run_multiple_experiments(task, agents, sp_schemes, experiment_config, seeds,
                             checkpoint_at_iterations, base_path,
                             number_of_runs, logger)
