from typing import List, Tuple, Dict
import sys
from functools import reduce
from copy import deepcopy
import logging
import dill
import os
import time
import argparse
import yaml

import multiprocessing_on_dill as multiprocessing
from multiprocessing import Process

from concurrent import futures
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
import pandas as pd

import gym_rock_paper_scissors
import gym_connect4
import gym_kuhn_poker

from regym.logging_server import create_logging_server_process, initialize_logger, SERVER_SHUTDOWN_MESSAGE

from regym.util.experiment_parsing import initialize_agents
from regym.util.experiment_parsing import initialize_training_schemes
from regym.util.experiment_parsing import filter_relevant_configurations

from regym.environments import generate_task, EnvType, Task

from regym.rl_loops.multiagent_loops import self_play_training
from regym.training_schemes import SelfPlayTrainingScheme

from regym.rl_algorithms import build_PPO_Agent
from regym.rl_algorithms import load_population_from_path

from regym.game_theory import compute_winrate_matrix_metagame, compute_nash_averaging

from relative_population_performance_experiment import compute_relative_pop_performance_all_populations


def run_multiple_experiments(task, agents, sp_schemes,
                             experiment_config: Dict,
                             checkpoint_at_iterations: List[int],
                             base_path: str,
                             run_ids: List[int],
                             logger: logging.Logger):
    experiment_processes = []
    start_time = time.time()
    for run_id in run_ids:
        logger.info(f'Run {run_id}: STARTED')

        process_agents = deepcopy(agents)
        for agent in process_agents:
            agent.algorithm.model.share_memory()

        experiment_process = Process(
            target=single_experiment,
            kwargs=dict(
                task=task,
                sp_schemes=sp_schemes,
                agents=process_agents,
                checkpoint_at_iterations=checkpoint_at_iterations,
                benchmarking_episodes=experiment_config['benchmarking_episodes'],
                base_path=f'{base_path}/run-{run_id}',
                run_id=run_id
                )
        )
        experiment_process.start()
        experiment_processes.append(experiment_process)

    for experiment_process in experiment_processes:
        experiment_process.join()

    total_experiments_time = time.time() - start_time
    logger.info('ALL DONE: total duration: {}'.format(total_experiments_time))


def single_experiment(task: Task, agents: List, sp_schemes: List[SelfPlayTrainingScheme],
                      checkpoint_at_iterations: List[int], base_path: str,
                      benchmarking_episodes: int, run_id: int):
    base_paths = [
        f'{base_path}/{sp_scheme.name}-{agent.name}'
        for sp_scheme in sp_schemes
        for agent in agents
    ]

    num_processes = len(sp_schemes) * len(agents)
    with ProcessPoolExecutor(max_workers=num_processes) as ex:
        training_futures = [
            ex.submit(
                train_and_evaluate,
                task=deepcopy(task),
                self_play_scheme=deepcopy(sp_scheme),
                training_agent=agent.clone(training=True),
                checkpoint_at_iterations=checkpoint_at_iterations,
                benchmarking_episodes=benchmarking_episodes,
                base_path=f'{base_path}/{sp_scheme.name}-{agent.name}',
                run_id=run_id
            )
            for sp_scheme in sp_schemes
            for agent in agents
        ]

        futures.wait(training_futures)

    relative_performances_path = f'{base_path}/relative_performances/'
    if not os.path.exists(relative_performances_path): os.mkdir(relative_performances_path)
    logging.info('Computing relative performances')

    relative_pop_performance_start_time = time.time()
    compute_relative_pop_performance_all_populations(
        base_paths,
        task,
        benchmarking_episodes,
        base_path=relative_performances_path
    )
    relative_pop_performance_total_time = time.time() - relative_pop_performance_start_time
    logging.info('Computing relative performances took: {:.2}'.format(relative_pop_performance_total_time))


    # logging.info('Loading all trained agents')
    # joint_trained_population = reduce(lambda succ, path: succ + load_population_from_path(path),
    #                                   base_paths, [])
    # logging.info('START winrate matrix computation of all trained policies')
    # final_winrate_matrix = compute_winrate_matrix_metagame(joint_trained_population,
    #                                                        episodes_per_matchup=5,
    #                                                        task=task)
    # logging.info('START Nash averaging computation of all trained policies')
    # maxent_nash, nash_avg = compute_nash_averaging(final_winrate_matrix,
    #                                                perform_logodds_transformation=True)
    # logging.info('Experiment FINISHED!')
    #             open(f'{base_path}/final_winrate_matrix.pickle', 'wb'))
    # dill.dump(maxent_nash,
    #             open(f'{base_path}/final_maxent_nash.pickle', 'wb'))


def train_and_evaluate(task: Task, training_agent, self_play_scheme: SelfPlayTrainingScheme,
                       checkpoint_at_iterations: List[int], base_path: str,
                       benchmarking_episodes: int, run_id: int):
    print(base_path)
    logger = initialize_logger(
        name=f'Run: {run_id}. Experiment: Task: {task.name}. SP: {self_play_scheme.name}. Agent: {training_agent.name}'
    )

    population = training_phase(task, training_agent, self_play_scheme,
                                checkpoint_at_iterations, base_path, run_id)
    logger.info('FINISHED training! Moving to saving')

    save_path = f'{base_path}/population'
    logger.info(f'FINISHED saving! Saved {len(population)} agents in {save_path}')


    winrate_submatrices, evolution_maxent_nash_and_nash_averaging = compute_optimality_metrics(
        population,
        task,
        benchmarking_episodes,
        logger)

    save_results(winrate_submatrices,
                 evolution_maxent_nash_and_nash_averaging,
                 checkpoint_at_iterations,
                 self_play_scheme,
                 save_path=f'{base_path}/results')
    logger.info('FINISHED saving')


def compute_optimality_metrics(population, task, benchmarking_episodes, logger):
    logger.info('Computing winrate matrix')
    winrate_matrix_start_time = time.time()
    winrate_matrix = compute_winrate_matrix_metagame(population, task=task,
                                                     episodes_per_matchup=benchmarking_episodes)
    winrate_submatrices = [winrate_matrix[:i, :i] for i in range(1, len(winrate_matrix) + 1)]
    winrate_matrix_total_time = time.time() - start_time
    logger.info('Computing winrate matrix took: {:.2} seconds'.format(winrate_matrix_total_time))

    nash_averaging_start_time = time.time()
    logger.info('Computing nash averagings for all submatrices')
    evolution_maxent_nash_and_nash_averaging = [compute_nash_averaging(m, perform_logodds_transformation=True)
                                                for m in winrate_submatrices]
    nash_averaging_total_time = time.time() - nash_averaging_start_time
    logger.info('Computing nash averagings for all submatrices too: {:.2} seconds'.format(nash_averaging_total_time))
    return winrate_submatrices, evolution_maxent_nash_and_nash_averaging


def save_results(winrate_submatrices: List[np.ndarray],
                 evolution_maxent_nash_and_nash_averaging: List[Tuple[np.ndarray]],
                 checkpoint_at_iterations: List[int],
                 sp_scheme: SelfPlayTrainingScheme,
                 save_path: str):
    if not os.path.exists(save_path): os.makedirs(save_path)
    save_winrate_matrices(winrate_submatrices, checkpoint_at_iterations, save_path)
    save_evolution_maxent_nash_and_nash_averaging(evolution_maxent_nash_and_nash_averaging,
                                                  checkpoint_at_iterations, save_path)
    # Self-play schemes like PSRO contain useful information
    dill.dump(sp_scheme, open(f'{save_path}/{sp_scheme.name}.pickle', 'wb'))


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
                   checkpoint_at_iterations: List[int], base_path: str, run_id: int):
    """
    :param task: Task on which agents will be trained
    :param training_agent: agent representation + training algorithm which will be trained in this process
    :param self_play_scheme: self play scheme used to meta train the param training_agent.
    :param checkpoint_at_iterations: array containing the episodes at which the agents will be cloned for benchmarking against one another
    :param agent_queue: queue shared among processes to submit agents that will be benchmarked
    :param process_name: String name identifier
    :param base_path: Base directory from where subdirectories will be accessed to reach menageries, save episodic rewards and save checkpoints of agents.
    """

    logger = initialize_logger(f'Run: {run_id}. TRAINING: Task: {task.name}. SP: {self_play_scheme.name}. Agent: {training_agent.name}')
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

        save_path = f'{trained_policy_save_directory}/{target_iteration}_iterations.pt'
        logger.info(f'Saving agent \'{trained_agent.name}\' in \'{save_path}\'')
        torch.save(trained_agent, save_path)

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


# TODO: consider not using this
def setup_loggers(base_path: str):
    log_format = logging.Formatter(fmt='[%(asctime)s]:%(levelname)s:%(name)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    logging.basicConfig(format='[%(asctime)s]:%(levelname)s:%(name)s: %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        level=logging.INFO)
    file_handler = logging.FileHandler(filename=f'{base_path}/log.logs')
    file_handler.setFormatter(log_format)
    logging.getLogger().addHandler(file_handler)


if __name__ == '__main__':
    import torch.multiprocessing as torch_mp
    torch_mp.set_start_method('forkserver')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path leading to YAML config file')
    parser.add_argument('--run_id', default=None,
                        help='Identifier for the single_run that will be run. Ignoring number of runs in config file.')
    args = parser.parse_args()

    experiment_config, agents_config, self_play_configs = load_configs(args.config)
    if not os.path.exists(experiment_config['experiment_id']):
        os.mkdir(experiment_config['experiment_id'])

    task, sp_schemes, agents, seeds = initialize_experiment(
        experiment_config,
        agents_config,
        self_play_configs)
    experiment_config['seeds'] = seeds

    save_used_configs(
        experiment_config,
        agents_config,
        self_play_configs,
        save_path=experiment_config['experiment_id'])

    step_size = agents_config['ppo']['horizon'] * experiment_config['agent_updates_per_checkpoint']
    number_checkpoints = experiment_config['number_checkpoints']
    checkpoint_at_iterations = list(range(step_size,
                                          step_size * (number_checkpoints + 1),
                                          step_size))

    logging_server = create_logging_server_process(
        log_path=f"{experiment_config['experiment_id']}/logs.logs")
    logger = initialize_logger(name='Nash experiment')

    if args.run_id: run_ids = [int(args.run_id)]
    else: run_ids = list(range(experiment_config['number_of_runs']))
    logger.info(f'Running runs: {run_ids}')

    run_multiple_experiments(
        task,
        agents,
        sp_schemes,
        experiment_config,
        checkpoint_at_iterations,
        experiment_config['experiment_id'],
        run_ids,
        logger)
