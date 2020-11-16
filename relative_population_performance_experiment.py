from time import time
import os
import argparse

import dill
import numpy as np

from regym.rl_algorithms.agent_hook import load_population_from_path
from regym.game_theory.compute_winrate_matrix_metagame import evolution_relative_population_performance
from regym.networks.preprocessing import batch_vector_observation


# TODO: this function could be really useful in the future
# find a way of refactoring it into regym
def compute_relative_pop_performance_all_populations(paths, task, 
                                                    benchmarking_episodes,
                                                    base_path):
    self_play_names = list(map(lambda s: s.split('/')[-1], paths))
    compare_iteration_number = lambda e: int(e.split('/')[-1].split('_')[0])
    populations = [load_population_from_path(path,
                                             sort_fn=compare_iteration_number)
                   for path in paths]
    for population in populations:
        for agent in population:
            agent.state_preprocessing = batch_vector_observation

    for i, (p1, sp_name_1) in enumerate(zip(populations, self_play_names)):
        for p2, sp_name_2 in list(zip(populations, self_play_names))[i+1:]:
            print(f'Relative population between {sp_name_1} VS {sp_name_2}')
            start_time = time()
            evolution_relative_performance = evolution_relative_population_performance(
                p1, p2, task,
                benchmarking_episodes,
                num_envs=-1)
            end_time = time() - start_time
            print('Took {:.2}'.format(end_time))
            # TODO: crucify me for adding string 'ppo' like a filthy hacker
            f1_name = f'{base_path}/{sp_name_1}-ppo_{sp_name_2}-ppo.pickle'
            f2_name = f'{base_path}/{sp_name_2}-ppo_{sp_name_1}-ppo.pickle'
            dill.dump(evolution_relative_performance, open(f1_name, 'wb'))
            dill.dump(-1 * evolution_relative_performance, open(f2_name, 'wb'))


if __name__ == '__main__':
    import os
    import gym_connect4
    from regym.environments import generate_task, EnvType
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path leading to YAML config file')
    args = parser.parse_args()

    paths = [os.path.join(args.path, e)
             for e in os.listdir(args.path)
             if 'relative_performances' not in e
            ]

    print(f'Computing relative population performance for path: {args.path}')
    compute_relative_pop_performance_all_populations(
        paths=paths,
        task=generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION),
	benchmarking_episodes=40,
	base_path=os.path.join(args.path, 'relative_performances')
    )
