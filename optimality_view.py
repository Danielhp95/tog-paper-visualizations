from os import listdir
from os.path import isdir, join
from typing import Dict, Tuple, List
import yaml
import streamlit as st
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from util import generate_random_winrate_matrix, generate_random_discrete_distribution
from util import compute_progression_of_nash_during_training
from util import highlight_text


def load_results(experiment_dir: str, run_id: str, selfplay_choice: str):
    configs = yaml.load(open(f'{experiment_dir}experiment_parameters.yml'), Loader=yaml.FullLoader)

    run_path = f'{experiment_dir}{run_id}'
    progression_nash = pd.read_csv(f'{run_path}/{selfplay_choice}/results/evolution_maxent_nash.csv', index_col=0)
    winrate_matrices = pickle.load(open(f'{run_path}/{selfplay_choice}/results/winrate_matrices.pickle', 'rb'))

    final_winrate_matrix = pickle.load(open(f'{run_path}/final_winrate_matrix.pickle', 'rb'))
    final_nash           = pickle.load(open(f'{run_path}/final_maxent_nash.pickle', 'rb'))

    rel_pop_performances = load_relative_performances(f'{experiment_dir}',
                                                      configs['experiment']['number_of_runs'],
                                                      selfplay_choice)
    return configs, final_winrate_matrix, final_nash, \
           progression_nash, winrate_matrices, rel_pop_performances


def load_relative_performances(path: str, num_runs: int, selfplay_choice: str):
    # Find names from run-0 directory
    # Loop over all run_id
    rel_performances = {}
    other_selfplay_names = []
    rel_perf = {f.split('_')[-1].split('.')[0]: []
                for f in listdir(f'{path}run-0/relative_performances/') if f.startswith(selfplay_choice)}

    for run_id in range(num_runs):
        run_path = f'{path}run-{run_id}/relative_performances'
        for f in [f for f in listdir(run_path) if f.startswith(selfplay_choice)]:
            other_sp = f.split('_')[-1].split('.')[0]
            rel_perf[other_sp].append(pickle.load(open(join(run_path,f), 'rb')))

    for k, v in rel_perf.items(): rel_perf[k] = np.array(v)
    return rel_perf


def optimality_view(experiment_dir):
    # TODO: refactor somehow, this is not pretty
    experiment_runs = [d for d in listdir(experiment_dir) if isdir(join(experiment_dir, d))]
    run_id = st.sidebar.selectbox('Select experiment run_id', experiment_runs, 0)
    results_dir = f'{experiment_dir}{run_id}/'
    selfplay_schemes = [d for d in listdir(results_dir)
                        if isdir(join(results_dir, d)) and d != 'relative_performances']

    selfplay_choice = st.sidebar.radio('Select Self-Play algorithm', selfplay_schemes)

    configs, final_winrate_matrix, final_nash, progression_nash, \
        winrate_matrices, rel_pop_performances = load_results(experiment_dir, run_id, selfplay_choice)

    environment_name = configs['experiment']['environment'][0]
    st.write(f'# Optimality view\n')
    st.write(f'# Environment: {environment_name}')
    st.write(f'## SP: {selfplay_choice}')
    st.write(f'## Run: {run_id}')

    min_checkpoint = int(progression_nash.index[0])
    max_checkpoint = int(progression_nash.index[-1])
    step_checkpoint = int(progression_nash.index[1] - progression_nash.index[0])
    checkpoint = st.sidebar.slider('Choose benchmarking checkpoint (episode number)',
                                   min_checkpoint, max_checkpoint, step=step_checkpoint)

    st.write('## Final winrate matrix and Nash support')
    plot_joint_final_winrate_matrix_and_nash(final_winrate_matrix, final_nash,
                                             selfplay_schemes=selfplay_schemes)

    st.write(f'## Progression of nash equilibrium for {selfplay_choice}')

    plot_progression_nash_equilibriums(progression_nash, highlight=checkpoint)

    st.write('## Winrate matrix and Logit matrix heatmaps')

    winrate_matrix = np.array(winrate_matrices[checkpoint])
    # TODO: fix this monstrosity
    nash_support = np.array(list(filter(lambda x: not np.isnan(x),
                                        progression_nash.iloc[int(checkpoint / step_checkpoint)])))
    plot_winrate_matrix_and_support(winrate_matrix, nash_support)
    
    population_choices = st.multiselect('Select Self-play schemes to compare population performance',
                                        list(rel_pop_performances.keys()))
    relevant_relative_performances = {pop_name: rel_pop_performances[pop_name]
                                      for pop_name in population_choices}
    checkpoints = list(range(min_checkpoint, max_checkpoint + 1, step_checkpoint))
    plot_relative_performance_evolutions(selfplay_choice,
                                         relevant_relative_performances,
                                         checkpoints)

    st.write('## Config file for experiment')
    st.write(configs)


def plot_relative_performance_evolutions(selfplay_choice: str,
                                         relative_performances: Dict[str, np.ndarray],
                                         checkpoints):
    fig, ax = plt.subplots(1,1)
    plt.title(f'Evolution of relative population performance of {selfplay_choice}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Relative population performance')

    for sp_name, relative_performance_evolution in relative_performances.items():
        mean = relative_performance_evolution.mean(axis=0)
        std  = relative_performance_evolution.std(axis=0)
        ax.plot(checkpoints, mean, label=f'vs {sp_name}')
        ax.fill_between(checkpoints, mean + std, mean - std, alpha=0.3)

    ax.legend()
    plt.tight_layout()
    st.pyplot()
    plt.close()


def plot_winrate_matrix_and_support(winrate_matrix, nash_support):
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]})
    max_comprehensible_size = 8
    show_annotations = winrate_matrix.shape[0] <= max_comprehensible_size
    plot_winrate_matrix(ax[0], winrate_matrix, show_annotations)
    plot_nash_support(ax[1], nash_support, show_ticks=True)

    plt.tight_layout()
    st.pyplot()
    plt.close()


def plot_winrate_matrix(ax, winrate_matrix, show_annotations):
    ax.set_title('Empirical winrate matrix')
    sns.heatmap(winrate_matrix, annot=show_annotations, ax=ax, square=True,
                cmap=sns.color_palette('RdYlGn_r', 50)[::-1],
                vmin=0.0, vmax=1.0, cbar_kws={'label': 'Head to head winrates'})
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Agent ID')
    ax.set_ylim(len(winrate_matrix) + 0.2, -0.2)


def plot_progression_nash_equilibriums(progression_nash, highlight):
    fig, ax = plt.subplots(1, 1)
    # Only show lower triangular
    max_comprehensible_size = 8
    show_annotations = len(progression_nash.shape) >= max_comprehensible_size
    sns.heatmap(progression_nash, annot=show_annotations, vmax=1.0, vmin=0.0,
                cmap=sns.color_palette('RdYlGn_r', 50)[::-1], cbar_kws={'label': 'Support under Nash'})
    # Workaround to prevent top and bottom of heatmaps to be cutoff
    # This is a known matplotlib bug
    ax.set_ylim(len(progression_nash) + 0.2, -0.2)
    plt.title('Progression of Nash equilibrium during training')
    plt.ylabel('Training iteration')
    plt.xlabel('Agent ID')

    highlight_text(ax, str(highlight))

    st.pyplot()
    plt.close()


def plot_joint_final_winrate_matrix_and_nash(winrate_matrix: np.ndarray,
                                             nash: np.ndarray,
                                             selfplay_schemes: List[str]):
    population_size = int(winrate_matrix.shape[0] / len(selfplay_schemes))
    number_populations = len(selfplay_schemes)
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})
    plot_final_winrate_matrix(ax[0], winrate_matrix, selfplay_schemes,
                              population_size, number_populations)
    plot_population_delimiting_lines(ax, winrate_matrix.shape[0],
                                     number_populations)
    plot_nash_support(ax[1], nash, show_ticks=False)

    fig.suptitle('Cross self-play nash evaluation')

    # Workaround to prevent top and bottom of heatmaps to be cutoff
    # This is a known matplotlib bug
    ax[0].set_ylim(len(winrate_matrix) + 0.2, -0.2)
    ax[1].set_ylim(len(winrate_matrix) + 0.2, -0.2)

    plt.tight_layout()
    st.pyplot()
    plt.close()


def plot_final_winrate_matrix(ax, winrate_matrix, selfplay_schemes,
                              population_size, number_populations):
    sns.heatmap(winrate_matrix, annot=False, ax=ax,
                vmin=0, vmax=1, cbar=False,
                cmap=sns.color_palette('RdYlGn_r', 50)[::-1])

    first_tick = population_size / 2
    ticks = [first_tick + i * population_size
             for i in range(number_populations)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(selfplay_schemes)
    ax.set_yticks(ticks)
    ax.set_yticklabels(selfplay_schemes)
    ax.set_yticks([]) # TODO Add episodes checkpoints where the agents were frozen


def plot_population_delimiting_lines(ax, length, number_populations):
    for i_delimiter in range(0, length + 1,
                             int(length / number_populations)):
        ax[0].vlines(x=i_delimiter, ymin=0, ymax=length,
                     color='b', lw=1)
        ax[0].hlines(y=i_delimiter, xmin=0, xmax=length,
                     color='b', lw=1)
        ax[1].hlines(y=i_delimiter, xmin=0, xmax=length,
                     color='b', lw=2)


def plot_nash_support(ax, nash, show_ticks):
    max_support = np.max(nash)
    column_nash = np.reshape(nash, (nash.shape[0], 1))
    sns.heatmap(column_nash, ax=ax,
                vmin=0, vmax=max_support,
                cmap=sns.color_palette('RdYlGn_r', 50)[::-1])
    if show_ticks:
        ax.set_xticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_ylim(len(nash) + 0.2, -0.2)
