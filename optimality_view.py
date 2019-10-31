from typing import Dict, Tuple
import streamlit as st
import pickle
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from util import generate_random_winrate_matrix, generate_random_discrete_distribution 
from util import compute_progression_of_nash_during_training
from util import highlight_text


def optimality_view(results_dir):
    name = 'Single empirical winrate game'
    st.write(f'# {name}')
    st.write(f'# {results_dir}')

    selfplay_choice = st.sidebar.radio('Select Self-Play algorithm',
                                       ('Naive SP', 'Full History SP', 'Iterated Nash Response'))

    progression_nash = pd.read_csv(f'{results_dir}/evolution_maxent_nash.csv', index_col=0)
    winrate_matrices = pickle.load(open(f'{results_dir}/winrate_matrices.pickle', 'rb'))

    min_checkpoint = int(progression_nash.index[0])
    max_checkpoint = int(progression_nash.index[-1])
    step_checkpoint = int(progression_nash.index[1] - progression_nash.index[0])
    checkpoint = st.sidebar.slider('Choose benchmarking checkpoint (episode number)',
                                   min_checkpoint, max_checkpoint, step=step_checkpoint)


    plot_progression_nash_equilibriums(progression_nash, highlight=checkpoint)

    st.write('## Winrate matrix and Logit matrix heatmaps')

    winrate_matrix = np.array(winrate_matrices[checkpoint])
    logit_matrix = np.log(winrate_matrix / (np.ones_like(winrate_matrix) - winrate_matrix))

    plot_game_matrices(winrate_matrix, logit_matrix)


def plot_game_matrices(winrate_matrix, logit_matrix):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Empirical winrate matrix')
    winrate_matrix_heatmap = sns.heatmap(winrate_matrix, annot=True, ax=ax[0], cmap=sns.color_palette('RdYlGn_r')[::-1],
                                         vmin=0.0, vmax=1.0, cbar_kws={'label': 'Head to head winrates'})
    ax[0].set_xlabel('Agent ID')
    ax[0].set_ylabel('Agent ID')

    ax[1].set_title('Log-odds winrate matrix')
    logit_matrix_heatmap = sns.heatmap(logit_matrix, annot=True, ax=ax[1], cmap=sns.color_palette('RdYlGn_r')[::-1],
                                       cbar=False)
    ax[1].set_xlabel('Agent ID')
    ax[0].set_ylim(len(winrate_matrix) + 0.2, -0.2)
    ax[1].set_ylim(len(logit_matrix) + 0.2, -0.2)
    plt.tight_layout()
    st.pyplot()
    plt.close()


def plot_progression_nash_equilibriums(progression_nash, highlight):
    fig, ax = plt.subplots(1, 1)
    # Only show lower triangular
    sns.heatmap(progression_nash, annot=True, vmax=1.0, vmin=0.0,
                cmap=sns.color_palette('RdYlGn_r')[::-1], cbar_kws={'label': 'Support under Nash'})
    # Workaround to prevent top and bottom of heatmaps to be cutoff
    # This is a known matplotlib bug
    ax.set_ylim(len(progression_nash) + 0.2, -0.2)
    plt.title('Progression of Nash equilibrium during training')
    plt.ylabel('Training iteration')
    plt.xlabel('Agent ID')

    highlight_text(ax, str(highlight))

    st.pyplot()
    plt.close()


@st.cache
def compute_winrate_matrices(num_matrices):
    matrices = {checkpoint: generate_random_winrate_matrix(size + 1)
                for size, checkpoint in enumerate(range(100, 1001, 100))}
    return matrices
