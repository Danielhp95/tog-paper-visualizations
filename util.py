import numpy as np
import pandas as pd
import streamlit as st


def softmax(x: np.array):
    score = np.exp(x)
    return score / score.sum(1) # sum over columns


def generate_random_discrete_distribution(size):
    return softmax(np.random.rand(1, size))


def generate_random_winrate_matrix(size):
    # Generate matrix of 0s
    winrate_matrix = np.zeros(shape=(size, size))
    # Generate upper triangular
    for i in range(size):
        winrate_vector = np.random.rand(size - i)
        winrate_matrix[i, i:] = winrate_vector
    # Generate complementary entries
    winrate_matrix += (np.triu(np.ones_like(winrate_matrix)) - winrate_matrix).transpose()
    for i in range(size): winrate_matrix[i,i] = 0.5 # Generate diagonal of 0.5

    return winrate_matrix.round(decimals=2)


@st.cache
def compute_progression_of_nash_during_training(range_checkpoints: range):
    # Fill with -1 to represent invalid values, these will be masked out
    number_of_nash_equilibriums = len(range_checkpoints)
    nash_progression = np.full((number_of_nash_equilibriums, number_of_nash_equilibriums),
                               fill_value=0, dtype=np.float64)
    nash_progression_df = pd.DataFrame()
    for i, checkpoint in enumerate(range_checkpoints):
        fake_nash = generate_random_discrete_distribution(i+1)
        zero_padded_fake_nash = np.concatenate((fake_nash.ravel(), np.zeros(number_of_nash_equilibriums - i -1))).round(decimals=2)
        nash_progression_df[checkpoint] = zero_padded_fake_nash
    return nash_progression_df


def highlight_text(ax, text_to_match, highlight_color='green', text_size=13):
    for lab, annot in zip(ax.get_yticklabels(), ax.texts):
        text =  lab.get_text()
        if text == text_to_match: # lets highlight row 2
            # set the properties of the ticklabel
            lab.set_weight('bold')
            lab.set_size(text_size)
            lab.set_color(highlight_color)
