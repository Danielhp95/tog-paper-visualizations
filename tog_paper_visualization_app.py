import os
import streamlit as st

import matplotlib
matplotlib.use('Agg') # This is mandatory, otherwise st crashes!

from exploration_view import exploration_view
from optimality_view import optimality_view


def data_directory_selection_sidebar_widget():
    import sys
    script_param_default_result = sys.argv[1] if len(sys.argv) > 1 else './'
    selected_results_directory = st.sidebar.text_input('Select results directory',
                                                       script_param_default_result)
    display_selected_directory(selected_results_directory)
    return selected_results_directory


def run():
    results_dir = data_directory_selection_sidebar_widget()
    VIEWS = {'Optimality (external) measurements': optimality_view,
             'Exploration (internal) measurements': exploration_view}
    view_name = st.sidebar.selectbox("Choose view", list(VIEWS.keys()), 0)
    view = VIEWS[view_name]
    view(results_dir)


def display_selected_directory(selected_dir: str):
    if st.sidebar.checkbox(f'list directory {selected_dir}: '):
        for f in os.listdir(selected_dir): st.sidebar.markdown(f)


if __name__ == '__main__':
    run()
