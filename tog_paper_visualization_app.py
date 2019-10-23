import streamlit as st

from exploration_view import exploration_view
from optimality_view import optimality_view


def run():
    VIEWS = {'Optimality (external) measurements': optimality_view,
             'Exploration (internal) measurements': exploration_view}
    view_name = st.sidebar.selectbox("Choose view", list(VIEWS.keys()), 0)
    view = VIEWS[view_name]

    view()


if __name__ == '__main__':
    run()
