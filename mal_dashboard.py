import streamlit as st
import pandas as pd

import torch
import pytorch_model_summary as pms

from utils import CompTwo
from st_utils import load_model_v2, load_model_v3, load_model_v4, load_model_v5

st.set_page_config(layout="wide")


@st.cache()
def get_bias(model, class_ref):
    bias = model.anime_bias.weight.squeeze()

    idxs = bias.argsort()[:10]
    bot = [class_ref['title'][i] for i in idxs]

    idxs = bias.argsort(descending=True)[:10]
    top = [class_ref['title'][i] for i in idxs]

    return top, bot


if __name__ == '__main__':
    nav = st.sidebar.radio("Page Selection", ['Landing', 'Comparison'])

    model, class_ref = load_model_v5()
    username_mapping = {x: i for i, x in enumerate(class_ref['username'].items)}
    anime_mapping = {x: i for i, x in enumerate(class_ref['title'].items)}

    if nav == 'Landing':

        st.title("MyAnimeList PMF")
        st.write("This is a quick-and-dirty Anime Recommender built off MyAnimeList's API and data.")
        st.write("The recommender is implemented using Probabilistic Matrix Factorization.")

        top, bot = get_bias(model, class_ref)

        st.header("Model Details")

        summary = pms.summary(model, (torch.rand([64, 2])*1000).int(), show_input=True, max_depth=None, show_parent_layers=True)
        st.text(summary)
        n_users, n_animes = len(class_ref['username']), len(class_ref['title'])

        st.write(F"This model was trained on the data of {n_users} random users")

        st.header("Some Results")

        st.write("Below are the top and bottom 10 anime by bias")
        st.write("This is essentially what the model has learned to be the best and worst regardless of watcher")

        cols = st.columns(2)
        cols[0].subheader("Top 10")
        cols[0].table(top)
        cols[1].subheader("Bottom 10")
        cols[1].table(bot)

    elif nav == 'Comparison':

        user_form = st.form("comp")
        cols = user_form.columns(2)
        u1 = cols[0].selectbox("User 1", options=[""]+list(username_mapping.keys()))
        u2 = cols[1].selectbox("User 2", options=[""]+list(username_mapping.keys()))

        submitted = user_form.form_submit_button("Submit")

        if submitted:
            c2 = CompTwo(u1, u2, model, username_mapping, anime_mapping)
            c2.gen_combined()

            user_cols = user_form.columns(2)

            user_cols[0].subheader(u1)
            user_cols[0].write(F"Recommendations for {u1} from the shows that {u2} has watched")

            recs = c2.show_missing_preds(0).sort_values(ascending=False)
            recs.name = 'Value'
            user_cols[0].table(recs)

            user_cols[1].subheader(u2)
            user_cols[1].write(F"Recommendations for {u2} from the shows that {u1} has watched")

            recs = c2.show_missing_preds(1).sort_values(ascending=False)
            recs.name = 'Value'
            user_cols[1].table(recs)


