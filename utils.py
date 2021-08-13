from functools import singledispatchmethod
import pandas as pd
import torch
import os
import pickle

from models import DotProduct, CollabNN


def _load_model_v2(path="/plex_test/models/v2"):
    n_users = 12373
    n_animes = 14205
    n_factors = 20

    m = DotProduct(n_users, n_animes, n_factors)
    m.load_state_dict(torch.load(path))
    m.eval()

    with open("/plex_test/models/v2_class_ref", 'rb') as f:
        class_ref = pickle.load(f)

    return m, class_ref


def _load_model_v3(path="/plex_test/models/v3"):
    n_users = 12373
    n_animes = 14205
    n_factors = 50

    m = DotProduct(n_users, n_animes, n_factors)
    m.load_state_dict(torch.load(path))
    m.eval()

    with open("/plex_test/models/v3_class_ref", 'rb') as f:
        class_ref = pickle.load(f)

    return m, class_ref


def _load_model_v4(path="/plex_test/models/v4"):
    m = CollabNN((16174, 364), (14313, 340))
    m.load_state_dict(torch.load(path))
    m.eval()

    with open("/plex_test/models/v4_class_ref", 'rb') as f:
        class_ref = pickle.load(f)

    return m, class_ref


def _load_model_v5(path="/plex_test/models/v5"):
    m = CollabNN((16177, 364), (14313, 340))
    m.load_state_dict(torch.load(path))
    m.eval()

    with open("/plex_test/models/v5_class_ref", 'rb') as f:
        class_ref = pickle.load(f)

    return m, class_ref


def read_user_data(username, root='/plex_test/mal'):
    user = pd.read_csv(
            F"{root}/{username}.csv",
            usecols=['title', 'status', 'score'],
            header=0,
            engine='c',
            dtype={'title': str, "status": str, 'score': "int8"}
        )
    user['username'] = username
    user = user[user['status'] == 'completed']
    user = user[user['score'] != 0]

    return user


def get_data(root='/plex_test/mal'):
    dfs = []
    files = set(os.listdir('/plex_test/mal')) - set(['models'])
    for f in list(files):
        username = f[:-4]
        try:
            user = read_user_data(username, root)
        except Exception:
            continue

        dfs.append(user)

    return pd.concat(dfs, axis=0)


def gen_user_df(user, model, username_mapping, anime_mapping):
    df = read_user_data(user)
    # df = df[~df['title'].isin(['Yakusoku no Neverland 2nd Season: Michishirube', 'Yakusoku no Neverland 2nd Season'])]
    user_1 = df[['username', 'title', 'score']]
    user_1['pred'] = user_1.loc[:, 'title'].apply(lambda x: make_prediction(user, x, model, username_mapping, anime_mapping)).values
    user_1['total'] = user_1[['score', 'pred']].sum(axis=1)
    user_1 = user_1.sort_values('total')

    return user_1


def make_prediction(user, anime, model, username_mapping, anime_mapping):
    inp = torch.tensor([[username_mapping[user], anime_mapping[anime]]])

    res = model(inp)

    return res.cpu()[0][0].item()


class CompTwo():
    def __init__(self, username_1, username_2, model, username_mapping, anime_mapping):
        self.username_1 = username_1
        self.username_2 = username_2

        self.model = model
        self.username_mapping = username_mapping
        self.anime_mapping = anime_mapping

        self.user_1 = gen_user_df(username_1, model, username_mapping, anime_mapping)
        self.user_2 = gen_user_df(username_2, model, username_mapping, anime_mapping)

        self._ref_s = {
            username_1: self.user_1,
            username_2: self.user_2
        }

        self._ref_i = {
            0: username_1,
            1: username_2
        }

    @singledispatchmethod
    def calc_var(self, ref):
        raise NotImplementedError("Must refer to user by position or by name")

    @calc_var.register
    def _(self, ref: int):
        df = self._ref_s[self._ref_i[ref]]

        return (df['pred'] - df['score']).var()

    @calc_var.register
    def _(self, ref: str):
        df = self._ref_s[ref]

        return (df['pred'] - df['score']).var()

    def gen_combined(self):
        self.combined = pd.concat([self.user_1, self.user_2]).pivot_table(index='title', columns='username', values=['score', 'pred'])
        self.combined = self._fill_preds_for_combined(self.combined)

    def _fill_preds_for_combined(self, combined):

        for user, sub_df in combined['pred'].iteritems():
            other = list(set([self.username_1, self.username_2]) - set([user]))[0]
            for anime, score in sub_df.iteritems():
                if score != score:  # if nan
                    pred = make_prediction(user, anime, self.model, self.username_mapping, self.anime_mapping)

                    combined.loc[anime, pd.IndexSlice['pred', other]] = pred
                    # print(user, other, anime, pred, make_prediction(user, anime, self.model, self.username_mapping, self.anime_mapping))

        return combined

    @singledispatchmethod
    def show_missing_preds(self, ref):
        raise NotImplementedError("Must refer to user by position or by name")

    @show_missing_preds.register
    def _(self, ref: int):

        username = self._ref_i[ref]
        other = self._ref_i[1-ref]

        animes_minus = self.combined['score', username].isnull()  # Animes that username has seen but the other hasnt

        return self.combined['pred', other][animes_minus].sort_values()

    @show_missing_preds.register
    def _(self, ref: str):

        username = ref
        other = list((set(self._ref_s.keys())-set([ref])))[0]  # Crazy shit just to get the other one
#         other = self._ref_s[other]

        animes_minus = self.combined['score', username].isnull()  # Animes that username has seen but the other hasnt

        return self.combined['pred', other][animes_minus].sort_values()
