import streamlit as st

from utils import _load_model_v2, _load_model_v3, _load_model_v4, _load_model_v5


@st.cache(allow_output_mutation=True)
def load_model_v2():
    return _load_model_v2()


@st.cache(allow_output_mutation=True)
def load_model_v3():
    return _load_model_v3()


@st.cache(allow_output_mutation=True)
def load_model_v4():
    return _load_model_v4()


@st.cache(allow_output_mutation=True)
def load_model_v5():
    return _load_model_v5()
