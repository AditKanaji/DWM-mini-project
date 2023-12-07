# ===== Importations ===== #
import streamlit as st
from utils import *
# ===== html/css config ===== #
st.set_page_config(layout="wide", 
    page_title="DWM Mini-Project", 
    menu_items={'About': "No-code AI Platform - réalisé par Antonin"},
)
st.markdown(CSS, unsafe_allow_html=True)

# ===== Page ===== #
st.markdown('<p class="first_titre">DWM Mini-Project</p>', unsafe_allow_html=True)
st.write("---")
c1, c2 = st.columns((3, 2))
with c2:
    st.write("##")
    st.write("##")
    st.image("logo/background.png")
st.write("##")
with c1:
    st.write("##")
    st.markdown(
        '<p class="intro">Authors: </p>',
        unsafe_allow_html=True)
    st.write(
        "Adit Kanaji")
    st.write("Ayushi Uttamani")
    st.write(
        "Tanaya Joshi")
    st.write("  ")
    st.markdown(
        '<p class="intro"></p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="intro"><b></b></p>',
        unsafe_allow_html=True)
c1, _, c2, _, _, _ = st.columns(6)
with c1:
    st.subheader(" ")
    st.write(
        " ")
    st.write("  ")
with c2:
    lottie_accueil = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_xRmNN8.json')
    st_lottie(lottie_accueil, height=200)
