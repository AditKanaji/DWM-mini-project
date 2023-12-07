# ===== Importations ===== #
import streamlit as st
from utils import *
# ===== html/css config ===== #
st.set_page_config(layout="wide", page_title="Correlation matrix")
st.markdown(CSS, unsafe_allow_html=True)

# ===== Session ===== #
if "couleur_corr" not in st.session_state:
    st.session_state.couleur_corr = ""
if "select_columns_corr" not in st.session_state:
    st.session_state.select_columns_corr = ""

# ===== Page ===== #
st.markdown('<p class="grand_titre">Correlation matrix</p>', unsafe_allow_html=True)
st.write("##")
if 'data' in st.session_state:
    col1, b, col2 = st.columns((1, 1, 2))
    df_sans_NaN = st.session_state.data
    with col1:
        st.session_state.couleur_corr = st.multiselect('Color',
                                                df_sans_NaN.columns.tolist(),
                                                help="Choose the categorical variable for class coloring",
                                                max_selections=1, default=st.session_state.couleur_corr if st.session_state.couleur_corr else None
                                                )

        st.write("##")
    st.session_state.select_columns_corr = st.multiselect("Choose at least two columns",
                                                            ["All columns"] + col_numeric(df_sans_NaN),
                                                            help="Choose your features", default=st.session_state.select_columns_corr if st.session_state.select_columns_corr else None
                                                            )
    if len(st.session_state.select_columns_corr) > 1 and "All columns" not in st.session_state.select_columns_corr:
        df_sans_NaN = pd.concat([st.session_state.data[col] for col in st.session_state.select_columns_corr],
                                axis=1).dropna()
        if len(df_sans_NaN) == 0:
            st.write("##")
            st.warning('The dataset with removal of NaNs along the lines is empty!')
        else:
            if st.session_state.couleur_corr:
                fig = px.scatter_matrix(st.session_state.data,
                                        dimensions=col_numeric(df_sans_NaN[st.session_state.select_columns_corr]),
                                        color=st.session_state.couleur_corr[0], color_continuous_scale='Bluered_r')
            else:
                fig = px.scatter_matrix(df_sans_NaN,
                                        dimensions=col_numeric(df_sans_NaN[st.session_state.select_columns_corr]))
            fig.update_layout(width=1000, height=700, margin=dict(l=40, r=50, b=40, t=40), font=dict(size=10))
            fig.update_layout({"xaxis" + str(i + 1): dict(showticklabels=False) for i in
                                range(len(col_numeric(df_sans_NaN[st.session_state.select_columns_corr])))})
            fig.update_layout({"yaxis" + str(i + 1): dict(showticklabels=False) for i in
                                range(len(col_numeric(df_sans_NaN[st.session_state.select_columns_corr])))})
            fig.update_traces(marker=dict(size=7))
            fig.update_traces(diagonal_visible=False)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    elif st.session_state.select_columns_corr == [
        "All columns"]:
        df_sans_NaN = st.session_state.data.dropna()
        if len(df_sans_NaN) == 0:
            st.write("##")
            st.warning('The dataset with removal of NaNs along the lines is empty!')
        else:
            if st.session_state.couleur_corr:
                fig = px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN),
                                        color=st.session_state.couleur_corr[0])
            else:
                fig = px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN))
            fig.update_layout(
                {"xaxis" + str(i + 1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN)))})
            fig.update_layout(
                {"yaxis" + str(i + 1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN)))})
            fig.update_traces(marker=dict(size=2))
            fig.update_layout(width=1000, height=700, margin=dict(l=40, r=50, b=40, t=40), font=dict(size=10))
            fig.update_traces(marker=dict(size=7))
            fig.update_traces(diagonal_visible=False)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.write("##")
            st.plotly_chart(fig, use_container_width=True)
    elif len(st.session_state.select_columns_corr) > 1 and "All columns" in st.session_state.select_columns_corr:
        st.error("Input error !")
    else:
        pass
else:
    st.info("Please upload your data in the Dataset section")
    st.write("##")
    st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)