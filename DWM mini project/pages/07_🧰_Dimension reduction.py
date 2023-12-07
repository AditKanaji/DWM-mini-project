# ===== Importations ===== #
import streamlit as st
from utils import *
# ===== html/css config ===== #
st.set_page_config(layout="wide", page_title="Dimension reduction")
st.markdown(CSS, unsafe_allow_html=True)

# ===== choix modèle ===== #
PAGES_reduction = ["🪛 PCA", "🪛 UMAP", "🪛 T-SNE"]
st.sidebar.selectbox(label="label", options=PAGES_reduction, key="choix_page_reduction", label_visibility='hidden')

# ===== Page ===== #
if st.session_state.choix_page_reduction == "🪛 PCA":
    st.markdown('<p class="grand_titre">PCA : Principal component analysis</p>', unsafe_allow_html=True)
    st.write("##")
    exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
    with exp2:
        with st.expander("Principle of PCA"):
            st.write("""
           PCA is applied to quantitative variables of the same importance.
            * 1st step: we determine the covariance matrix between each variable
            * 2nd step: By normalization of the latter, we obtain the correlation matrix
            * 3rd step: By diagonalization we find the proper elements
            * 4th step: We determine the main components
            * 5th step: We calculate the saturation matrix
            * 6th step: Graphical representation
            """)
    if 'data' in st.session_state:
        _, col1_pca, _ = st.columns((0.1, 1, 0.1))
        _, sub_col1, _ = st.columns((0.4, 0.5, 0.4))
        _, col2_pca, _ = st.columns((0.1, 1, 0.1))
        with col1_pca:
            st.write("##")
            st.markdown('<p class="section">Selection of columns for the model PCA (target+features)</p>',
                        unsafe_allow_html=True)
            st.session_state.choix_col_PCA = st.multiselect("Choose at least two columns",
                                                            st.session_state.data.columns.tolist(),
                                                            )
        if len(st.session_state.choix_col_PCA) > 1:
            df_ml = st.session_state.data[st.session_state.choix_col_PCA]
            df_ml = df_ml.dropna(axis=0)
            st.session_state.df_ml_origine_PCA = df_ml.copy()
            if len(df_ml) == 0:
                with col1_pca:
                    st.write("##")
                    st.warning('The dataset with removal of NaNs along the lines is empty!')
            else:
                with col1_pca:
                    # encodage !
                    st.session_state.col_to_encodage_PCA = st.multiselect("Select the columns to encode",
                                                                            st.session_state.choix_col_PCA)
                with sub_col1:
                    with st.expander('Encoding'):
                        for col in st.session_state.col_to_encodage_PCA:
                            st.write(
                                "column encoding " + col + " : " + str(df_ml[col].unique().tolist()) + "->" + str(
                                    np.arange(len(df_ml[col].unique()))))
                            df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())),
                                                inplace=True)  # encodage
                    ## on choisit notre modèle
                    model = make_pipeline(StandardScaler(), PCA(n_components=2))
                with col2_pca:
                    ## création des target et features à partir du dataset
                    st.write("##")
                    st.write("##")
                    st.session_state.target_PCA = st.selectbox("Target :",
                                                                ["Select a target"] + col_numeric(df_ml),
                                                                )
                if st.session_state.target_PCA != "Select a target":
                    y = df_ml[st.session_state.target_PCA]  # target
                    X = df_ml.drop(st.session_state.target_PCA, axis=1)  # features

                    try:
                        with col2_pca:
                            model.fit(X)
                            x_pca = model.transform(X)
                            st.write("##")
                            st.markdown('<p class="section">Results</p>', unsafe_allow_html=True)
                            st.session_state.df = pd.concat([pd.Series(x_pca[:, 0]).reset_index(drop=True),
                                                                pd.Series(x_pca[:, 1]).reset_index(drop=True),
                                                                pd.Series(st.session_state.df_ml_origine_PCA[
                                                                            st.session_state.target_PCA]).reset_index(
                                                                    drop=True)], axis=1)
                            st.session_state.df.columns = ["x", "y", str(st.session_state.target_PCA)]
                            fig = px.scatter(st.session_state.df, x="x", y="y",
                                                color=str(st.session_state.target_PCA),
                                                labels={'color': '{}'.format(str(st.session_state.target_PCA))},
                                                color_discrete_sequence=px.colors.qualitative.Plotly)
                            fig.update_layout(
                                showlegend=True,
                                template='simple_white',
                                font=dict(size=10),
                                autosize=False,
                                width=1050, height=650,
                                margin=dict(l=40, r=50, b=40, t=40),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                            )
                            fig.update(layout_coloraxis_showscale=False)
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        with col2_pca:
                            st.write("##")
                            st.warning("Cannot use this model with this data")
    else:
        with exp2:
            st.write("##")
            st.info('Go to the Dataset section to import your dataset')
            st.write("##")
        st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)

elif st.session_state.choix_page_reduction == "🪛 UMAP":
    st.markdown('<p class="grand_titre">UMAP : Uniform Manifold Approximation and Projection</p>',
                unsafe_allow_html=True)
    exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
    if 'data' in st.session_state:
        _, col1_umap, _ = st.columns((0.1, 1, 0.1))
        _, sub_col1, _ = st.columns((0.4, 0.5, 0.4))
        _, col2_umap, _ = st.columns((0.1, 1, 0.1))
        with col1_umap:
            st.write("##")
            st.markdown('<p class="section">Selection of columns for the model UMAP (target+features)</p>',
                        unsafe_allow_html=True)
            st.session_state.choix_col_UMAP = st.multiselect("Choose at least two columns",
                                                                st.session_state.data.columns.tolist(),
                                                                help="The target must be selected"
                                                                )
        if len(st.session_state.choix_col_UMAP) > 1:
            df_ml = st.session_state.data[st.session_state.choix_col_UMAP]
            df_ml = df_ml.dropna(axis=0)
            st.session_state.df_ml_origine_UMAP = df_ml.copy()
            if len(df_ml) == 0:
                with col1_umap:
                    st.write("##")
                    st.warning('The dataset with removal of NaNs along the lines is empty!')
            else:
                with col1_umap:
                    # encodage !
                    st.session_state.col_to_encodage_UMAP = st.multiselect("Select the columns to encode",
                                                                            st.session_state.choix_col_UMAP
                                                                            )
                with sub_col1:
                    with st.expander('Encoding'):
                        for col in st.session_state.col_to_encodage_UMAP:
                            st.write(
                                "column encoding " + col + " : " + str(df_ml[col].unique().tolist()) + "->" + str(
                                    np.arange(len(df_ml[col].unique()))))
                            df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())),
                                                inplace=True)  # encodage

                with col2_umap:
                    ## on choisit notre modèle
                    model = make_pipeline(StandardScaler(), umap.UMAP())
                    ## création des target et features à partir du dataset
                    st.write("##")
                    st.session_state.target_UMAP = st.selectbox("Target :",
                                                                ["Select a target"] + col_numeric(df_ml),
                                                                )
                if st.session_state.target_UMAP != "Select a target":
                    y = df_ml[st.session_state.target_UMAP]  # target
                    X = df_ml.drop(st.session_state.target_UMAP, axis=1)  # features
                    with col2_umap:
                        try:
                            model.fit(X)
                            x_umap = model.transform(X)
                            st.write("##")
                            st.write("##")
                            st.markdown('<p class="section">Results</p>', unsafe_allow_html=True)
                            st.session_state.df = pd.concat([pd.Series(x_umap[:, 0]), pd.Series(x_umap[:, 1]),
                                                                pd.Series(st.session_state.df_ml_origine_UMAP[
                                                                            st.session_state.target_UMAP])], axis=1)
                            st.session_state.df.columns = ["x", "y", str(st.session_state.target_UMAP)]
                            fig = px.scatter(st.session_state.df, x="x", y="y",
                                                color=str(st.session_state.target_UMAP),
                                                labels={'color': '{}'.format(str(st.session_state.target_UMAP))},
                                                color_discrete_sequence=px.colors.qualitative.Plotly)
                            fig.update_layout(
                                showlegend=True,
                                template='simple_white',
                                font=dict(size=10),
                                autosize=False,
                                width=1050, height=650,
                                margin=dict(l=40, r=50, b=40, t=40),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                            )
                            fig.update(layout_coloraxis_showscale=False)
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.write("##")
                            st.warning("Cannot use this model with this data")
    else:
        with exp2:
            st.write("##")
            st.info('Go to the Dataset section to import your dataset')
            st.write("##")
            st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)

elif st.session_state.choix_page_reduction == "🪛 T-SNE":
    st.markdown('<p class="grand_titre">T-SNE : t-distributed stochastic neighbor embedding</p>',
                unsafe_allow_html=True)
    exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
    if 'data' in st.session_state:
        _, col1_tsne, _ = st.columns((0.1, 1, 0.1))
        _, sub_col1, _ = st.columns((0.4, 0.5, 0.4))
        _, col2_tsne, _ = st.columns((0.1, 1, 0.1))
        with col1_tsne:
            st.write("##")
            st.markdown('<p class="section">Selection of columns for the model T-SNE (target+features)</p>',
                        unsafe_allow_html=True)
            st.session_state.choix_col_tsne = st.multiselect("Choose at least two columns",
                                                                st.session_state.data.columns.tolist(),
                                                                help="The target must be selected"
                                                                )
        if len(st.session_state.choix_col_tsne) > 1:
            df_ml = st.session_state.data[st.session_state.choix_col_tsne]
            df_ml = df_ml.dropna(axis=0)
            st.session_state.df_ml_origine_tsne = df_ml.copy()
            if len(df_ml) == 0:
                with col1_tsne:
                    st.write("##")
                    st.warning('The dataset with removal of NaNs along the lines is empty!')
            else:
                with col1_tsne:
                    # encodage !
                    st.session_state.col_to_encodage_tsne = st.multiselect("Select the columns to encode",
                                                                            st.session_state.choix_col_tsne,
                                                                            )
                with sub_col1:
                    with st.expander('Encodage'):
                        for col in st.session_state.col_to_encodage_tsne:
                            st.write(
                                "column encoding " + col + " : " + str(df_ml[col].unique().tolist()) + "->" + str(
                                    np.arange(len(df_ml[col].unique()))))
                            df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())),
                                                inplace=True)  # encodage

                with col2_tsne:
                    ## on choisit notre modèle
                    model = TSNE(n_components=2, random_state=0)
                    ## création des target et features à partir du dataset
                    st.write("##")
                    st.session_state.target_tsne = st.selectbox("Target :",
                                                                ["Select a target"] + col_numeric(df_ml),
                                                                )
                if st.session_state.target_tsne != "Select a target":
                    y = df_ml[st.session_state.target_tsne]  # target
                    X = df_ml.drop(st.session_state.target_tsne, axis=1)  # features
                    with col2_tsne:
                        try:
                            x_tsne = model.fit_transform(X, )
                            st.write("##")
                            st.write("##")
                            st.markdown('<p class="section">Results</p>', unsafe_allow_html=True)
                            st.session_state.df = pd.concat([pd.Series(x_tsne[:, 0]), pd.Series(x_tsne[:, 1]),
                                                                pd.Series(st.session_state.df_ml_origine_tsne[
                                                                            st.session_state.target_tsne])], axis=1)
                            st.session_state.df.columns = ["x", "y", str(st.session_state.target_tsne)]
                            fig = px.scatter(st.session_state.df, x="x", y="y",
                                                color=str(st.session_state.target_tsne),
                                                labels={'color': '{}'.format(str(st.session_state.target_tsne))},
                                                color_discrete_sequence=px.colors.qualitative.Plotly)
                            fig.update_layout(
                                showlegend=True,
                                template='simple_white',
                                font=dict(size=10),
                                autosize=False,
                                width=1050, height=650,
                                margin=dict(l=40, r=50, b=40, t=40),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                            )
                            fig.update(layout_coloraxis_showscale=False)
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.write("##")
                            st.warning("Cannot use this model with this data")
    else:
        with exp2:
            st.write("##")
            st.info('Go to the Dataset section to import your dataset')
            st.write("##")
            st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)