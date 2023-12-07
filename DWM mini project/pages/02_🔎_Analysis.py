import streamlit as st
from utils import *

st.set_page_config(layout="wide", page_title="Column analysis")
st.markdown(CSS, unsafe_allow_html=True)

# ===== Session ===== #
if "slider_col" not in st.session_state:
    st.session_state.slider_col = ""

# ===== Page ===== #
st.markdown('<p class="grand_titre">Column analysis</p>', unsafe_allow_html=True)
st.write('##')
if 'data' in st.session_state:
    options = st.session_state.data.columns.to_list()
    st.session_state.slider_col = st.multiselect(
        'Select one or more columns',
        options, help="Choose the columns to analyze",
        default=st.session_state.slider_col if st.session_state.slider_col else None
    )

    if st.session_state.slider_col:
        col1, b, col2, c = st.columns((1.1, 0.1, 1.1, 0.3))
        with col1:
            st.write('##')
            st.markdown('<p class="section">Preview</p>', unsafe_allow_html=True)
        with col2:
            st.write('##')
            st.markdown('<p class="section">Features</p>', unsafe_allow_html=True)
        for col in st.session_state.slider_col:
            ### Données ###
            data_col = st.session_state.data[col].copy()
            n_data = st.session_state.data[col].to_numpy()

            st.write('##')
            col1, b, col2, c = st.columns((1, 1, 2, 0.5))
            with col1:
                st.markdown('<p class="nom_colonne_page3">' + col + '</p>', unsafe_allow_html=True)
                st.write(data_col.head(20))
            with col2:
                st.write('##')
                st.write(' ● column type :', str(type(data_col)))
                st.write(' ● type of values :', str(type(data_col.iloc[1])))
                if n_data.dtype == float:
                    moyenne = data_col.mean()
                    variance = data_col.std()
                    max_ = data_col.max()
                    min_ = data_col.min()
                    st.write(' ● Average :', round(moyenne, 3))

                    st.write(' ● Variance :', round(variance, 3))

                    st.write(' ● Maximum :', max_)

                    st.write(' ● Minimum :', min_)

                st.write(' ● Most present values:', (Counter(n_data).most_common()[0])[0], 'appears',
                            (Counter(n_data).most_common()[0])[1], 'times', ', ', (Counter(n_data).most_common()[1])[0],
                            'appears',
                            (Counter(n_data).most_common()[1])[1], 'times')

                st.write(' ● Number of missing values:',
                            sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))

                st.write(' ● Length:', n_data.shape[0])

                st.write(' ● Number of different non-NaN values:',
                            abs(len(Counter(n_data)) - sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist())))
                ### Fin section données ###
            st.write('##')

else:
    st.info("Please upload your data in the Dataset section")
    st.write("##")
    st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)
