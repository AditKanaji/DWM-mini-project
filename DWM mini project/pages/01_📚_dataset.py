# ===== Importations ===== #
from utils import *
from pydataset import data

# ===== html/css config ===== #
st.set_page_config(layout="wide", page_title="Dataset")
st.markdown(CSS, unsafe_allow_html=True)

# ===== Session ===== #
if "drop_col" not in st.session_state:
    st.session_state.drop_col = ""
if "col_to_time" not in st.session_state:
    st.session_state.col_to_time = ""
if "col_to_float_money" not in st.session_state:
    st.session_state.col_to_float_money = ""
if "col_to_float_coma" not in st.session_state:
    st.session_state.col_to_float_coma = ""
if "separateur" not in st.session_state:
    st.session_state.separateur = ""
if "file_details" not in st.session_state:
    st.session_state.file_details = ""


# ===== Load data ===== #
def load_data():
    try:
        if 'csv' in st.session_state.file_details['FileName']:
            if st.session_state.separateur != "":
                st.session_state.data = pd.read_csv(uploaded_file, sep=st.session_state.separateur, engine='python')
            else:
                st.session_state.data = pd.read_csv(uploaded_file)
        else:
            if st.session_state.separateur != "":
                st.session_state.data = pd.read_excel(uploaded_file, sep=st.session_state.separateur, engine='python')
            else:
                st.session_state.data = pd.read_excel(uploaded_file)
    except:
        pass


# ===== Page ===== #
st.markdown('<p class="grand_titre">Loading the dataset</p>', unsafe_allow_html=True)
st.write('##')
col1_1, b_1, col2_1 = st.columns((1, 0.1, 1))
col1, b, col2 = st.columns((2.7, 0.2, 1))
with col2_1:
    st_lottie(load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_zidar9jm.json'), height=200)
with col1_1:
    dataset_choix = st.selectbox("Dataset",
                                 ["-- Choose an option --"] +
                                 LOCAL_DATASET_NAMES + PYDATASET_DISPLAY_NAMES +
                                 ["Choose a personal dataset"])
    message_ = st.empty()

if 'choice_dataset' in st.session_state:
    with col1_1:
        message_.success(st.session_state.choice_dataset)

if dataset_choix == "-- Choose an option --":
    if 'choice_dataset' in st.session_state:
        if st.session_state.choice_dataset in [f"The loaded file is the dataset {name}" for name in LOCAL_DATASET_NAMES+PYDATASET_DISPLAY_NAMES]:
            with col1:
                st.write("##")
                st.markdown('<p class="section">Preview</p>', unsafe_allow_html=True)
                st.write(st.session_state.data.head(50))
                st.write("##")

            with col2:
                st.write("##")
                st.markdown('<p class="section">Features</p>', unsafe_allow_html=True)
                st.write('- Size:', st.session_state.data.shape)
                st.write(' - Number of values:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                st.write(' - Column type:', st.session_state.data.dtypes.value_counts())
                st.write(' - Percentage of missing values:', round(
                    sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                            st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                         ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')
        elif st.session_state.choice_dataset == "You have chosen to select your dataset" and 'data' in st.session_state:
            with col1:
                st.write("##")
                st.markdown('<p class="section">Preview</p>', unsafe_allow_html=True)
                st.write(st.session_state.data.head(50))
                st.write("##")

            with col2:
                st.write("##")
                st.markdown('<p class="section">Features</p>', unsafe_allow_html=True)
                st.write(' - Size:', st.session_state.data.shape)
                st.write(' - Number of values:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                st.write(' - Column type:', st.session_state.data.dtypes.value_counts())
                st.write(' - Percentage of missing values:', round(
                    sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                            st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                         ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')

            col1_modif, col2_modif = st.columns((2, 1))
            with col1_modif:
                st.write("##")
                st.info("Reload your dataset to modify it, by clicking TWICE on the button below")
            if st.button("Edit the dataset"):
                dataset_choix = ""
                st.session_state.choice_dataset = ""
                st.session_state.clear()

# Local datasets
for i, j in zip(LOCAL_DATASET_NAMES, LOCAL_PATH_DATASET):
    if dataset_choix == i:
        st.session_state.data = pd.read_csv(j)
        st.session_state.choice_dataset = "The loaded file is the dataset " + i
        with col1_1:
            message_.success(st.session_state.choice_dataset)

        with col1:
            st.write("##")
            st.markdown('<p class="section">Preview</p>', unsafe_allow_html=True)
            st.write(st.session_state.data.head(50))
            st.write("##")

        with col2:
            st.write("##")
            st.markdown('<p class="section">Features</p>', unsafe_allow_html=True)
            st.write(' - Size:', st.session_state.data.shape)
            st.write(' - Number of values::', st.session_state.data.shape[0] * st.session_state.data.shape[1])
            st.write(' - Column type:', st.session_state.data.dtypes.value_counts())
            st.write(' - Percentage of missing values:', round(
                sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                        st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                     ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')
# Pydataset datasets
for i, j in zip(PYDATASET_NAMES, PYDATASET_DISPLAY_NAMES):
    if dataset_choix == j:
        st.session_state.data = data(i)
        st.session_state.choice_dataset = "The loaded file is the dataset " + j
        with col1_1:
            message_.success(st.session_state.choice_dataset)

        with col1:
            st.write("##")
            st.markdown('<p class="section">Preview</p>', unsafe_allow_html=True)
            st.write(st.session_state.data.head(50))
            st.write("##")

        with col2:
            st.write("##")
            st.markdown('<p class="section">Features</p>', unsafe_allow_html=True)
            st.write(' - Size:', st.session_state.data.shape)
            st.write(' - Number of values::', st.session_state.data.shape[0] * st.session_state.data.shape[1])
            st.write(' - Column type:', st.session_state.data.dtypes.value_counts())
            st.write(' - Percentage of missing values: ', round(
                sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                        st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                     ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')

if dataset_choix == "Choose a personal dataset":
    with col1_1:
        uploaded_file = st.file_uploader("", type=['csv', 'xls'])
        # uploaded_file = 0
        if uploaded_file is not None:
            st.session_state.file_details = {"FileName": uploaded_file.name,
                                             "FileType": uploaded_file.type,
                                             "FileSize": uploaded_file.size}
            st.success('File ' + st.session_state.file_details['FileName'] + ' loaded successfully !')

    if 'data' in st.session_state:
        del st.session_state.data

    if uploaded_file is None:
        with col1_1:
            st.info("Please load a dataset")

    if "data" not in st.session_state:
        load_data()

    if "data" in st.session_state:
        my_expander = st.expander(label="Preprocessing options")
        with my_expander:
            with col1_1:
                st.write("##")
                st.write("##")
                st.write("##")
                st.session_state.separateur = st.text_input("Separator (optional): ")
            st.write("##")

            load_data()

            st.markdown("<p class='petite_section'>Changes to the dataset : </p>", unsafe_allow_html=True)
            col1_1, b_1, col2_1, c_1, col3_1 = st.columns((1, 0.2, 1, 0.2, 1)) 
            st.write("##")
            option_col_update = st.session_state.data.columns.tolist()

            with col1_1:
                st.session_state.col_to_time = st.multiselect(label='Time Series Conversion',
                                                              options=option_col_update,
                                                              )
            with col2_1:
                st.session_state.col_to_float_money = st.multiselect('Currency Conversion',
                                                                     options=option_col_update,
                                                                     )
            with col3_1:
                st.session_state.col_to_float_coma = st.multiselect('Convert string with commas to float',
                                                                    options=option_col_update,
                                                                    )
            with col1_1:
                st.session_state.drop_col = st.multiselect(label='Drop columns',
                                                           options=option_col_update,
                                                           )

            with col1_1:
                for col in st.session_state["col_to_time"]:
                    try:
                        st.session_state.data[col] = pd.to_datetime(st.session_state.data[col])
                        st.success("Transformation of " + col + " performed !")
                    except:
                        st.error("Transformation impossible or already done")
            with col2_1:
                for col in st.session_state.col_to_float_money:
                    try:
                        st.session_state.data[col] = st.session_state.data[col].apply(clean_data).astype('float')
                        st.success("Transformation of " + col + " performed !")
                    except:
                        st.error("Transformation impossible or already done")
            with col3_1:
                for col in st.session_state.col_to_float_coma:
                    try:
                        st.session_state.data[col] = st.session_state.data[col].apply(
                            lambda x: float(str(x).replace(',', '.')))
                        st.success("Transformation of " + col + " performed !")
                    except:
                        st.error("Transformation impossible or already done")
            with col1_1:
                for col in st.session_state["drop_col"]:
                    try:
                        st.session_state.data = st.session_state.data.drop(columns=col, axis=1)
                        st.success("Column " + col + " deleted !")
                    except:
                        st.error("Transformation impossible or already done")

            with col1:
                st.write("##")
                st.markdown('<p class="section">Preview</p>', unsafe_allow_html=True)
                st.write(st.session_state.data.head(50))
                st.write("##")

            with col2:
                st.write("##")
                st.markdown('<p class="section">Features</p>', unsafe_allow_html=True)
                st.write(' - Size:', st.session_state.data.shape)
                st.write(' - Number of values::', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                st.write(' - Column type:', st.session_state.data.dtypes.value_counts())
                st.write(' - Percentage of missing values:', round(
                    sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                            st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                         ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')
            st.download_button(data=st.session_state.data.to_csv(), label="Download the modified dataset",
                               file_name='dataset.csv')
    st.session_state.choice_dataset = "You have chosen to select your dataset"
    with col1_1:
        message_.success(st.session_state.choice_dataset)
