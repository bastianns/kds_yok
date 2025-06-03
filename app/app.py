import os
import streamlit as st
import pandas as pd
import joblib
import torch
import json
from custom_utils import ProteinRNN

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(APP_DIR), 'models')
# * <---------------------------- Conf starts here ----------------->
try:
    hyperparams_path = os.path.join(MODELS_DIR, 'rnn_hyperparams.json')
    with open(hyperparams_path, 'r') as f:
        RNN_HYPERPARAMS = json.load(f)
    VOCAB_SIZE = RNN_HYPERPARAMS.get('VOCAB_SIZE', 25)
    EMBEDDING_DIM = RNN_HYPERPARAMS.get('EMBEDDING_DIM', 100)
    HIDDEN_DIM = RNN_HYPERPARAMS.get('HIDDEN_DIM', 128)
    OUTPUT_DIM = RNN_HYPERPARAMS.get('OUTPUT_DIM', 5)
    N_LAYERS = RNN_HYPERPARAMS.get('N_LAYERS', 2)
    BIDIRECTIONAL = RNN_HYPERPARAMS.get('BIDIRECTIONAL', True)
    DROPOUT_P = RNN_HYPERPARAMS.get('DROPOUT_P', 0.3)
    PADDING_IDX = 0;
    MAX_SEQ_LENGTH = RNN_HYPERPARAMS.get('MAX_SEQ_LENGTH', 300)

except FileNotFoundError:
    st.error("rnn_hyperparams.json not found! RNN model cannot be instantiated correctly. Please define hyperparameters manually if the file is missing.")
    VOCAB_SIZE = 25 ; EMBEDDING_DIM = 100; HIDDEN_DIM = 128; OUTPUT_DIM = 5;
    N_LAYERS = 2; BIDIRECTIONAL = True; DROPOUT_P = 0.3; PADDING_IDX = 0; MAX_SEQ_LENGTH = 300;

try:
    if hasattr(torch, 'classes') and hasattr(torch.classes, '__path__'):
        torch.classes.__path__ = []
except Exception:
    pass

# * <---------------------------- Conf ends here ----------------->

# * <---------------------------- LOGIC starts here ----------------->
def load_model_and_dependencies(model_name):
    model = None
    preprocessor = None
    label_encoder = None
    try:
        label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder_custom.pkl')
        label_encoder = joblib.load(label_encoder_path)
        if model_name == "Random Forest":
            model = joblib.load(os.path.join(MODELS_DIR,'best_rf_model.pkl'))
            preprocessor = joblib.load(os.path.join(MODELS_DIR,'physicochemical_pipeline.pkl'))
            st.success("Loaded Random Forest model, physicochemical pipeline, and label encoder.")
        elif model_name == "Gradient Boosting (XGBoost)":
            model = joblib.load(os.path.join(MODELS_DIR,'best_xgb_model.pkl'))
            preprocessor = joblib.load(os.path.join(MODELS_DIR,'physicochemical_pipeline.pkl'))
            st.success("Loaded XGBoost model, physicochemical pipeline, and label encoder.")
        elif model_name == "RNN (LSTM/GRU)":
            model = ProteinRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
                               N_LAYERS, BIDIRECTIONAL, DROPOUT_P, PADDING_IDX)
            model.load_state_dict(torch.load(os.path.join(MODELS_DIR,'best_model_rnn.pth'), map_location=torch.device('cpu')))
            model.eval()  # ! IMPORTANT: Set model to evaluation mode
            # ? Load RNN-specific preprocessing components
            char_to_int_map = joblib.load(os.path.join(MODELS_DIR,'char_to_int.pkl'))
            preprocessor = {'char_to_int': char_to_int_map, 'max_seq_len': MAX_SEQ_LENGTH, 'padding_idx': PADDING_IDX}
            st.success("Loaded RNN model, tokenizer map, and label encoder.")

    except FileNotFoundError as e:
        st.error(f"Error: A required model or preprocessor file not found: {e}. Please ensure all .pkl and .pth files are in the correct path.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred during loading: {e}")
        return None, None, None

    return model, preprocessor, label_encoder

def preprocess_input_data(df_input, model_name, preprocessor_data):
    if model_name in ["Random Forest", "Gradient Boosting (XGBoost)"]:
        if preprocessor_data is None:
            st.error("Physicochemical preprocessor pipeline not loaded.")
            return None
        physicochemical_pipeline = preprocessor_data
        expected_physchem_cols = [
            'Massa_Molecular', 'Ponto_Isoelétrico', 'Hidrofobicidade', 'Carga_Total',
            'Proporção_Polar', 'Proporção_Apolar', 'Comprimento_Sequência'
        ]
        missing_cols = [col for col in expected_physchem_cols if col not in df_input.columns]
        if missing_cols:
            st.error(f"Missing expected columns in uploaded CSV for physicochemical models: {', '.join(missing_cols)}")
            return None
        try:
            df_features_to_process = df_input[expected_physchem_cols]
            processed_features = physicochemical_pipeline.transform(df_features_to_process)
            return processed_features
        except Exception as e:
            st.error(f"Error during physicochemical preprocessing: {e}")
            return None

    elif model_name == "RNN (LSTM/GRU)":
        if preprocessor_data is None:
            st.error("RNN preprocessing components (tokenizer, max_len) not loaded.")
            return None
        if 'Sequência' not in df_input.columns:
            st.error("Uploaded CSV for RNN model is missing the 'Sequência' column.")
            return None
        char_to_int_map = preprocessor_data['char_to_int']
        max_len = preprocessor_data['max_seq_len']
        pad_token_id = preprocessor_data.get('padding_idx', 0)

        sequences_str = df_input['Sequência'].tolist()
        processed_sequences = []
        for seq_str in sequences_str:
            tokenized_sequence = [char_to_int_map.get(char, pad_token_id) for char in str(seq_str)] # Ensure seq_str is string
            padded_sequence = tokenized_sequence[:max_len]
            padded_sequence = padded_sequence + [pad_token_id] * (max_len - len(padded_sequence))
            processed_sequences.append(padded_sequence)

        return torch.tensor(processed_sequences, dtype=torch.long)
    return None


def make_predictions(model, processed_data, model_name):
    if model is None or processed_data is None:
        return None
    try:
        if model_name in ["Random Forest", "Gradient Boosting (XGBoost)"]:
            predictions_numerical = model.predict(processed_data)
        elif model_name == "RNN (LSTM/GRU)":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            processed_data = processed_data.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(processed_data)
                _, predictions_numerical = torch.max(outputs, 1)
                predictions_numerical = predictions_numerical.cpu().numpy()
        return predictions_numerical
    except Exception as e:
        st.error(f"Error during prediction with {model_name}: {e}")
        return None

# * <---------------------------- LOGIC ends here ----------------->

# * <---------------------------- UI starts here ----------------->
st.set_page_config(page_title="Protein Function Predictor", layout="wide")
st.title("Protein Function Predictor")
st.write("Hi ges")

st.markdown("""
This application predicts the function of synthetic proteins based on their sequences and physicochemical properties.
Upload a CSV file with protein data to get started.
""")

# ? sidebar
st.sidebar.header("Controls")

# ? models
model_option = st.sidebar.selectbox(
    "Choose a prediction model:",
    ("Random Forest", "Gradient Boosting (XGBoost)", "RNN (LSTM/GRU)")
)

# ? upload file
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# ? main
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview (First 5 Rows)")
        st.dataframe(input_df.head())

        model, preprocessor, label_encoder = load_model_and_dependencies(model_option)

        if model and preprocessor and label_encoder:
            if st.button(f"Predict Protein Functions using {model_option}"):
                with st.spinner("Preprocessing data and making predictions..."):
                    processed_input = preprocess_input_data(input_df.copy(), model_option, preprocessor)

                    if processed_input is not None:
                        numerical_predictions = make_predictions(model, processed_input, model_option)

                        if numerical_predictions is not None:
                            try:
                                string_predictions = label_encoder.inverse_transform(numerical_predictions)
                                results_df = input_df.copy() # ?    original data
                                results_df['Predicted_Function'] = string_predictions
                                st.subheader("Prediction Results")
                                st.dataframe(results_df)

                                # ? history placeholder
                                if 'prediction_history' not in st.session_state:
                                    st.session_state.prediction_history = []
                                st.session_state.prediction_history.append({
                                    "model_used": model_option,
                                    "file_name": uploaded_file.name,
                                    "predictions_summary": results_df[['ID_Proteína', 'Predicted_Function']].head().to_dict('records') # Example summary
                                })

                                # ?  results_df to CSV for download
                                csv_results = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Prediction Results as CSV",
                                    data=csv_results,
                                    file_name=f"predictions_{uploaded_file.name}",
                                    mime='text/csv',
                                )
                            except Exception as e:
                                st.error(f"Error converting predictions to class names: {e}. Are you sure the label encoder is correctly loaded and predictions are numerical?")
                        else:
                            st.error("Could not make predictions. Please check model and data.")
                    else:
                        st.error("Could not preprocess the input data. Please ensure the CSV format is correct and columns match.")
        else:
            st.warning(f"Could not load the selected model '{model_option}' or its dependencies. Prediction functionality is disabled.")

    except Exception as e:
        st.error(f"An error occurred processing the uploaded file: {e}")
else:
    st.info("Awaiting CSV file upload...")

# ?   displaying model performance graphics placeholder
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance (Illustrative)")

# ?  prediction history display placehoder
if 'prediction_history' in st.session_state and st.session_state.prediction_history:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recent Prediction History")
    for i, entry in enumerate(reversed(st.session_state.prediction_history[-3:])):
        st.sidebar.markdown(f"**{i+1}. File:** {entry['file_name']}, **Model:** {entry['model_used']}")
        st.sidebar.json(entry['predictions_summary']) # Displaying a few predictions
else:
    st.sidebar.text("No prediction history yet.")

st.sidebar.markdown("---")
st.sidebar.info(f"Current time in Bandung: {pd.Timestamp.now(tz='Asia/Jakarta').strftime('%Y-%m-%d %H:%M:%S %Z')}")
# * <---------------------------- UI ends here ----------------->
