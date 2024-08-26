import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import gradio as gr
import tempfile
import os

# Fonction pour nettoyer et prétraiter les données
def clean_and_preprocess_data(data):
    data.dropna(inplace=True)
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    cleaned_data = preprocessor.fit_transform(data)
    return cleaned_data, preprocessor

# Fonction pour entraîner et sauvegarder le modèle
def train_and_save_model(data, preprocessor, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    
    joblib.dump(kmeans, 'clustering_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

# Fonction pour charger le modèle et faire des prédictions
def load_and_predict(new_data):
    kmeans = joblib.load('clustering_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    
    cleaned_new_data = preprocessor.transform(new_data)
    predictions = kmeans.predict(cleaned_new_data)
    
    return predictions

# Variable pour stocker le chemin du fichier généré
output_file_path = ""

# Interface Gradio combinée pour l'entraînement et la prédiction
def combined_interface(file_path, n_clusters):
    global output_file_path
    data = pd.read_csv(file_path)
    cleaned_data, preprocessor = clean_and_preprocess_data(data)
    train_and_save_model(cleaned_data, preprocessor, n_clusters)
    
    predictions = load_and_predict(data)
    data['Segment'] = predictions
    
    output_file_path = os.path.join(tempfile.gettempdir(), "segmented_customers.csv")
    data.to_csv(output_file_path, index=False)
    
    return "File has been generated. Click the download button to get the file."

# Fonction pour retourner le fichier généré
def download_file():
    return output_file_path

# Interface Gradio pour l'entraînement et les segmentation
def interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Customer Segmentation Training and Prediction")
        gr.Markdown("Upload a CSV file to train the model and get predictions with segments in the same file. Choose the number of segments.")

        file_input = gr.File(type="filepath", label="Upload CSV File")
        num_segments = gr.Slider(minimum=2, maximum=10, step=1, value=3, label="Number of Segments")
        submit_button = gr.Button("Submit")
        status = gr.Textbox(label="Status")
        download_button = gr.Button("Download Segmented CSV")
        output_file = gr.File(label="Download Segmented CSV")

        submit_button.click(combined_interface, inputs=[file_input, num_segments], outputs=status)
        download_button.click(download_file, outputs=output_file)

        gr.Row([submit_button, download_button])

    demo.launch()

if __name__ == "__main__":
    interface()
