import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import gradio as gr

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

# Entraîner et sauvegarder le modèle
def train_and_save_model(data, preprocessor, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    
    joblib.dump(kmeans, 'clustering_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

# Charger le modèle et faire des segments
def load_and_predict(new_data):
    kmeans = joblib.load('clustering_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    
    cleaned_new_data = preprocessor.transform(new_data)
    predictions = kmeans.predict(cleaned_new_data)
    
    return predictions

# Interface gradio pour l'entraînement et la prédiction
def combined_interface(file_path, n_clusters):
    data = pd.read_csv(file_path)
    cleaned_data, preprocessor = clean_and_preprocess_data(data)
    train_and_save_model(cleaned_data, preprocessor, n_clusters)
    
    predictions = load_and_predict(data)
    data['Segment'] = predictions
    
    output_file = "segmented_customers.csv"
    data.to_csv(output_file, index=False)
    
    return output_file

# CSS personnalisé pour la personnalisation
custom_css = """
body {
    font-family: Arial, sans-serif;
    display: flex;
}

#sidebar {
    width: 220px;
    background-color: #f8f9fa;  /* Couleur initiale */
    padding: 10px;
    position: fixed;
    top: 0;
    bottom: 0;
    left: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
}

#logo {
    height: 170px;  
    margin-bottom: 20px;
}

#menu {
    width: 100%;
}

#menu button {
    width: 100%;
    margin-bottom: 10px;
    background-color: #f8f9fa;  
    color: black;  /* Texte noir */
    border: none;
    padding: 10px;
    font-size: 16px;
    cursor: pointer;
}

#menu button:hover {
    background-color: #e0e0e0;  /* Couleur de survol */
}

#main {
    margin-left: 240px;  /* Marge ajustée */
    padding: 20px;
    width: calc(100% - 240px);  /* Largeur ajustée */
}



#description {
    text-align: center;
    margin: 40px 20px;
    font_size: 80;
    width: 100%;
    
}

#footer {
    text-align: center;
    margin-top: 20px;
    font-size: 14px;
}
"""

# Interface gradio pour l'entraînement et la segmentation
def create_interface():
    logo_url = "D:/ETUDES/Etude Machine Learning Marouan/Segmentation client/logo.png"  # Remplacez par le chemin de votre logo

    with gr.Blocks(css=custom_css) as interface:
        with gr.Column(elem_id="sidebar"):
            gr.Image(logo_url, elem_id="logo", show_label=False)
            with gr.Column(elem_id="menu"):
                gr.Markdown("## Menu")
                gr.Button("Bases de Données")
                gr.Button("Visuals")
                gr.Button("Buyer Personas")
                gr.Button("Recommandations")

        with gr.Column(elem_id="main"):
            gr.Markdown("<div id='description'>Customer Segmentation with IA</div>")
            gr.Markdown("<div id='description'>Upload a CSV file from your computer to get the segments, and choose the number of segments.</div>")
            file_input = gr.File(label="Select CSV file", type="filepath")
            slider = gr.Slider(minimum=2, maximum=10, step=1, value=3, label="Number of Segments")
            output_file = gr.File(label="Download Segmented File")

            def process_file(file_path, n_clusters):
                return combined_interface(file_path, n_clusters)

            gr.Button("Run").click(process_file, inputs=[file_input, slider], outputs=output_file)
        
        gr.Markdown("<div id='footer'>Developed by Your Name</div>")

    return interface

if __name__ == "__main__":
    # Lancer l'interface gradio
    create_interface().launch()
