import pandas as pd

# Charger le fichier Excel
file_path = 'D:/TELECHARGEMENTS/Beauty Marketing List July 2024 (1).xlsx'
xl = pd.ExcelFile(file_path)

# Lire les tables
df_main = xl.parse(xl.sheet_names[0])
instructions = xl.parse(xl.sheet_names[1])
sources = [xl.parse(sheet) for sheet in xl.sheet_names[2:]]

# Ajouter une colonne "Source" dans df_main
df_main['Source'] = 'Unknown'

# Identifier la source pour chaque email
for idx, source_df in enumerate(sources):
    source_name = xl.sheet_names[2 + idx]
    df_main.loc[df_main['Emails'].isin(source_df['Email']), 'Source'] = source_name



print("Fichier modifié enregistré avec succès.")

# Enregistrez le fichier modifié
output_path = 'D:/ETUDES/Etude Machine Learning Marouan/fichier list contact.xlsx'
df.to_excel(output_path, index=False)

print("Fichier modifié enregistré avec succès.")
