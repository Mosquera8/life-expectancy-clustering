import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Diccionario de textos para soportar varios idiomas
# Diccionario de textos para soportar varios idiomas
texts = {
    'en': {
        'title': 'Life Expectancy Clustering Analysis',
        'sidebar_header': 'Configuration Options',
        'dataset_info': 'Dataset Information',
        'show_head': 'Show first rows of the dataset',
        'rows_columns': 'Number of rows and columns:',
        'select_features': 'Select features for clustering',
        'error_no_features': 'Please select at least one feature to continue.',
        'correlation_matrix': 'Correlation Matrix',
        'elbow_method': 'Elbow Method for Determining Optimal Clusters',
        'num_clusters': 'Select number of clusters (k)',
        'silhouette_index': 'Silhouette Index:',
        'silhouette_explanation': 'The Silhouette Score measures how well samples are clustered. A higher score indicates better-defined clusters.',
        'data_with_clusters': 'Data with Assigned Clusters',
        'cluster_distribution': 'Cluster Distribution',
        'visualization_title': 'Clusters Visualization in Variable Space',
        'x_axis': 'Select variable for X-axis',
        'y_axis': 'Select variable for Y-axis',
        'download_data': 'Download Data with Assigned Clusters',
        'download_button': 'Download CSV',
        'colab_link': 'View the development on Google Colab'
    },
    'es': {
        'title': 'Análisis de Clustering de Expectativa de Vida',
        'sidebar_header': 'Opciones de Configuración',
        'dataset_info': 'Información del Dataset',
        'show_head': 'Mostrar los primeros registros del dataset',
        'rows_columns': 'Número de filas y columnas:',
        'select_features': 'Selecciona las características para el clustering',
        'error_no_features': 'Por favor, selecciona al menos una característica para continuar.',
        'correlation_matrix': 'Matriz de Correlación',
        'elbow_method': 'Método del Codo para Determinar el Número Óptimo de Clusters',
        'num_clusters': 'Selecciona el número de clusters (k)',
        'silhouette_index': 'Índice de silueta:',
        'silhouette_explanation': 'El Índice de Silueta mide qué tan bien están agrupadas las muestras. Un índice mayor indica clusters mejor definidos.',
        'data_with_clusters': 'Datos con Clusters Asignados',
        'cluster_distribution': 'Distribución de Clusters',
        'visualization_title': 'Visualización de Clusters en el Espacio de Variables',
        'x_axis': 'Selecciona la variable para el eje X',
        'y_axis': 'Selecciona la variable para el eje Y',
        'download_data': 'Descargar Datos con Clusters Asignados',
        'download_button': 'Descargar CSV',
        'colab_link': 'Ver el desarrollo en Google Colab'
    }
}

# Selección de idioma
language = st.sidebar.selectbox('Language / Idioma', options=['en', 'es'], index=1)
text = texts[language]

# Cargar datos
data_path = 'LifeExpectancy.csv'  # Ruta al archivo
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error("File not found. Please check the path." if language == 'en' else "No se encontró el archivo de datos. Por favor, verifica la ruta.")
    st.stop()

# Mostrar información básica del dataset
st.title(text['title'])

st.sidebar.header(text['sidebar_header'])

# Muestra un resumen inicial de los datos
st.subheader(text['dataset_info'])
if st.checkbox(text['show_head']):
    st.write(df.head())
st.write(f"**{text['rows_columns']}**", df.shape)

# Selección dinámica de características
features = st.sidebar.multiselect(
    text['select_features'],
    options=df.columns,
    default=['Adult Mortality', ' BMI ', ' HIV/AIDS', 'Income composition of resources', 'Schooling']
)

if not features:
    st.error(text['error_no_features'])
    st.stop()

X = df[features]

# Imputar valores nulos con la media
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Mostrar el mapa de calor de correlación
st.subheader(text['correlation_matrix'])
correlation_matrix = pd.DataFrame(X_imputed, columns=features).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.title(text['correlation_matrix'])
st.pyplot(fig)

# Método del codo para determinar el número óptimo de clusters
st.subheader(text['elbow_method'])
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_title(text['elbow_method'])
ax.set_xlabel('Número de clusters' if language == 'es' else 'Number of clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Seleccionar el número de clusters
optimal_k = st.sidebar.slider(text['num_clusters'], min_value=2, max_value=10, value=4)

# Aplicar KMeans
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Mostrar el DataFrame original con la columna de clusters
st.subheader(text['data_with_clusters'])
st.write(df)

# Visualizar la distribución de los clusters
st.subheader(text['cluster_distribution'])
fig, ax = plt.subplots()
sns.countplot(x='Cluster', data=df, ax=ax)
ax.set_title(text['cluster_distribution'])
st.pyplot(fig)

# Visualizar los clusters en un par de variables seleccionadas
st.subheader(text['visualization_title'])
selected_x_var = st.sidebar.selectbox(text['x_axis'], features)
selected_y_var = st.sidebar.selectbox(text['y_axis'], features)

# Calcular el índice de silueta
silhouette_avg = silhouette_score(X_scaled, clusters)
st.sidebar.write(f"{text['silhouette_index']} {silhouette_avg:.2f}")
st.sidebar.write(text['silhouette_explanation'])

fig, ax = plt.subplots()
sns.scatterplot(x=selected_x_var, y=selected_y_var, hue='Cluster', data=df, palette='viridis', ax=ax)
ax.set_title(f"{text['visualization_title']}: {selected_x_var} vs {selected_y_var}")
st.pyplot(fig)

# Opción para descargar los datos con clusters
st.subheader(text['download_data'])
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label=text['download_button'],
    data=csv,
    file_name='life_expectancy_clusters.csv',
    mime='text/csv',
)

# Enlace a Google Colab
st.markdown(f"[**{text['colab_link']}**](https://colab.research.google.com/drive/1H7__ROCqEAYI91tPsyh5SpLCrdT1czkm?usp=sharing)")
