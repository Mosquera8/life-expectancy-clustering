import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar datos
data_path = 'LifeExpectancy.csv'  # Cambia esto a la ruta correcta
df = pd.read_csv(data_path)

# Seleccionar características
features = ['Adult Mortality', ' BMI ', ' HIV/AIDS', 'Income composition of resources', 'Schooling']
X = df[features]

# Imputar valores nulos con la media
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Aplicación de Streamlit
st.title('Análisis de Clustering de Expectativa de Vida')
st.write("""
Esta aplicación realiza un análisis de clustering sobre el conjunto de datos de expectativa de vida. 
Selecciona el número de clusters y explora los resultados visualmente.
""")

st.subheader('Explicación de las Columnas Utilizadas')
st.write("""
Para la creación de los clusters, utilizamos las siguientes columnas:
- **Adult Mortality**: Mortalidad adulta.
- **BMI**: Índice de masa corporal.
- **HIV/AIDS**: Tasa de mortalidad por VIH/SIDA.
- **Income composition of resources**: Composición de los ingresos de los recursos.
- **Schooling**: Muertes infantiles.
Estas características fueron seleccionadas para identificar patrones en los datos que puedan ser útiles para el análisis de la expectativa de vida.
""")

# Mostrar el mapa de calor de correlación
st.subheader('Matriz de Correlación')
st.write("""
La matriz de correlación muestra la relación entre las diferentes variables seleccionadas. Un valor cercano a 1 o -1 indica una fuerte correlación positiva o negativa, respectivamente, mientras que un valor cercano a 0 indica poca o ninguna correlación.
""")
correlation_matrix = pd.DataFrame(X_imputed, columns=features).corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.title('Matriz de Correlación')
st.pyplot(fig)

# Determinar el número óptimo de clusters usando el método del codo
st.subheader('Método del Codo para Determinar el Número Óptimo de Clusters')
st.write("""
El método del codo se utiliza para determinar el número óptimo de clusters. El gráfico muestra la suma de los cuadrados dentro del cluster (WCSS) en función del número de clusters. El punto donde la tasa de disminución se reduce drásticamente se considera el número óptimo de clusters.
""")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss)
ax.set_title('Método del Codo')
ax.set_xlabel('Número de clusters')
ax.set_ylabel('WCSS')  # Within-Cluster Sum of Squares
st.pyplot(fig)

# Seleccionar el número de clusters
optimal_k = st.slider('Selecciona el número de clusters (k)', min_value=2, max_value=10, value=4)

# Aplicar KMeans
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Mostrar el DataFrame original con la columna de clusters
st.subheader('Datos con Clusters Asignados')
st.write("""
En la siguiente tabla se muestran los datos originales con una columna adicional que indica el cluster asignado a cada registro. Esto permite identificar a qué grupo pertenece cada observación según las características seleccionadas.
""")
st.write(df)

# Visualizar la distribución de los clusters
st.subheader('Distribución de Clusters')
st.write("""
Este gráfico muestra la cantidad de observaciones en cada cluster. Nos ayuda a entender cuántas observaciones han sido agrupadas en cada cluster, proporcionando una idea sobre el tamaño y balance de los grupos formados.
""")
fig, ax = plt.subplots()
sns.countplot(x='Cluster', data=df, ax=ax)
st.pyplot(fig)

# Visualizar los clusters en un par de variables seleccionadas
st.subheader('Visualización de Clusters en el Espacio de Variables')
st.write("""
A continuación, puedes seleccionar dos variables para visualizar cómo se distribuyen los clusters en el espacio de esas variables. Esto proporciona una visualización intuitiva de cómo se agrupan las observaciones según las características seleccionadas.
""")
selected_x_var = st.selectbox('Selecciona la variable para el eje X', features)
selected_y_var = st.selectbox('Selecciona la variable para el eje Y', features)

fig, ax = plt.subplots()
sns.scatterplot(x=selected_x_var, y=selected_y_var, hue='Cluster', data=df, palette='viridis', ax=ax)
st.pyplot(fig)

# Opción para descargar los datos con clusters
st.subheader('Descargar Datos con Clusters Asignados')
st.write("""
Puedes descargar los datos con los clusters asignados en formato CSV.
""")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar CSV",
    data=csv,
    file_name='life_expectancy_clusters.csv',
    mime='text/csv',
)