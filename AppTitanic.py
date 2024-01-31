#Librerías 
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from PIL import Image  

#Funciones:
@st.cache_data   # Esto hará que la carga de datos sea más eficiente
def load_data():
    data = pd.read_csv('Titanic_modificado.csv')
    return data

data = load_data()


# Cargar y mostrar una imagen en el encabezado
image = Image.open('Imagenes\PortadaTitanic.jpg')
st.image(image, width=1000)

#Sidebar
st.sidebar.title('Breve historia del Titanic')
st.sidebar.write('El legendario **RMS Titanic**, que formaba parte de un trío de trasatlánticos británicos lujosos y vanguardistas para la ingeniería del siglo XX, se destacaba por ser el más imponente de los tres y era famoso por su supuesta condición de insumergible.')
st.sidebar.write('Sin embargo, en la madrugada del 15 de abril de 1912, durante su viaje inaugural de Southampton a Nueva York, encontró su trágico destino al chocar con un iceberg y naufragar. Este desastre cobró la vida de 1,496 de las 2,208 personas a bordo.')
st.sidebar.write('El dataset del Titanic recopila información real de 891 pasajeros y es a partir de este que se analiza si factores como la edad, el género o la clase social influyeron en la supervivencia al naufragio')

#App:
st.title('Análisis de la supervivencia en el Titanic')
st.subheader('Introducción')
st.write("""Algunas de los aspectos que se realizarán son: 
* Ver el número de valores nulos 
* Representar el porcentaje de filas con atributos nulos.
* Limpieza de columnas.
* Saber la edad mínima y máxima de las personas del barco.
* Conocer la mediana de las edades.
* Ver los precios (columna `fares`) más altos y bajos.
* Número de pasajeros embarcados (columna `Embarked`).
* Ver la distribución de sexos en las personas embarcadas.
            """
)
variables = pd.DataFrame({
    'Variable': ['PassengerID', 'Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],
    'Descripción': ['Identificador del pasajero. Único por cada pasajero.', 
                    'Indica 1 si el pasajero sobrevivió al naufragio, 0 si no sobrevivió.',
                    'Clase a la que pertenecía el pasajero: 1, 2 o 3.',
                    'Nombre del pasajero','Género del pasajero. 0: hombre, 1: mujer',
                    'Edad del pasajero', 'Número de hermanos, hermanas, hermanastros o hermanastras en el barco.',
                    'Número de padres e hijos en el barco.', 'Identificador del billete',
                    'Precio pagado por el billete','Identificador del camarote asignado al pasajero',
                    'Puerto en el que embarcó. S: Southampton, C:Chebourg, Q: Queenstown']
})
st.table(variables)



st.title('Aplicación del dataset Titanic')

tab1, tab2, tab3 = st.tabs(["Análisis Interactivo", "Algunas Gráficas", "Predicción de Supervivencia"])

with tab1:
    st.header("Análisis Interactivo")

    # Selección del tipo de gráfico
    chart_type = st.selectbox("Selecciona el tipo de gráfico", ['Histograma', 'Gráfico de Barras', 'Gráfico de Dispersión'])

    # Selección de variables
    columns = data.columns.tolist()
    x_axis = st.selectbox("Selecciona la variable para el eje X", columns)

    # Opciones adicionales dependiendo del tipo de gráfico
    if chart_type != 'Histograma':
        y_axis = st.selectbox("Selecciona la variable para el eje Y", columns)

    # Generar y mostrar el gráfico
    if st.button("Generar Gráfico"):
        fig, ax = plt.subplots()

        if chart_type == 'Histograma':
            sns.histplot(data[x_axis], ax=ax, kde=True)
        elif chart_type == 'Gráfico de Barras':
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
        elif chart_type == 'Gráfico de Dispersión':
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)

        st.pyplot(fig)

with tab2:
    st.header("Gráficas de supervivencia por género, clase y rango de edad")
    st.image('Imagenes\SupervivenciaGenero.png', width=500)
    st.image('Imagenes\SupervivenciaClase.png', width=500)
    st.image('Imagenes\SupervivenciaRangoEdad.png')


#Modelo de predicción
with tab3:
    st.header("Predicción de Supervivencia")

    # Cargar el modelo
    with open('modeloRFC_Titanic.pkl', 'rb') as file:
        modelo = pickle.load(file)

    # Inputs del usuario
    Pclass = st.number_input("Clase (1, 2, o 3):", min_value=1, max_value=3, value=1)
    Sex = st.selectbox("Género:", options=[('Masculino', 0), ('Femenino', 1)], format_func=lambda x: x[0])
    Age = st.number_input("Edad:", min_value=0, max_value=100, value=30)
    Fare = st.number_input("Tarifa pagada:", min_value=0.0, step=0.01, value=0.0)
    Embarked = st.selectbox("Puerto de embarque:", options=[('Southampton', 1), ('Chesbourg', 2), ('Queenstown', 3)], format_func=lambda x: x[0])

    if st.button("Predecir"):
        # Convertir las selecciones del usuario a la codificación numérica
        selected_sex = Sex[1]
        selected_embarked = Embarked[1]

        input_data = pd.DataFrame([[Pclass, selected_sex,  Age, Fare, selected_embarked]], 
                          columns=['Pclass', 'Sex',  'Age','Fare', 'Embarked'])        
        # Realizar la predicción
        prediction = modelo.predict(input_data)

        # Mostrar resultados
        if prediction[0] == 1:
            st.success("Resultado de la Predicción: ¡Sobrevivió!")
        else:
            st.error("Resultado de la Predicción: No Sobrevivió.")