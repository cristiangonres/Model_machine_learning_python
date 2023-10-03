import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

array_undi = np.array([1, 2, 3, 4, 5])
array_bidi = np.array([[1, 2, 3], [1, 2, 3]])
array_tridi = np.array([[[1, 2],[3, 4]], [[5, 6],[7, 8]]])


print(array_tridi.shape, array_tridi.size, array_tridi.dtype, array_tridi.ndim, type(array_tridi))

df = pd.DataFrame(array_bidi)
print(df)

unos = np.ones((4, 3))
print(unos)

ceros = np.zeros(((2, 4, 3)))
print(ceros)

cien = np.arange(0, 101, 5)
print(cien)

# Creamos un array de números aleatorios enteros comprendidos en entre 0 y 10, de tamaño (2, 5)
cerodiez = np.random.randint(0, 10, (2, 5))
print(cerodiez)

# Creamos un array de números aleatorios decimales comprendidos en entre 0 y 1, de tamaño (3, 5)
cerouno = np.random.random((3, 5))
print(cerouno)

# Establecemos la "semilla" de números aleatorios en 27
np.random.seed(27)

# Creamos un array de números aleatorios enteros comprendidos en entre 0 y 10, de tamaño (3, 5)
array_cerodiez = np.random.randint(0, 10, (3, 5))
print(array_cerodiez)

# Encontramos los valores únicos del array_4
np.unique(array_cerodiez)

# Extraemos el elemento de índice 1 del array_4
array_cerodiez[1]

# Extraemos las primeras dos filas del array_4
array_cerodiez[:2]

# Extraemos los dos primeros datos de las primeras dos filas del array_4
array_cerodiez[:2][:2]

# Creamos dos arrays de tamaño 3x4: uno relleno de números aleatorios entre 0 y 10, y otro relleno de unos
array_cerodiez_2 = np.random.randint(0, 10, (3, 4))
array_unos = np.ones((3, 4))
array_cerodiez_2
array_unos

# Sumamos los dos arrays
array_cerodiez_2 + array_unos

# Creamos ahora un array de tamaño (4,3) lleno de unos
array_unos_2 = np.ones((4, 3))

# Entonces crearemos otro array de tamaño (4,3) lleno de unos
array_unos_3 = np.ones((4, 3))

# Restamos el array_8 al array_7
array_unos_2 - array_unos_3

# Creamos otros dos arrays de tamaño 3x3 con números aleatorios del 1 al 5
array_unocinco_1 = np.random.randint(1, 6, (3, 3))
array_unocinco_2 = np.random.randint(1, 6, (3, 3))


# Multiplicamos los últimos dos arrays entre sí
array_unocinco_2 * array_unocinco_1

# Elevamos el array_9 al cuadrado
np.power(array_unocinco_2, 2)

# Buscamos la raíz cuadrada del array_10
np.sqrt(array_unocinco_2)

# Hallamos el promedio de los valores del array_9
np.mean(array_unocinco_2)

# Hallamos el valor máximo de los valores del array_9
np.max(array_unocinco_2)

# Hallamos el valor mínimo de los valores del array_9
np.min(array_unocinco_2)

# Cambiamos la forma del array_9 por una de 9x1, y lo almacenamos como array_11
array_11 = array_unocinco_2.reshape((9, 1))

# Transponemos el array_11
array_11.T

# Comparamos el array_9 y el array_10, para saber cuáles elementos del array_9 son mayores a los del array_10
array_comparacion = array_unocinco_2 > array_unocinco_1 
array_comparacion

# Veamos sus nuevos tipos de datos
array_comparacion.dtype

# Alguno de los elementos del array_9 es igual su equivalente del array_10?
array_unocinco_2 == array_unocinco_1 

# Comparamos nuevamente ambos arrays, en esta ocasión con >=

array_unocinco_2 >= array_unocinco_1 

# Buscamos los elementos del array_9 que son mayores a 2
array_unocinco_2 > 2

# Ordenamos de menor a mayor los elementos dentro del array_9
np.sort(array_unocinco_2)

# Importamos Pandas
import pandas as pd

# Creamos una serie de números y hallamos su media
numeros = pd.Series([1, 8, 19, 23, 31, 32, 33, 60, 100])
numeros.mean()

# Hallamos la suma de dichos números
numeros.sum()

# Creamos una SERIE de tres colores diferentes
colores = pd.Series(['amarillo', 'rojo', 'azul', 'negro'])

# Visualizamos la serie creada
colores

# Creamos una serie con tipos de autos, y la visualizamos
autos = pd.Series(['Ford', 'Mercedes', 'Porche', 'Seat'])

# Combinamos las series de tipos de autos y colores en un DATAFRAME
combi = pd.DataFrame({'Marca': autos, 'Color': colores})
combi

# Conectamos el cuaderno actual con nuestro Drive
from google.colab import drive
drive.mount('/content/drive')

# Importar "ventas-autos.csv" y convertirlo en un nuevo DATAFRAME
ventas_autos = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ventas-autos.csv')
ventas_autos

# Exportar el Dataframe como un archivo CSV a mi carpeta "/content/drive/MyDrive/Colab Notebooks/pruebas/"
ventas_autos.to_csv("/content/drive/MyDrive/Colab Notebooks/pruebas/archivo.csv")

# Analicemos los tipos de datos disponibles en el dataset de ventas autos
ventas_autos.dtypes

# Apliquemos estadística descriptiva (cantidad de valores, media, desviación estándar, valores mínimos y máximos, cuartiles) al dataset
ventas_autos.describe()

# Obtenemos información del dataset utilizando info()
ventas_autos.info()

# Listamos los nombres de las columnas de nuestro dataset
ventas_autos.columns

# Averiguamos el "largo" de nuestro dataset
len(ventas_autos)

# Mostramos las primeras 5 filas del dataset
ventas_autos.head()

# Mostramos las primeras 7 filas del dataset
ventas_autos.head(7)

# Mostramos las últimas 5 filas del dataset
ventas_autos.tail()

# Utilizamos .loc para seleccionar la fila de índice 3 del DataFrame
ventas_autos.loc[3]

# Utilizamos .iloc para seleccionar las filas 3, 7 y 9
ventas_autos.iloc[[3, 7, 9]]

# Seleccionar la columna "Kilometraje"
ventas_autos['Kilometraje']

# Encontrar el valor medio de la columnas "Kilometraje"
ventas_autos['Kilometraje'].mean()

# Seleccionar aquellas columnas que tengan valores superiores a 100,000 kilómetros en la columna Kilometraje
ventas_autos[ventas_autos['Kilometraje'] > 100000]

# Creamos una tabla cruzada de doble entrada entre Fabricante y cantidad de puertas
crosstab = pd.crosstab(ventas_autos['Fabricante'], ventas_autos['Puertas'])
crosstab


# Agrupamos las columnas por fabricante y buscandos el valor medio de las columnas numéricas
ventas_autos.groupby(["Fabricante"]).mean()

# Importamos Matplotlib y creamos un gráfico con los valores de la columna Kilometraje
import matplotlib as plt
%matplotlib inline

ventas_autos['Kilometraje'].plot()

# Puede que un gráfico más apropiado en este caso sea un histograma?
ventas_autos['Kilometraje'].hist()

# Intentamos graficar la columna de precios
ventas_autos

# Elimina la puntuación de la columna de precios
ventas_autos['Precio (USD)'].replace('\D', '', regex=True).astype(int).plot()

# Importamos el módulo de Matplotlib como plt
import matplotlib.pyplot as plt

# La siguiente linea nos permite ver los gráficos directamente al ejecutarlos en el notebook
%matplotlib inline

# Creamos un gráfico utilizando plt.plot()
plt.plot()

# Graficomos una lista de números
n = [1, 4, 6, 7, 8, 92]
plt.plot(n)

# Creamos dos listas, x e y. Llenamos a la lista x de valores del 1 al 100.
x = list(range(101))
y = []


# Los valores de y van a equivaler al cuadrado del respectivo valor en x con el mísmo índice
for numero in x:
    y.append(numero**2)
    
# Graficamos ambas listas creadas
plt.plot(x, y)

# Creamos el gráfico utilizando plt.subplots()
fig, ax = plt.subplots()
ax.plot(x, y)

# Importar y preparar la librería
import matplotlib.pyplot as plt
%matplotlib inline
# Preparar los datos
x = list(range(101))
y = []
for n in x:
  y.append(n ** 2)

# Preparamos el área del gráfico (fig) y el gráfico en sí (ax) utilizando plt.subplots()
fig, ax = plt.subplots()

# Añadimos los datos al gráfico
ax.plot(x, y)

# Personalizamos el gráfico añadiendo título al gráfico y a los ejes x e y
ax.set(title = "Casos de muerte súbita en España en 2022", xlabel = "Días", ylabel = "Decesos")

# Guardamos nuestro gráfico empleando fig.savefig()
fig.savefig("Ejemplo gráfico decesos 2022")

#creamos un nuveo set de datos utilizando la librería Numpy
import numpy as np
x_1 = np.linspace(0, 100, 10)
y_1 = x_1 ** 2

# Creamos el gráfico de dispersión de x vs y
fig, ax = plt.subplots()
ax.scatter(x_1, y_1)

# Visualizamos ahora la función seno, utilizando np.sin(X)
fig, ax = plt.subplots()
x_2 = np.linspace(-10, 10, 100)
y_2 = np.sin(x_2)
ax.scatter(x_2, y_2)

# Creemos un diccionario con tres platos y su respectivo precio
# Las claves del diccionario serán los nombres de las comidas, y los valores asociados, su precio
comidas = {"pollo": 800, "jamón": 150, "trufado": 600}

# Crearemos un gráfico de barras donde el eje x está formado por las claves del diccionario,
# y el eje y contiene los valores.
fig, ax = plt.subplots()
ax.bar(comidas.keys(), comidas.values())

# Añadimos los títulos correspondientes
ax.set(title = "Menu comidas", xlabel = "Platos", ylabel = "Precios")

# Probemos a continuación con un gráfico de barras horizontales
fig, ax = plt.subplots()
ax.barh(list(comidas.keys()), list(comidas.values()))

# Añadimos los títulos correspondientes
ax.set(title = "Menu comidas", ylabel = "Platos", xlabel = "Precios")

# Creamos una distribución de 1000 valores aleatorios distribuidos normalmente
x = np.random.randn(1000)
y = x * 3.1416
fig, ax = plt.subplots()
# Creamos el histograma
ax.hist(x)


# Creamos una figura con 4 subgráficos (2 por fila)
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,5))

# Creamos la misma disposición de gráficos, con un tamaño de figura de 10x5
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,5))

# Para nuestro primer gráfico, tomamos el conjunto x_1, y_1, y generamos un gráfico de líneas
ax1.plot(x_1, y_1)


# Para nuestro segundo gráfico, tomamos el conjunto x_2, y_2, y generamos un gráfico de dispersión
ax2.scatter(x_2, y_2)

# Creamos un gráfico con los precios de tres comidas en la esquina inferior izquierda
ax3.bar(comidas.keys(), comidas.values())

# El gráfico de la esquina inferior derecha será un histograma de valores aleatorios con distribución normal
ax4.hist(np.random.randn(100))

# Verificamos estilos disponibles
plt.style.available


# Cambiamos el estilo predeterminado por "seaborn-whitegrid"
plt.style.use('seaborn-v0_8-darkgrid')

# Copiamos los valores de los gráficos anteriores
# Creamos la misma disposición de gráficos, con un tamaño de figura de 10x5
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,5))

# Para nuestro primer gráfico, tomamos el conjunto x_1, y_1, y generamos un gráfico de líneas
ax1.plot(x_1, y_1, color="#fba012")

# Para nuestro segundo gráfico, tomamos el conjunto x_2, y_2, y generamos un gráfico de dispersión
ax2.scatter(x_2, y_2, color="#fba012")

# Creamos un gráfico con los precios de tres comidas en la esquina inferior izquierda
ax3.bar(comidas.keys(), comidas.values(), color="#fba012")

# El gráfico de la esquina inferior derecha será un histograma de valores aleatorios con distribución normal
ax4.hist(np.random.randn(100), color="#fba012")

