import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Cargar los datos desde el archivo
archivo = 'C:/Users/Usuario/Desktop/DatosDobleRendija.xlsx'
df = pd.read_excel(archivo, sheet_name='SingleSlit')
df1 = pd.read_excel(archivo, sheet_name='DoubleSlit')


# Función para convertir a tipo float
def convertir_a_float(series):
    return pd.to_numeric(series, errors='coerce').to_numpy()


# Cargar los datos de corriente y voltaje y convertir a float
voltajes1 = convertir_a_float(df['Voltios'])
distancias1 = convertir_a_float(df['distancia'])
distancias1 *= 10  # Convertir a micrómetros
voltajes2 = convertir_a_float(df1['Voltaje (V)'])
distancias2 = convertir_a_float(df1['Distancia (*10micras)'])
distancias2 *= 10  # Convertir a micrómetros
desplazamiento = 2500.0 -58.0 # Ejemplo: 5 micrómetros, cambia este valor al que conozcas

# Restar el desplazamiento a las distancias
distancias1 = distancias1.astype(np.float64)

# Restar el desplazamiento a las distancias
distancias1 -= desplazamiento
distancias1/=1e6
# Definir la función de ajuste con solo tres parámetros: A, B y C
def funcion_ajuste(x, A, B ,C):
    # Para evitar división por cero, agregamos un pequeño valor al denominador
    denominador = B * x  # Eliminamos C, ya que la centramos en el origen
#    denominador = np.where(np.abs(denominador) < 1e-10, 1e-10, denominador)  # Reemplazar valores pequeños por 1e-10
    return (A * (np.sin(B * x))**2) / (denominador)**2 + C


# Estimación inicial de los parámetros
p0 = [max(voltajes1), np.pi*1e-4/670e-9, min(voltajes1)]  # Valores iniciales para A y B, sin C

# Intentar el ajuste con un mayor número de evaluaciones
try:
    parametros_opt, covarianza = curve_fit(funcion_ajuste, distancias1, voltajes1, p0=p0, maxfev=10000)

    # Obtener los parámetros ajustados
    A, B , C= parametros_opt

    # Evaluar la covarianza para ver la calidad del ajuste
    perr = np.sqrt(np.diag(covarianza))

    # Mostrar los parámetros ajustados y su incertidumbre
    print("Parámetros ajustados:")
    print(f"A = {A:.3f} ± {perr[0]:.3f}")
    print(f"B = {B:.3f} ± {perr[1]:.3f}")
    # Calcular los valores predichos por el modelo
    voltajes_predichos = funcion_ajuste(distancias1, *parametros_opt)

    # Calcular los residuales (diferencia entre valores observados y predichos)
    residuales = voltajes1 - voltajes_predichos

    # Graficar los residuales
    plt.figure(figsize=(8, 6))
    plt.plot(distancias1, residuales, 'bo', label="Residuales")
    plt.axhline(0, color='r', linestyle='--')  # Línea horizontal en 0
    plt.title("Residuales del Ajuste")
    plt.xlabel("Distancia (Metros)")
    plt.ylabel("Residuales (V)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Crear el gráfico 1 con el ajuste
    plt.figure(figsize=(8, 6))
    plt.plot(distancias1, voltajes1, 'bo', label="Datos Experimentales (SingleSlit)")
    plt.plot(distancias1, voltajes_predichos, 'r-', label="Ajuste de Curva")

    # Personalizar el gráfico 1
    plt.title("Patrón de Difracción (Ajuste de Curva Centrado en el Origen)")
    plt.xlabel("Distancia (metros)")
    plt.ylabel("Voltaje (V)")
    plt.grid(True)
    plt.legend()

    # Crear el gráfico 2
    plt.figure(figsize=(8, 6))
    plt.plot(distancias2, voltajes2, 'ro', label="Voltaje vs Distancia (DoubleSlit)")

    # Personalizar el gráfico 2
    plt.title("Patrón de Interferencia")
    plt.xlabel("Distancia (Micrómetros)")
    plt.ylabel("Voltaje (V)")
    plt.grid(True)
    plt.legend()

    # Mostrar los gráficos
    plt.show()

except RuntimeError as e:
    print(f"Error en el ajuste: {e}")