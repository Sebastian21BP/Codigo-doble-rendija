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

# Ajuste de desplazamiento
desplazamiento = 3728.0  # En micrómetros
distancias2 = distancias2.astype(np.float64)
distancias2 -= desplazamiento

# Convertir a metros
distancias2 /= 1e6  # Convertir micrómetros a metros

# Definir la función de ajuste con parámetro extra B
def funcion_ajuste_doble(x, A, C, D, B):
    return A * (np.cos(D * x)) ** 2 * (np.sin(C * x) / (C * x)) ** 2 + B

# Estimación inicial de los parámetros A, C, D y B
p0_doble = [max(voltajes2), np.pi / np.mean(distancias2), np.pi / np.mean(distancias2), min(voltajes2)]

# Intentar el ajuste
try:
    parametros_opt_doble, covarianza_doble = curve_fit(funcion_ajuste_doble, distancias2, voltajes2, p0=p0_doble, maxfev=10000)

    # Obtener los parámetros ajustados
    A_doble, C_doble, D_doble, B_doble = parametros_opt_doble
    perr_doble = np.sqrt(np.diag(covarianza_doble))

    # Mostrar resultados
    print("Parámetros ajustados para la segunda toma de datos:")
    print(f"A = {A_doble:.3f} ± {perr_doble[0]:.3f}")
    print(f"C = {C_doble:.3f} ± {perr_doble[1]:.3f}")
    print(f"D = {D_doble:.3f} ± {perr_doble[2]:.3f}")
    print(f"B = {B_doble:.3f} ± {perr_doble[3]:.3f}")

    # Calcular los valores predichos
    voltajes_predichos_doble = funcion_ajuste_doble(distancias2, *parametros_opt_doble)
    residuales_doble = voltajes2 - voltajes_predichos_doble

    # Graficar los residuales
    plt.figure(figsize=(8, 6))
    plt.plot(distancias2, residuales_doble, 'bo', label="Residuales del Ajuste")
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Residuales del Ajuste (Segunda Toma de Datos)")
    plt.xlabel("Distancia (Metros)")
    plt.ylabel("Residuales (V)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Graficar el ajuste
    plt.figure(figsize=(8, 6))
    plt.plot(distancias2, voltajes2, 'ro', label="Datos Experimentales (DoubleSlit)")
    plt.plot(distancias2, voltajes_predichos_doble, 'g-', label="Ajuste de Curva")
    plt.title("Ajuste de Curva al patrón de interferencia")
    plt.xlabel("Distancia (m)")
    plt.ylabel("Voltaje (V)")
    plt.grid(True)
    plt.legend()
    plt.show()

except RuntimeError as e:
    print(f"Error en el ajuste: {e}")

