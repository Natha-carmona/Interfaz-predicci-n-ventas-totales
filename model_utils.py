import pandas as pd
import numpy as np
import os
import pickle
import traceback

# Ruta predeterminada del modelo entrenado
MODEL_PATH = "modelo_demanda.pkl"

class DemandForecaster:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_mock = True
        self.cargar_modelo()

    def cargar_modelo(self):
        """Intenta cargar el modelo entrenado desde el disco."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_mock = False
                print(f"[INFO] Modelo cargado con éxito desde {self.model_path}")
            except Exception as e:
                print(f"[ERROR] Error al cargar el modelo: {str(e)}")
                traceback.print_exc()
                self.is_mock = True
        else:
            print(f"[WARN] No se encontró un modelo en '{self.model_path}'. Se utilizará el motor de simulación para demostración.")
            self.is_mock = True

    def preprocesar_datos(self, df_input):
        """
        Asegura que las variables requeridas estén presentes y tengan el formato adecuado.
        Variables: ['Vendedor', 'Fec Factura', 'Nombre del solicitante', 
                   'Descripción de material (producto)', 'Cantidad facturada', 
                   'Ofc. Venta', 'Período contable', 'Pobl. Destino', 
                   'Canal de Distribución', 'Hora facturación', 'Departamento', 'Mes']
        """
        df = df_input.copy()
        
        # Conversión de fechas
        if 'Fec Factura' in df.columns:
            df['Fec Factura'] = pd.to_datetime(df['Fec Factura'], errors='coerce')
            
        # Asegurar tipos de datos categóricos y numéricos básicos
        num_cols = ['Cantidad facturada', 'Mes', 'Período contable']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        return df

    def predecir(self, df_input):
        """
        Realiza el pronóstico de demanda utilizando el modelo entrenado
        o un algoritmo predictivo de respaldo basado en patrones históricos.
        """
        df_processed = self.preprocesar_datos(df_input)
        
        # Si el modelo real fue cargado con éxito
        if not self.is_mock and self.model is not None:
            try:
                # Aquí debes adaptar las columnas exactas que tu modelo espera
                # (e.g. One-Hot Encoding, eliminación de columnas de texto libre o fechas, etc.)
                # Por ejemplo, seleccionamos las columnas que coincidan con la estructura entrenada:
                columnas_entrenamiento = [
                    'Vendedor', 'Ofc. Venta', 'Período contable', 'Pobl. Destino', 
                    'Canal de Distribución', 'Hora facturación', 'Departamento', 'Mes'
                ]
                
                # Ejemplo de extracción/codificación simple (adaptar según tu modelo específico)
                X = pd.get_dummies(df_processed[columnas_entrenamiento], drop_first=True)
                
                # Predicción del modelo de Scikit-Learn o equivalente
                predicciones = self.model.predict(X)
                return np.maximum(0, predicciones)  # Evitar demandas negativas
            except Exception as e:
                print(f"[ERROR] Falló la predicción con el modelo real: {str(e)}. Utilizando simulación de respaldo.")
        
        # --- MOTOR DE SIMULACIÓN DE RESPALDO (MOCK) ---
        # Si no hay modelo real cargado, generamos una predicción realista basada en las variables ingresadas
        np.random.seed(42)
        predicciones = []
        
        for _, row in df_processed.iterrows():
            # Generar una base simulada basada en el Departamento y el Mes (estacionalidad básica)
            base_demanda = 100.0
            
            # Estacionalidad por mes (picos a fin de año)
            mes = int(row.get('Mes', 6))
            estacionalidad = 1.0 + 0.3 * np.sin(2 * np.pi * mes / 12) + (0.5 if mes in [11, 12] else 0.0)
            
            # Factores categóricos simulados
            factor_canal = 1.2 if str(row.get('Canal de Distribución', '')).lower() in ['mayorista', 'online', 'directo'] else 0.9
            factor_vendedor = 1.0 + (len(str(row.get('Vendedor', ''))) % 5) * 0.05
            
            # Efecto del producto (variabilidad de longitud del nombre como semilla de aleatoriedad)
            factor_producto = 1.0 + (len(str(row.get('Descripción de material (producto)', ''))) % 10) * 0.1
            
            # Cálculo final con un ligero componente de ruido aleatorio
            prediccion = base_demanda * estacionalidad * factor_canal * factor_vendedor * factor_producto
            ruido = np.random.normal(0, prediccion * 0.05) # 5% de variabilidad aleatoria
            
            predicciones.append(max(0, round(prediccion + ruido, 2)))
            
        return np.array(predicciones)

def generar_datos_ejemplo(n_filas=100):
    """Genera un DataFrame sintético con las 12 variables del modelo para pruebas."""
    np.random.seed(42)
    
    vendedores = ['Vendedor Juan', 'Vendedora Maria', 'Vendedor Carlos', 'Vendedora Ana']
    productos = ['Laptop Pro 15', 'Monitor UltraWide 34', 'Teclado Mecánico RGB', 'Mouse Ergonómico Inalámbrico', 'Silla Executive Grey']
    solicitantes = ['Tech Corp S.A.', 'Global Solutions Inc.', 'Distribuidora Nova', 'Tiendas del Norte', 'Suministros Industriales']
    oficinas = ['Oficina Norte', 'Oficina Sur', 'Oficina Centro', 'Oficina Virtual']
    poblaciones = ['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena']
    canales = ['Canal Directo', 'Distribuidor', 'E-Commerce', 'Retail']
    departamentos = ['Tecnología', 'Oficina', 'Mobiliario', 'Accesorios']
    
    fechas = pd.date_range(start='2025-01-01', end='2025-12-31', periods=n_filas)
    
    data = {
        'Vendedor': np.random.choice(vendedores, n_filas),
        'Fec Factura': fechas,
        'Nombre del solicitante': np.random.choice(solicitantes, n_filas),
        'Descripción de material (producto)': np.random.choice(productos, n_filas),
        'Cantidad facturada': np.random.randint(5, 150, n_filas),
        'Ofc. Venta': np.random.choice(oficinas, n_filas),
        'Período contable': 2025,
        'Pobl. Destino': np.random.choice(poblaciones, n_filas),
        'Canal de Distribución': np.random.choice(canales, n_filas),
        'Hora facturación': [f"{np.random.randint(8, 18):02d}:{np.random.randint(0, 59):02d}:00" for _ in range(n_filas)],
        'Departamento': np.random.choice(departamentos, n_filas),
        'Mes': fechas.month
    }
    
    return pd.DataFrame(data)
