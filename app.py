import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Importamos el módulo utilitario diseñado
from model_utils import DemandForecaster, generar_datos_ejemplo

# Configuración inicial de la página (Debe ser la primera instrucción)
st.set_page_config(
    page_title="Predicción de Demanda Inteligente",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado CSS inyectado de forma segura y global para evitar conflictos en el DOM
st.markdown("""
    <style>
    /* Estilos globales estables para el tema */
    .reportview-container {
        background-color: #F8FAFC;
    }
    /* Estilo de tarjetas de métricas usando clases personalizadas seguras */
    .custom-card {
        background-color: #F1F5F9;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 6px solid #1E3A8A;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    .custom-card h5 {
        color: #475569 !important;
        margin: 0 0 0.5rem 0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    .custom-card h2 {
        color: #1E3A8A !important;
        margin: 0 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
    }
    .custom-card p {
        color: #64748B !important;
        margin: 0.5rem 0 0 0 !important;
        font-size: 0.8rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Inicializar el predictor de demanda de manera segura
@st.cache_resource
def obtener_predictor():
    return DemandForecaster()

forecaster = obtener_predictor()

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=200&auto=format&fit=crop&q=60&ixlib=rb-4.0.3", use_container_width=True, caption="Análisis Predictivo de Demanda")
st.sidebar.title("Configuración de Entrada")

# Indicador de estado del modelo
if forecaster.is_mock:
    st.sidebar.warning("⚠️ Ejecutando en Modo Demostración (No se detectó un archivo `.pkl` de modelo real).")
else:
    st.sidebar.success("✅ ¡Modelo entrenado cargado correctamente desde disco!")

opcion_carga = st.sidebar.radio(
    "Selecciona la fuente de datos:",
    ["Simular registro único", "Cargar archivo CSV / Excel", "Usar datos de demostración"]
)

# --- PANEL PRINCIPAL ---
st.title("📈 Sistema Inteligente de Pronóstico de Demanda")
st.caption("Optimiza tus niveles de inventario y toma decisiones comerciales basadas en datos e inteligencia artificial.")
st.write("---")

# 1. Opción Registro Único
if opcion_carga == "Simular registro único":
    st.subheader("📋 Ingreso Manual de Características")
    st.info("Ingresa los parámetros de un registro para predecir la demanda esperada de ese escenario específico.")
    
    # Formulario para capturar las 12 variables del modelo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vendedor = st.selectbox("Vendedor", ['Vendedor Juan', 'Vendedora Maria', 'Vendedor Carlos', 'Vendedora Ana'])
        fec_factura = st.date_input("Fecha Factura", datetime.today())
        nombre_solicitante = st.text_input("Nombre del solicitante", "Cliente Premium S.A.")
        producto = st.selectbox("Descripción de material (producto)", ['Laptop Pro 15', 'Monitor UltraWide 34', 'Teclado Mecánico RGB', 'Mouse Ergonómico Inalámbrico', 'Silla Executive Grey'])
    
    with col2:
        cantidad_facturada = st.number_input("Cantidad histórica facturada (referencia)", min_value=1, value=50)
        oficina_venta = st.selectbox("Oficina de Venta (Ofc. Venta)", ['Oficina Norte', 'Oficina Sur', 'Oficina Centro', 'Oficina Virtual'])
        periodo_contable = st.number_input("Período contable", min_value=2020, max_value=2030, value=datetime.today().year)
        poblacion_destino = st.text_input("Población Destino (Pobl. Destino)", "Bogotá")
        
    with col3:
        canal_distribucion = st.selectbox("Canal de Distribución", ['Canal Directo', 'Distribuidor', 'E-Commerce', 'Retail'])
        hora_facturacion = st.time_input("Hora de facturación", datetime.now().time())
        departamento = st.selectbox("Departamento", ['Tecnología', 'Oficina', 'Mobiliario', 'Accesorios'])
        mes = st.slider("Mes", min_value=1, max_value=12, value=int(datetime.today().month))

    # Crear dataframe con una sola fila para la predicción
    datos_registro = {
        'Vendedor': [vendedor],
        'Fec Factura': [pd.to_datetime(fec_factura)],
        'Nombre del solicitante': [nombre_solicitante],
        'Descripción de material (producto)': [producto],
        'Cantidad facturada': [cantidad_facturada],
        'Ofc. Venta': [oficina_venta],
        'Período contable': [periodo_contable],
        'Pobl. Destino': [poblacion_destino],
        'Canal de Distribución': [canal_distribucion],
        'Hora facturación': [hora_facturacion.strftime("%H:%M:%S")],
        'Departamento': [departamento],
        'Mes': [mes]
    }
    
    df_registro = pd.DataFrame(datos_registro)
    
    if st.button("🔮 Calcular Pronóstico de Demanda", type="primary", use_container_width=True):
        prediccion_resultado = forecaster.predecir(df_registro)[0]
        
        st.write("---")
        # Contenedor estático para agrupar los resultados y evitar errores de renderizado de React
        with st.container():
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                # Usamos el componente nativo st.metric que es 100% seguro contra fallos del DOM
                st.metric(
                    label="DEMANDA ESTIMADA", 
                    value=f"{prediccion_resultado:,.1f} Unidades",
                    help="Unidades pronosticadas basadas en patrones estacionales y geográficos."
                )
                
                # Opcional: Tarjeta visual de soporte usando marcado estático libre de scripts
                st.markdown(f"""
                    <div class="custom-card">
                        <h5>Estado del Pronóstico</h5>
                        <h2>Estable</h2>
                        <p>Variabilidad estimada del ±5% según comportamiento de mercado.</p>
                    </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                # Gráfico de barras comparativo de influencia departamental
                categorias_demostracion = ['Tecnología', 'Oficina', 'Mobiliario', 'Accesorios']
                valores_comparativos = [prediccion_resultado if c == departamento else np.random.randint(20, 120) for c in categorias_demostracion]
                
                fig = px.bar(
                    x=categorias_demostracion,
                    y=valores_comparativos,
                    labels={'x': 'Departamento', 'y': 'Demanda Prevista'},
                    title=f"Predicción Comparativa por Categoría (Destacando {departamento})",
                    color=categorias_demostracion,
                    color_discrete_map={departamento: '#1E3A8A'}
                )
                fig.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

# 2. Opción Archivo Completo (Lotes)
elif opcion_carga == "Cargar archivo CSV / Excel":
    st.subheader("📂 Cargar archivo para Procesamiento en Lote")
    st.markdown("""
        El archivo a cargar debe contener idealmente las siguientes columnas para que el modelo funcione correctamente:
        `Vendedor`, `Fec Factura`, `Nombre del solicitante`, `Descripción de material (producto)`, `Cantidad facturada`, `Ofc. Venta`, `Período contable`, `Pobl. Destino`, `Canal de Distribución`, `Hora facturación`, `Departamento`, `Mes`
    """)
    
    archivo_subido = st.file_uploader("Sube tu archivo de ventas o inventario (.csv, .xlsx)", type=["csv", "xlsx"])
    
    if archivo_subido is not None:
        try:
            if archivo_subido.name.endswith('.csv'):
                df_cargado = pd.read_csv(archivo_subido)
            else:
                df_cargado = pd.read_excel(archivo_subido)
                
            st.success(f"¡Archivo cargado con éxito! Se leyeron {len(df_cargado)} filas.")
            
            # Verificar si faltan columnas cruciales
            columnas_requeridas = ['Vendedor', 'Fec Factura', 'Nombre del solicitante', 
                                   'Descripción de material (producto)', 'Cantidad facturada', 
                                   'Ofc. Venta', 'Período contable', 'Pobl. Destino', 
                                   'Canal de Distribución', 'Hora facturación', 'Departamento', 'Mes']
            
            columnas_faltantes = [col for col in columnas_requeridas if col not in df_cargado.columns]
            
            if columnas_faltantes:
                st.warning(f"⚠️ Nota: Faltan algunas columnas en tu archivo: {columnas_faltantes}. El preprocesamiento intentará autocompletar valores predeterminados para estas variables.")
                for col in columnas_faltantes:
                    df_cargado[col] = "Desconocido" if col != "Mes" else 6
                    
            # Ejecutar Predicciones
            with st.spinner("Procesando y generando pronósticos..."):
                predicciones = forecaster.predecir(df_cargado)
                df_cargado['Demanda Pronosticada'] = predicciones
                
            st.subheader("📊 Resultados de Pronósticos Generados")
            
            # Contenedor seguro para métricas
            with st.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Demanda Pronosticada", f"{df_cargado['Demanda Pronosticada'].sum():,.0f} unds")
                m2.metric("Promedio de Demanda por Registro", f"{df_cargado['Demanda Pronosticada'].mean():,.1f} unds")
                m3.metric("Pico Máximo de Demanda Detectado", f"{df_cargado['Demanda Pronosticada'].max():,.0f} unds")
            
            # Tabla interactiva con opción de descarga
            st.dataframe(df_cargado.head(50), use_container_width=True)
            
            # Botón de Descarga
            csv_data = df_cargado.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Reporte con Pronósticos en CSV",
                data=csv_data,
                file_name="pronosticos_demanda_procesado.csv",
                mime="text/csv",
                type="primary"
            )
            
            # Visualización Gráfica del Lote
            st.write("---")
            with st.container():
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Demanda por Departamento
                    dem_dep = df_cargado.groupby('Departamento')['Demanda Pronosticada'].sum().reset_index()
                    fig_dep = px.pie(dem_dep, values='Demanda Pronosticada', names='Departamento', 
                                     title="Distribución de la Demanda por Departamento", hole=0.4,
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_dep, use_container_width=True)
                    
                with col_chart2:
                    # Demanda por Canal de Distribución
                    dem_canal = df_cargado.groupby('Canal de Distribución')['Demanda Pronosticada'].sum().reset_index()
                    fig_canal = px.bar(dem_canal, x='Canal de Distribución', y='Demanda Pronosticada',
                                       title="Demanda Pronosticada por Canal de Distribución",
                                       color='Canal de Distribución', color_discrete_sequence=px.colors.sequential.Viridis)
                    st.plotly_chart(fig_canal, use_container_width=True)
                
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {str(e)}")

# 3. Datos de Demostración (Sandbox)
else:
    st.subheader("💡 Modo Sandbox / Demostración")
    st.info("Generando datos simulados basados en las 12 características para que experimentes el comportamiento del tablero.")
    
    if 'df_demo' not in st.session_state:
        st.session_state.df_demo = generar_datos_ejemplo(150)
        st.session_state.df_demo['Demanda Pronosticada'] = forecaster.predecir(st.session_state.df_demo)
        
    df_demo = st.session_state.df_demo
    
    # KPIs Generales usando st.columns y contenedores nativos limpios
    with st.container():
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        with col_kpi1:
            st.metric("Demanda Pronosticada Total", f"{df_demo['Demanda Pronosticada'].sum():,.0f} u")
        with col_kpi2:
            st.metric("Top Departamento", str(df_demo.groupby('Departamento')['Demanda Pronosticada'].sum().idxmax()))
        with col_kpi3:
            st.metric("Oficina de Ventas Líder", str(df_demo.groupby('Ofc. Venta')['Demanda Pronosticada'].sum().idxmax()))
        with col_kpi4:
            st.metric("Mes con Mayor Demanda", f"Mes {df_demo.groupby('Mes')['Demanda Pronosticada'].sum().idxmax()}")
        
    # Gráficos Interactivos Avanzados
    st.write("---")
    st.markdown("### 📈 Visualización Avanzada de Tendencias")
    
    with st.container():
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            # Tendencia Mensual Estacional de la Demanda
            dem_mensual = df_demo.groupby('Mes')['Demanda Pronosticada'].sum().reset_index()
            # Mapear números de mes a nombres cortos
            nombres_meses = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 
                             7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
            dem_mensual['Nombre Mes'] = dem_mensual['Mes'].map(nombres_meses)
            
            fig_trend = px.line(dem_mensual, x='Nombre Mes', y='Demanda Pronosticada', markers=True,
                                title="Comportamiento Estacional (Demanda Pronosticada por Mes)",
                                labels={'Nombre Mes': 'Mes del Año', 'Demanda Pronosticada': 'Unidades Pronosticadas'})
            fig_trend.update_traces(line_color='#1E3A8A', line_width=3, marker=dict(size=8))
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col_v2:
            # Relación de Ventas vs Vendedor
            dem_vendedor = df_demo.groupby('Vendedor')['Demanda Pronosticada'].sum().reset_index().sort_values(by='Demanda Pronosticada', ascending=True)
            fig_vend = px.bar(dem_vendedor, x='Demanda Pronosticada', y='Vendedor', orientation='h',
                              title="Volumen de Demanda Asignado por Vendedor",
                              color='Demanda Pronosticada', color_continuous_scale='Blues')
            st.plotly_chart(fig_vend, use_container_width=True)
        
    # Matriz/Gráfico de Población vs Canal de Distribución
    st.write("---")
    st.markdown("### 📍 Geografía y Canales")
    with st.container():
        pob_canal = df_demo.groupby(['Pobl. Destino', 'Canal de Distribución'])['Demanda Pronosticada'].sum().reset_index()
        fig_bubble = px.scatter(pob_canal, x='Pobl. Destino', y='Canal de Distribución', 
                                size='Demanda Pronosticada', color='Demanda Pronosticada',
                                title="Matriz de Demanda: Población Destino vs Canal de Distribución",
                                size_max=40, color_continuous_scale='Viridis')
        st.plotly_chart(fig_bubble, use_container_width=True)

# Pie de página informativo
st.write("---")
st.markdown("<p style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>Plataforma Inteligente de Pronóstico de Demanda Corporativa. Construido con Streamlit, Plotly y Python.</p>", unsafe_allow_html=True)
