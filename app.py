import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import io

# --- Configuração Inicial da Página ---
st.set_page_config(page_title="Visualizador de LOG's Motronic 1.5.4", layout="wide", initial_sidebar_state="expanded")

# --- Inicialização do Estado (Navegação) ---
if 'view' not in st.session_state:
    st.session_state.view = 'dashboard'  # Controla a tela atual: 'dashboard' ou 'comunidade'
if 'log_selecionado' not in st.session_state:
    st.session_state.log_selecionado = None

# --- Mapeamento das 76 Colunas exatas geradas pelo C++ ---
COLUNAS = [
    "RTM (s)", "ID_Modulo", "MAP (V)", "MAP (Kg/h)", "MAP (kPa)", 
    "CTS (V)", "CTS (°C)", "IAT (V)", "IAT (°C)", "TPS (V)", 
    "Bateria (V)", "Sonda (mV)", "RPM", "RPM_Alvo", "VSS (km/h)", 
    "Tempo_Inj_Banco (ms)", "Tempo_Inj_Ciclo (ms)", "Avanço (°)", "Atraso_Detonacao (%)", 
    "TPS (%)", "Canister (%)", "EGR (%)", "Tempo_Carga (ms)", 
    "Vazao_Ar_Atual (Kg/h)", "Vazao_Ar_Alvo (Kg/h)", "IAC (Passos)", "IAC_BLM (Passos)", 
    "IAC_Integrador", "Sonda_Integrador", "Sonda_BLM_Lenta", "Sonda_BLM_Parcial", 
    "Flag_VSS", "Flag_RPM", "Flag_ParkNeutral", "Flag_TorqueCtrl", 
    "Flag_TPS_Lenta", "Flag_TPS_Plena", "Flag_AC_Pressao", "Flag_AC_Botao", 
    "Flag_Diag_Rqst", "Flag_Malha_Fechada", "Flag_EGR_Ativa", "Flag_Knock", 
    "Flag_AC_Embreagem", "Flag_Bomba_Comb", "Flag_Bomba_Ar", "Flag_Check_Engine", 
    "Flag_Mistura", "Flag_MotorCil", "Flag_Transmissao", "Flag_Imob_Rec", 
    "Flag_Imob_Act", "Flag_Ventoinha", 
    "AlphaCode", "Codigo_GM", "NumFalhas",
    "Err_Code_0", "Err_Stat_0", "Err_Code_1", "Err_Stat_1", "Err_Code_2", "Err_Stat_2",
    "Err_Code_3", "Err_Stat_3", "Err_Code_4", "Err_Stat_4", "Err_Code_5", "Err_Stat_5",
    "Err_Code_6", "Err_Stat_6", "Err_Code_7", "Err_Stat_7",
    "Consumo_Medio (km/L)", "Consumo_Inst (L/h)", "Distancia_Total (km)", "Versao_HW"
]

# --- Configuração dos Limites (Min/Max) Exatos para Motronic ---
# Adicionado TODOS os sensores analógicos e calculados para aparecerem nos gráficos
LIMITES_SENSORES = {
    "MAP (V)": (0.0, 5.0),
    "MAP (Kg/h)": (0.0, 300.0),
    "MAP (kPa)": (10, 105),
    "CTS (V)": (0.0, 5.0),
    "CTS (°C)": (0, 120),
    "IAT (V)": (0.0, 5.0),
    "IAT (°C)": (0, 100),
    "TPS (V)": (0.0, 5.0),
    "Bateria (V)": (8.0, 16.0),
    "Sonda (mV)": (0, 1000),
    "RPM": (0, 7500),
    "RPM_Alvo": (0, 3000),
    "VSS (km/h)": (0, 240),
    "Tempo_Inj_Banco (ms)": (0.0, 30.0),
    "Tempo_Inj_Ciclo (ms)": (0.0, 30.0),
    "Avanço (°)": (0, 45),
    "Atraso_Detonacao (%)": (0, 100),
    "TPS (%)": (0, 100),
    "Canister (%)": (0, 100),
    "EGR (%)": (0, 100),
    "Tempo_Carga (ms)": (0.0, 20.0),
    "Vazao_Ar_Atual (Kg/h)": (0.0, 300.0),
    "Vazao_Ar_Alvo (Kg/h)": (0.0, 300.0),
    "IAC (Passos)": (0, 255),
    "IAC_BLM (Passos)": (0, 255),
    "IAC_Integrador": (0, 255),
    "Sonda_Integrador": (0, 255),
    "Sonda_BLM_Lenta": (0, 255),
    "Sonda_BLM_Parcial": (0, 255),
    "Consumo_Medio (km/L)": (0.0, 30.0),
    "Consumo_Inst (L/h)": (0.0, 30.0),
    "Distancia_Total (km)": (0.0, 1000.0)
}

# --- Tabela de DTCs Motronic 1.5.4 ---
DTC_TABLE = {
    10: "Codificacao Var. Nao Programada",
    13: "Sonda Lambda - Circuito Aberto",
    14: "Sensor Temp. Agua - Tensao Baixa",
    15: "Sensor Temp. Agua - Tensao Alta",
    16: "Circuito do Sinal de Detonacao",
    18: "Modulo de Controle de Detonacao",
    19: "Sinal Incorreto de RPM",
    21: "Sensor Posicao Borboleta - Tensao Alta",
    22: "Sensor Posicao Borboleta - Tensao Baixa",
    24: "Sem Sinal Velocidade (VSS)",
    25: "Falha no Injetor 1 - Tensao Alta",
    26: "Falha no Injetor 2 - Tensao Alta",
    27: "Falha no Injetor 3 - Tensao Alta",
    28: "Falha no Injetor 4 - Tensao Alta",
    29: "Falha no Injetor 5 - Tensao Alta",
    31: "Falta de Sinal de Rotacao do Motor",
    32: "Falha no Injetor 6 - Tensao Alta",
    33: "Valvula EGR - Tensao Baixa",
    34: "Valvula EGR - Tensao Alta",
    35: "Rele Partida Frio - Tensao Baixa",
    37: "Rele Partida Frio - Tensao Alta",
    38: "Sonda Lambda - Tensao Baixa (Pobre)",
    39: "Sonda Lambda - Tensao Alta (Rica)",
    44: "Mistura Pobre - Valor Constante",
    45: "Mistura Rica - Valor Constante",
    48: "Tensao Baixa da Bateria",
    49: "Tensao Alta da Bateria",
    52: "Lampada de Avaria - Tensao Alta",
    53: "Rele Bomba Combustivel - Tensao Baixa",
    54: "Rele Bomba Combustivel - Tensao Alta",
    55: "Falha na Unidade de Comando",
    56: "Controle de Ar Marcha-Lenta - Tensao Alta",
    57: "Controle de Ar Marcha-Lenta - Tensao Baixa",
    61: "Valvula de Ventilacao - Tensao Baixa",
    62: "Valvula de Ventilacao - Tensao Alta",
    69: "Sensor Temp. Ar - Tensao Baixa",
    71: "Sensor Temp. Ar - Tensao Alta",
    73: "Sensor MAP/MAF - Tensao Baixa",
    74: "Sensor MAP/MAF - Tensao Alta",
    75: "Circuito Controle Torque - Tensao Baixa",
    76: "Controle de Torque Continuo",
    77: "Rele Vent. Baixa - Tensao Baixa",
    78: "Rele Vent. Baixa - Tensao Alta",
    81: "Falha no Injetor 1 - Tensao Baixa",
    82: "Falha no Injetor 2 - Tensao Baixa",
    83: "Falha no Injetor 3 - Tensao Baixa",
    84: "Falha no Injetor 4 - Tensao Baixa",
    85: "Falha no Injetor 5 - Tensao Baixa",
    86: "Falha no Injetor 6 - Tensao Baixa",
    87: "Rele Corte A/C - Tensao Baixa",
    88: "Rele Corte A/C - Tensao Alta",
    93: "Sensor Hall (Fase) - Tensao Baixa",
    94: "Sensor Hall - Tensao Alta",
    97: "Sinal Contr. Tracao - Tensao Alta",
    119: "Sensor MAP - Valor Incorreto Partida",
    125: "Sensor MAP - Abaixo do limite minimo",
    126: "Sensor MAP - Acima do Limite maximo",
    129: "Sinal Valvula EGR - Tensao Baixa",
    131: "Sinal Valvula EGR - Tensao Alta",
    132: "Valvula EGR - Sinal Incorreto",
    135: "Lampada de Avaria - Tensao Baixa",
    136: "Erro Saida Nao Reconhecido",
    137: "Alta Temp. Caixa ECU",
    138: "Sensor MAP - Tensao Baixa",
    139: "Sensor MAP - Tensao Alta",
    143: "Imobilizador - Falha na codificacao",
    144: "Imobilizador - Falta de Sinal",
    145: "Imobilizador - Sinal Incorreto",
    146: "Sensor Rotacao - Tensao Baixa",
    147: "Sensor Rotacao - Tensao Alta",
    171: "Rele Ventoinha - Tensao Baixa",
    172: "Rele Ventoinha - Tensao Alta",
    174: "Pressao A/C - Tensao Baixa",
    175: "Pressao A/C - Tensao Alta"
}

DTC_STATUS = {
    1: "MEMORIZADO",
    2: "PRESENTE",
    3: "INTERMITENTE"
}

# --- FUNÇÃO: Ler Planilha de Logs Públicos ---
@st.cache_data(ttl=60)
def carregar_lista_logs_publicos():
    sheet_id = "1UhlGeITGM2ZmyVfuZZl9aWh4TxQjO-S69eDZ01z2gzs"
    url_planilha = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        df = pd.read_csv(url_planilha)
        num_colunas = len(df.columns)
        
        if num_colunas >= 16:
            df = df.iloc[:, :16] # Blinda contra colunas extras fantasma
            df.columns = [
                "Data/Hora", "ID", "Duração", "Usuário", "Veículo", "Comentário", "Obs_Moderador", 
                "Status_Geral", "Tipo_Trajeto", "F_Engasgo", "F_Partida", "F_Potencia", 
                "F_MarchaLenta", "F_Apagando", "F_Consumo", "ID_Arquivo"
            ]
        # Preenche os NaNs
        df = df.fillna("")
                
        return df
    except Exception as e:
        st.error(f"Erro ao carregar lista de comunidade: {e}")
        return pd.DataFrame()

# --- FUNÇÃO: Carregamento de Dados (Local ou Nuvem) ---
@st.cache_data
def carregar_dados(arquivo_ou_url, colunas):
    try:
        if isinstance(arquivo_ou_url, str) and arquivo_ou_url.startswith("http"):
            resposta = requests.get(arquivo_ou_url)
            resposta.raise_for_status()
            conteudo = io.StringIO(resposta.text)
            df = pd.read_csv(conteudo, sep="|", header=None, names=colunas)
        else:
            if hasattr(arquivo_ou_url, 'seek'):
                arquivo_ou_url.seek(0)
            df = pd.read_csv(arquivo_ou_url, sep="|", header=None, names=colunas)
            
        df["RTM (s)"] = pd.to_numeric(df["RTM (s)"], errors="coerce")
        df = df.dropna(subset=["RTM (s)"]).copy()
        df = df.sort_values(by="RTM (s)").reset_index(drop=True)
        
        if len(df) > 1:
            diferencas = df["RTM (s)"].diff()
            if diferencas.head(10).max() > 10:
                idx_salto = diferencas.head(10).idxmax()
                df = df.iloc[idx_salto:].reset_index(drop=True)
        
        counts = df.groupby("RTM (s)")["RTM (s)"].transform('count')
        cumcounts = df.groupby("RTM (s)").cumcount()
        df["RTM_Continuo"] = df["RTM (s)"] + (cumcounts / counts)
        
        df["Tempo_Relogio"] = pd.to_datetime(df["RTM_Continuo"], unit='s')
        
        return df
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return None

# ==========================================
# BARRA LATERAL (MENU DE NAVEGAÇÃO)
# ==========================================
with st.sidebar:
    st.markdown("<p style='text-align: center; font-size: 15px; font-weight: bold; margin-top: 10px; color: #cccccc;'>Visualizador de LOG's<br>Motronic 1.5.4 DashBoard</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.header("Navegação")
    if st.button("📊 Arquivo Local", use_container_width=True):
        st.session_state.view = 'dashboard'
        st.rerun()

    if st.button("🌐 LOG's da Comunidade", use_container_width=True):
        st.session_state.view = 'comunidade'
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.view == 'dashboard':
        st.header("📂 Enviar Arquivo Log")
        arquivo_local = st.file_uploader("Selecione o arquivo .TXT", type=["txt"])
        
        if arquivo_local:
            try:
                conteudo = arquivo_local.getvalue().decode('utf-8', errors='ignore')
                linhas = [l for l in conteudo.split('\n') if l.strip()]
                
                if not linhas:
                    st.error("❌ O arquivo selecionado está vazio.")
                    st.session_state.log_selecionado = None
                else:
                    ultima_linha = linhas[-1].split('|')
                    
                    if len(ultima_linha) < 76:
                        st.error("❌ Arquivo incompatível! Faltam parâmetros da injeção Motronic.")
                        st.session_state.log_selecionado = None
                    else:
                        st.session_state.log_selecionado = arquivo_local
            except Exception as e:
                st.error("❌ Erro ao tentar ler a assinatura do arquivo. Arquivo corrompido.")
                st.session_state.log_selecionado = None
        else:
            if st.session_state.log_selecionado is not None and not isinstance(st.session_state.log_selecionado, str):
                st.session_state.log_selecionado = None

    st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# ÁREA PRINCIPAL DO APLICATIVO
# ==========================================

# ----------------------------------------------------
# TELA 1: GALERIA DA COMUNIDADE (Lista Ampla)
# ----------------------------------------------------
if st.session_state.view == 'comunidade':
    st.title("LOG's da Comunidade Motronic 1.5.4")
    st.write("Clique no botão à esquerda da linha de registro do Log que deseja visualizar.")
    
    df_publicos = carregar_lista_logs_publicos()
    
    if not df_publicos.empty:
        event = st.dataframe(
            df_publicos,
            column_order=["Data/Hora", "Duração", "Usuário", "Veículo", "Comentário", "Obs_Moderador"],
            column_config={
                "Data/Hora": st.column_config.TextColumn("Data de Registo", width=130),
                "Duração": st.column_config.TextColumn("Duração do Registo", width=130),
                "Usuário": st.column_config.TextColumn("Enviado por", width=150),
                "Veículo": st.column_config.TextColumn("Modelo", width=250),
                "Comentário": st.column_config.TextColumn("Observações do Utilizador", width=550),
                "Obs_Moderador": st.column_config.TextColumn("Observações do Moderador", width=750),
                "ID": None, "Status_Geral": None, "Tipo_Trajeto": None,
                "F_Engasgo": None, "F_Partida": None, "F_Potencia": None,
                "F_MarchaLenta": None, "F_Apagando": None, "F_Consumo": None, "ID_Arquivo": None
            },
            hide_index=True,
            use_container_width=True, 
            on_select="rerun",
            selection_mode="single-row",
            height=600
        )
        
        if len(event.selection.rows) > 0:
            idx = event.selection.rows[0]
            id_arq = df_publicos.iloc[idx]['ID_Arquivo']
            st.session_state.log_selecionado = f"https://drive.google.com/uc?export=download&id={id_arq}"
            st.session_state.view = 'dashboard'
            st.rerun()
            
    else:
        st.warning("Nenhum log público foi encontrado ou a base de dados encontra-se vazia.")

# ----------------------------------------------------
# TELA 2: DASHBOARD E GRÁFICOS (Visão Principal)
# ----------------------------------------------------
elif st.session_state.view == 'dashboard':
    if st.session_state.log_selecionado is not None:
        df = carregar_dados(st.session_state.log_selecionado, COLUNAS)
        
        if df is not None and not df.empty:
            versao_dash = str(df["Versao_HW"].iloc[-1])

            aba1, aba2, aba3, aba4, aba5 = st.tabs([
                "📊 Visão Geral", 
                "📈 Telemetria (Gráficos)", 
                "⚠️ Diagnóstico (Scanner)", 
                "📋 Dados Brutos",
                "📖 Glossário"
            ])

            with aba1:
                st.success(f"Log carregado com sucesso! (Dashboard v{versao_dash} | {len(df)} registos)")
                
                try:
                    alpha = str(df["AlphaCode"].iloc[-1]).strip()
                    gm_code = str(df["Codigo_GM"].iloc[-1]).strip()
                    nome_modulo = f"Módulo GM: {gm_code} | AlphaCode: {alpha}"
                except:
                    nome_modulo = "Módulo Desconhecido"
                    
                st.info(f"**Identificação da ECU:** {nome_modulo}")
                
                st.subheader("Resumo do Percurso")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("RPM Máximo", f"{df['RPM'].max():.0f} RPM")
                col2.metric("Temp Máxima Água", f"{df['CTS (°C)'].max():.0f} °C")
                col3.metric("Distância Percorrida", f"{df['Distancia_Total (km)'].iloc[-1]:.2f} km")
                col4.metric("Velocidade Máxima", f"{df['VSS (km/h)'].max():.0f} km/h")

                st.markdown("---")
                st.subheader("Médias de Funcionamento")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Tensão Média Bateria", f"{df['Bateria (V)'].mean():.2f} V")
                col_b.metric("Avanço Médio", f"{df['Avanço (°)'].mean():.1f} °")
                col_c.metric("Sonda Lambda Média", f"{df['Sonda (mV)'].mean():.0f} mV")
                col_d.metric("MAP Médio", f"{df['MAP (kPa)'].mean():.1f} Kg/h")

            with aba2:
                colunas_analogicas = list(LIMITES_SENSORES.keys())
                colunas_flags = [c for c in df.columns if c.startswith("Flag_")]
                
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    selecionados_analog = st.multiselect(
                        "Sensores Analógicos:", 
                        options=colunas_analogicas, 
                        default=["RPM", "MAP (kPa)", "Sonda (mV)", "TPS (%)", "VSS (km/h)", "CTS (°C)"]
                    )
                with col_sel2:
                    selecionados_flags = st.multiselect(
                        "Sinais Digitais / Flags (ON/OFF):", 
                        options=colunas_flags, 
                        default=["Flag_Malha_Fechada", "Flag_Knock", "Flag_AC_Embreagem"]
                    )

                if selecionados_analog or selecionados_flags:
                    fig = go.Figure()
                    cores = px.colors.qualitative.Plotly
                    layout_updates = {}
                    
                    tem_analog = len(selecionados_analog) > 0
                    tem_flags = len(selecionados_flags) > 0
                    
                    if tem_analog:
                        for idx, sensor in enumerate(selecionados_analog):
                            axis_name = f"y{idx + 1}"
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=df['Tempo_Relogio'], 
                                    y=df[sensor], 
                                    name=sensor,
                                    mode='lines',
                                    line=dict(color=cores[idx % len(cores)]),
                                    yaxis=axis_name 
                                )
                            )
                            
                            vmin, vmax = LIMITES_SENSORES.get(sensor, (df[sensor].min(), df[sensor].max()))
                            axis_key = f"yaxis{idx + 1}" if idx > 0 else "yaxis"
                            
                            layout_updates[axis_key] = dict(
                                range=[vmin, vmax],       
                                overlaying="y" if idx > 0 else None, 
                                visible=False,            
                                fixedrange=True
                            )

                    if tem_flags:
                        flag_axis_idx = len(selecionados_analog) + 1 if tem_analog else 1
                        axis_name_flag = f"y{flag_axis_idx}"
                        axis_key_flag = f"yaxis{flag_axis_idx}"
                        
                        for f_idx, flag in enumerate(selecionados_flags):
                            cor_idx = (len(selecionados_analog) + f_idx) % len(cores)
                            
                            valores_numericos = pd.to_numeric(df[flag], errors='coerce').fillna(0)
                            y_plot = valores_numericos * 0.5
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=df['Tempo_Relogio'], 
                                    y=y_plot, 
                                    name=flag,
                                    mode='lines',
                                    line_shape='hv', 
                                    line=dict(color=cores[cor_idx], width=2),
                                    customdata=valores_numericos.astype(int), 
                                    hovertemplate=f"<b>{flag}</b>: %{{customdata}}<extra></extra>",
                                    yaxis=axis_name_flag 
                                )
                            )
                        
                        layout_updates[axis_key_flag] = dict(
                            range=[0.0, 1.0], 
                            overlaying="y" if tem_analog else None, 
                            visible=False,            
                            fixedrange=True 
                        )

                    fig.update_layout(
                        **layout_updates,
                        height=600, 
                        hovermode="x unified",
                        template="plotly_dark",
                        margin=dict(l=20, r=20, t=50, b=20),
                        title="Gráficos do arquivo LOG"
                    )

                    tempo_inicial = df['Tempo_Relogio'].min()
                    tempo_final_log = df['Tempo_Relogio'].max()
                    tempo_1_min = tempo_inicial + pd.Timedelta(minutes=1)
                    range_inicial = [tempo_inicial, min(tempo_1_min, tempo_final_log)]

                    fig.update_xaxes(
                        title_text="Tempo (hh:mm:ss)",
                        tickformat="%H:%M:%S",
                        hoverformat="%H:%M:%S.%L",
                        range=range_inicial,
                        rangeslider=dict(
                            visible=True,
                            thickness=0.05 
                        )
                    )

                    st.plotly_chart(fig, width="stretch")

            with aba3:
                st.subheader("Módulo de Diagnóstico Dinâmico (Motronic)")
                
                st.markdown("### Erros Registados na ECU")
                
                # Motronic armazena falhas nos slots 0 a 7
                falhas_encontradas = {}
                
                for idx in range(8):
                    col_code = f"Err_Code_{idx}"
                    col_stat = f"Err_Stat_{idx}"
                    
                    # Procura todos os códigos de erro únicos que apareceram na coluna e que sejam > 0
                    codigos_ativos = df[df[col_code] > 0][col_code].unique()
                    
                    for cod in codigos_ativos:
                        # Pega o último status em que esse código apareceu
                        ultimo_status = df[df[col_code] == cod][col_stat].iloc[-1]
                        falhas_encontradas[cod] = ultimo_status
                
                if falhas_encontradas:
                    st.error(f"Atenção! Foram detetadas {len(falhas_encontradas)} falhas neste percurso:")
                    
                    lista_visual = []
                    for cod, stat in falhas_encontradas.items():
                        desc = DTC_TABLE.get(cod, "Falha Desconhecida")
                        status_str = DTC_STATUS.get(stat, f"Desconhecido ({stat})")
                        lista_visual.append({"Código": cod, "Descrição da Falha": desc, "Status na ECU": status_str})
                        
                    st.dataframe(pd.DataFrame(lista_visual), width="stretch", hide_index=True)
                else:
                    st.success("Nenhum código de falha registado na memória. Sistema OK.")

            with aba4:
                st.subheader("Tabela de Dados Brutos")
                st.dataframe(df.drop(columns=["Tempo_Relogio", "RTM_Continuo"]), width="stretch", height=500)

            with aba5:
                st.subheader("📖 Glossário de Parâmetros Motronic 1.5.4")
                st.markdown("Consulta rápida de todos os parâmetros lidos e processados pelo sistema.")
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("#### 🌡️ Sensores Analógicos e Medidas")
                    st.markdown("""
                    * **RTM (s):** TEMPO EM SEGUNDOS
                    * **ID_Modulo:** IDENTIFICAÇÃO DO MODULO
                    * **MAP (V):** TENSÃO DO MAP EM VOLTS
                    * **MAP (kPa):** LEITURA DO MAPA EM Kpa
                    * **CTS (V):** TENSÃO DO SENSOR CTS VOLTS
                    * **CTS (°C):** TEMPERATURA EM GRAUS DO SENSOR CTS (ÁGUA)
                    * **IAT (V):** TENSÃO DO SENSOR IAT EM VOLTS
                    * **IAT (°C):** TEMPERATURA DO SENSOR IAT EM GRAUS (AR)
                    * **TPS (V):** TENSAO DO SENSOR TPS EM VOLTS
                    * **Bateria (V):** TENSAO DA BATERIA EM VOLTS
                    * **Sonda (mV):** TENSAO DA SONDA LAMBDA EM MILIVOLTS
                    * **RPM:** ROTAÇÃO DO MOTOR EM RPM
                    * **VSS (km/h):** VELOCIDADE EM KM/H
                    """)

                    st.markdown("#### ⚙️ Parâmetros Calculados / Atuadores / Outros")
                    st.markdown("""
                    * **RPM_Alvo:** ROTAÇÃO ALVO PARA MARCHA LENTA EM RPM
                    * **Tempo_Inj_Banco (ms):** TEMPO DE INJEÇÃO POR BANCO EM ms
                    * **Tempo_Inj_Ciclo (ms):** TEMPO DE INJEÇÃO POR CICLO EM ms
                    * **Avanço (°):** AVANÇO DE IGNIÇÃO EM GRAUS
                    * **Atraso_Detonacao (%):** RETARDO DA IGNIÇÃO EM PORCENTAGEM
                    * **TPS (%):** ABERTURA DO TPS EM PORCENTAGEM
                    * **Canister (%):** ABERTURA DO CANISTER EM PORCENTAGEM
                    * **EGR (%):** ABERTURA DA VALVULA EGR EM PORCENTAGEM
                    * **Tempo_Carga (ms):** TEMPO DE CARGA EM ms
                    * **Vazao_Ar_Atual (Kg/h):** VAZAO DE AR ATUAL EM Kg/h
                    * **Vazao_Ar_Alvo (Kg/h):** VAZAO DE ALVO PARA MARCHA LENTA EM Kg/h
                    * **IAC (Passos):** POSIÇÃO DO ATUADOR DE MARCHA LENTA EM PASSOS
                    * **IAC_BLM (Passos):** IAC BLOCK LEARN EM PASSOS
                    * **IAC_Integrador:** INTEGRADOR IAC EM PASSOS
                    * **Sonda_Integrador:** INTEGRADOR O2 (SONDA LAMBDA)
                    * **Sonda_BLM_Lenta:** INTEGRADOR O2 PARA MARCHA LENTA EM PASSOS
                    * **Sonda_BLM_Parcial:** INTEGRADOR O2 PARA CARGA PARCIAL EM PASSOS
                    * **Consumo_Medio (km/L):** MEDIA DE CONSUMO EM KM/L
                    * **Consumo_Inst (L/h):** CONSUMO INSTANTANEO EM LITROS/H
                    * **Distancia_Total (km):** DISTANCIA PERCORRIDA EM KM
                    * **AlphaCode:** ALPHACODE DE REFERENCIA DO MODULO
                    * **Codigo_GM:** CODIGO NUMERICO PADRÃO GM
                    * **NumFalhas:** NUMERO DE FALHAS REGISTRADAS
                    * **Versao_HW:** VERSÃO DO MOTRONIC DASHBOARD
                    """)

                with col_g2:
                    st.markdown("#### 🚩 Flags (Sinais Digitais / Status)")
                    st.markdown("""
                    * **Flag_VSS:** FLAG DO SINAL DO SENSOR DE VELOCIDADE
                    * **Flag_RPM:** FLAG DO SINAL DO SENSOR DE ROTAÇÃO
                    * **Flag_ParkNeutral:** FLAG DO SENSOR PARK/NEUTRAL
                    * **Flag_TorqueCtrl:** FLAG DO CONTROLE DE TORQUE
                    * **Flag_TPS_Lenta:** FLAG DO TPS EM MARCHA LENTA
                    * **Flag_TPS_Plena:** FLAG DO TPS EM CARGA TOTAL
                    * **Flag_AC_Pressao:** FLAG DO SENSOR DE ALTA PRESSÃO DO A/C
                    * **Flag_AC_Botao:** FLAG DA CHAVE DE ACIONAMENTO DO AR CONDICIONADO
                    * **Flag_Diag_Rqst:** FLAG DE INDICAÇÃO DO MODO DE DIAGNÓSTICO
                    * **Flag_Malha_Fechada:** FLAG DE INDICAÇÃO DA MALHA
                    * **Flag_EGR_Ativa:** FLAG DE ACIONAMENTO DA VALVULA EGR
                    * **Flag_Knock:** FLAG DO SENSOR DE RETARDO
                    * **Flag_AC_Embreagem:** FLAG DO RELÉ DE EMBREAGEM DO AR CONDICIONADO
                    * **Flag_Bomba_Comb:** FLAG DO ESTADO DA BOMBA DE COMBUSTIVEL
                    * **Flag_Check_Engine:** FLAG DE INDICAÇÃO DE FALHA
                    * **Flag_Mistura:** FLAG DE INDICAÇÃO DA MISTURA AFR
                    * **Flag_MotorCil:** FLAG DO TIPO DE MOTOR
                    * **Flag_Transmissao:** FLAG DO TIPO DE TRANSMISSÃO
                    * **Flag_Imob_Rec:** FLAG DO SINAL RECEPTOR DO IMOBILIZADOR
                    * **Flag_Imob_Act:** FLAG DO IMOBILIZADOR
                    * **Flag_Ventoinha:** FLAG DA VENTOINHA
                    """)
                    
                st.markdown("---")
                st.markdown("#### ⚠️ Códigos de Erro (DTCs)")
                st.markdown("Lista completa e pesquisável de avarias mapeadas pelo sistema Motronic 1.5.4.")
                
                dtc_df = pd.DataFrame(list(DTC_TABLE.items()), columns=["Código da Falha", "Descrição da Avaria ECU"])
                st.dataframe(dtc_df, hide_index=True, use_container_width=True)

    else:
        st.info("👈 Utilize o menu lateral esquerdo para carregar um Arquivo de log local ou explore a opção \"LOG's da Comunidade\".")
