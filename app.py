import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
import numpy as np
import time
import re
import gc # adicionado para liberação de memória dos objetos que não são mais necessários

# Suprimir o FutureWarning do pacote antigo google.generativeai nos logs da Railway
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

# ==============================================================================
# 🔴 KILL SWITCHES (CONTROLOS DE SEGURANÇA) 🔴
# Altere para False caso note alguma instabilidade no servidor ou queira desligar as funções.
# ==============================================================================
ENABLE_AI_DIAGNOSIS = False       # Liga/Desliga todo o módulo de Inteligência Artificial
ENABLE_LLM_EXPLANATION = False    # Liga/Desliga apenas a resposta humanizada (ChatGPT/Gemini)
ENABLE_LOCAL_UPLOAD = False       # Liga/Desliga o upload manual de LOGs locais na página inicial

# ==============================================================================
# TENTATIVA DE IMPORTAÇÃO DOS MÓDULOS DE IA E IA GENERATIVA
# ==============================================================================
IA_DISPONIVEL = False
NOVO_SDK_GENAI = False

if ENABLE_AI_DIAGNOSIS:
    try:
        import joblib
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # IMPORTANTE PARA MEMÓRIA: Importar o backend do Keras para limpar a sessão
        import tensorflow.keras.backend as K
        from tensorflow.keras.models import load_model
        
        # ATENÇÃO: Estes arquivos precisarão ser criados/adaptados para o Motronic futuramente
        from data_pipeline import MotronicDataPipeline
        from config_ia import COLUNAS_IA, SENSORES_CAUSA_RAIZ
        from scanner_especialista import MecanicoEspecialista_Motronic154, calcular_mad_threshold, COLUNAS as COLUNAS_SCANNER
        from biblioteca_dtw import BibliotecaDefeitosDTW
        
        # Teste de compatibilidade para a nova e velha biblioteca do Google Gemini
        try:
            from google import genai
            NOVO_SDK_GENAI = True
        except ImportError:
            import google.generativeai as genai_old
            NOVO_SDK_GENAI = False
            
        IA_DISPONIVEL = True
    except Exception as e:
        IA_DISPONIVEL = False
        ERRO_CARREGAMENTO_IA = str(e)


# --- Configuração Inicial da Página ---
st.set_page_config(page_title="Visualizador de LOG's Motronic 1.5.4", layout="wide", initial_sidebar_state="collapsed")

# --- Inicialização do Estado (Navegação e Memória de Nome do Arquivo) ---
if 'log_selecionado' not in st.session_state:
    st.session_state.log_selecionado = None
    st.session_state.nome_log_selecionado = ""

def limpar_selecao():
    st.session_state.log_selecionado = None
    st.session_state.nome_log_selecionado = ""
    # Forçar limpeza de memória ao voltar para a home
    gc.collect()

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
LIMITES_SENSORES = {
    "MAP (V)": (0.0, 5.0), "MAP (Kg/h)": (0.0, 300.0), "MAP (kPa)": (10, 105),
    "CTS (V)": (0.0, 5.0), "CTS (°C)": (0, 120), "IAT (V)": (0.0, 5.0), "IAT (°C)": (0, 100),
    "TPS (V)": (0.0, 5.0), "TPS (%)": (0, 100), "Bateria (V)": (8.0, 16.0), "Sonda (mV)": (0, 10000),
    "RPM": (0, 7500), "RPM_Alvo": (0, 3000), "VSS (km/h)": (0, 240),
    "Tempo_Inj_Banco (ms)": (0.0, 30.0), "Tempo_Inj_Ciclo (ms)": (0.0, 30.0),
    "Avanço (°)": (0, 50), "Atraso_Detonacao (%)": (0, 100),
    "Canister (%)": (0, 100), "EGR (%)": (0, 100), "Tempo_Carga (ms)": (0.0, 20.0),
    "Vazao_Ar_Atual (Kg/h)": (0.0, 300.0), "Vazao_Ar_Alvo (Kg/h)": (0.0, 300.0),
    "IAC (Passos)": (0, 255), "IAC_BLM (Passos)": (0, 255), "IAC_Integrador": (0, 255),
    "Sonda_Integrador": (0, 255), "Sonda_BLM_Lenta": (0, 255), "Sonda_BLM_Parcial": (0, 255)
}

# --- Mapeamento de AlphaCodes (Motronic 1.5.4) ---
ALPHACODE_MAP = {
    "D3":  "VECTRA GL/GLS/CD 2.2 8V",
    "M5":  "VECTRA GL/GLS/CD 2.2 8V",
    "B3":  "VECTRA CD 2.0 16V",
    "D6":  "VECTRA CD 2.0 16V",
    "D7":  "VECTRA CD 2.0 16V",
    "C9":  "VECTRA GLS/CD 2.0 16V",
    "D2":  "VECTRA CD 2.2 16V",
    "D5":  "VECTRA GL/GLS/CD 2.2 16V",
    "H2":  "VECTRA GL/GLS/CD 2.2 16V",
    "M1":  "VECTRA GL/GLS/CD 2.2 16V",
    "M6":  "VECTRA GL/GLS/CD 2.2 16V",
    "M7":  "VECTRA GL/GLS/CD 2.2 16V",
    "P2":  "VECTRA GL/GLS/CD 2.2 16V",
    "P3":  "VECTRA GL/GLS/CD 2.2 16V",
    "C8":  "VECTRA GL/GLS/CD 2.0 8V",
    "A9":  "VECTRA GL/GLS/CD 2.0 8V",
    "D9":  "VECTRA GLS/CD 2.0 8V",
    "G6":  "VECTRA GLS/CD 2.0 8V",
    "S5":  "VECTRA GL/GLS/CD 2.0 8V",
    "X6":  "VECTRA GL/GLS/CD 2.0 8V",
    "W9":  "VECTRA GL/GLS/CD 2.0 8V",
    "E1":  "BLAZER/S10 2.2 MPFI 8V",
    "F7":  "BLAZER/S10 2.2 MPFI 8V",
    "U2":  "BLAZER/S10 2.4 MPFI 8V",
    "U8":  "BLAZER/S10 2.4 MPFI 8V",
    "U1":  "BLAZER/S10 2.4 MPFI 8V",
    "U5":  "BLAZER/S10 2.4 MPFI 8V",
    "A5":  "KADETT/IPANEMA MPFI 8V"
    # Adicione os demais AlphaCodes aqui conforme necessário
}

# --- Tabela de DTCs Motronic 1.5.4 ---
DTC_TABLE = {
    10: "Codificacao Var. Nao Programada", 13: "Sonda Lambda - Circuito Aberto",
    14: "Sensor Temp. Agua - Tensao Baixa", 15: "Sensor Temp. Agua - Tensao Alta",
    16: "Circuito do Sinal de Detonacao", 18: "Modulo de Controle de Detonacao",
    19: "Sinal Incorreto de RPM", 21: "Sensor Posicao Borboleta - Tensao Alta",
    22: "Sensor Posicao Borboleta - Tensao Baixa", 24: "Sem Sinal Velocidade (VSS)",
    25: "Falha no Injetor 1 - Tensao Alta", 26: "Falha no Injetor 2 - Tensao Alta",
    27: "Falha no Injetor 3 - Tensao Alta", 28: "Falha no Injetor 4 - Tensao Alta",
    29: "Falha no Injetor 5 - Tensao Alta", 31: "Falta de Sinal de Rotacao do Motor",
    32: "Falha no Injetor 6 - Tensao Alta", 33: "Valvula EGR - Tensao Baixa",
    34: "Valvula EGR - Tensao Alta", 35: "Rele Partida Frio - Tensao Baixa",
    37: "Rele Partida Frio - Tensao Alta", 38: "Sonda Lambda - Tensao Baixa (Pobre)",
    39: "Sonda Lambda - Tensao Alta (Rica)", 44: "Mistura Pobre - Valor Constante",
    45: "Mistura Rica - Valor Constante", 48: "Tensao Baixa da Bateria",
    49: "Tensao Alta da Bateria", 52: "Lampada de Avaria - Tensao Alta",
    53: "Rele Bomba Combustivel - Tensao Baixa", 54: "Rele Bomba Combustivel - Tensao Alta",
    55: "Falha na Unidade de Comando", 56: "Controle de Ar Marcha-Lenta - Tensao Alta",
    57: "Controle de Ar Marcha-Lenta - Tensao Baixa", 61: "Valvula de Ventilacao - Tensao Baixa",
    62: "Valvula de Ventilacao - Tensao Alta", 69: "Sensor Temp. Ar - Tensao Baixa",
    71: "Sensor Temp. Ar - Tensao Alta", 73: "Sensor MAP/MAF - Tensao Baixa",
    74: "Sensor MAP/MAF - Tensao Alta", 75: "Circuito Controle Torque - Tensao Baixa",
    76: "Controle de Torque Continuo", 77: "Rele Vent. Baixa - Tensao Baixa",
    78: "Rele Vent. Baixa - Tensao Alta", 81: "Falha no Injetor 1 - Tensao Baixa",
    82: "Falha no Injetor 2 - Tensao Baixa", 83: "Falha no Injetor 3 - Tensao Baixa",
    84: "Falha no Injetor 4 - Tensao Baixa", 85: "Falha no Injetor 5 - Tensao Baixa",
    86: "Falha no Injetor 6 - Tensao Baixa", 87: "Rele Corte A/C - Tensao Baixa",
    88: "Rele Corte A/C - Tensao Alta", 93: "Sensor Hall (Fase) - Tensao Baixa",
    94: "Sensor Hall - Tensao Alta", 97: "Sinal Contr. Tracao - Tensao Alta",
    119: "Sensor MAP - Valor Incorreto Partida", 125: "Sensor MAP - Abaixo do limite minimo",
    126: "Sensor MAP - Acima do Limite maximo", 129: "Sinal Valvula EGR - Tensao Baixa",
    131: "Sinal Valvula EGR - Tensao Alta", 132: "Valvula EGR - Sinal Incorreto",
    135: "Lampada de Avaria - Tensao Baixa", 136: "Erro Saida Nao Reconhecido",
    137: "Alta Temp. Caixa ECU", 138: "Sensor MAP - Tensao Baixa",
    139: "Sensor MAP - Tensao Alta", 143: "Imobilizador - Falha na codificacao",
    144: "Imobilizador - Falta de Sinal", 145: "Imobilizador - Sinal Incorreto",
    146: "Sensor Rotacao - Tensao Baixa", 147: "Sensor Rotacao - Tensao Alta",
    171: "Rele Ventoinha - Tensao Baixa", 172: "Rele Ventoinha - Tensao Alta",
    174: "Pressao A/C - Tensao Baixa", 175: "Pressao A/C - Tensao Alta"
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
        df = df.fillna("")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar lista de comunidade: {e}")
        return pd.DataFrame()

# --- FUNÇÃO: Carregamento de Dados Robusto ---
@st.cache_data(ttl=600, max_entries=1)
def carregar_dados(arquivo_ou_url_ou_conteudo, colunas, nome_sugerido=""):
    """Retorna o DataFrame e o nome real do ficheiro detetado"""
    nome_original = nome_sugerido
    try:
        if isinstance(arquivo_ou_url_ou_conteudo, str):
            if arquivo_ou_url_ou_conteudo.startswith("http"):
                resposta = requests.get(arquivo_ou_url_ou_conteudo)
                resposta.raise_for_status()
                texto_cru = resposta.text
                
                # Tenta capturar o nome real do ficheiro extraindo do cabeçalho do Google Drive
                cd = resposta.headers.get('content-disposition', '')
                match = re.findall(r'filename="?([^";]+)"?', cd)
                if match:
                    nome_original = match[0]
            else:
                texto_cru = arquivo_ou_url_ou_conteudo
        else:
            if hasattr(arquivo_ou_url_ou_conteudo, 'seek'):
                arquivo_ou_url_ou_conteudo.seek(0)
            texto_cru = arquivo_ou_url_ou_conteudo.read()
            if isinstance(texto_cru, bytes):
                texto_cru = texto_cru.decode('utf-8', errors='ignore')
        
        # Filtro de pacotes válidos (Evita dados corrompidos)
        linhas_validas = []
        qtd_esperada = len(colunas)
        for linha in texto_cru.split('\n'):
            linha = linha.strip()
            if not linha: continue
            campos = linha.split('|')
            if len(campos) == qtd_esperada:
                linhas_validas.append(linha)
                
        if not linhas_validas: return None, nome_original

        conteudo_limpo = io.StringIO('\n'.join(linhas_validas))
        df = pd.read_csv(conteudo_limpo, sep="|", header=None, names=colunas)
        
        # OTIMIZAÇÃO DE MEMÓRIA: Downcast imediato para float32 e int32
        float_cols = df.select_dtypes(include=['float64']).columns
        int_cols = df.select_dtypes(include=['int64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        df[int_cols] = df[int_cols].astype('int32')
        
        df["RTM (s)"] = pd.to_numeric(df["RTM (s)"], errors="coerce")
        df = df.dropna(subset=["RTM (s)"]).copy()
        df = df[df["RTM (s)"] > 0].copy()
        df = df.sort_values(by="RTM (s)").reset_index(drop=True)
        
        # Tratamento de saltos temporais
        if len(df) > 1:
            diferencas = df["RTM (s)"].diff()
            if diferencas.head(10).max() > 10:
                idx_salto = diferencas.head(10).idxmax()
                df = df.iloc[idx_salto:].reset_index(drop=True)
        
        counts = df.groupby("RTM (s)")["RTM (s)"].transform('count')
        cumcounts = df.groupby("RTM (s)").cumcount()
        df["RTM_Continuo"] = df["RTM (s)"] + (cumcounts / counts).astype('float32')
        df["Tempo_Relogio"] = pd.to_datetime(df["RTM_Continuo"], unit='s')
        
        return df, nome_original
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return None, nome_original

# --- CÉREBRO DA IA (Placeholder para o Motronic) ---
@st.cache_resource(max_entries=1)
def carregar_cerebro_ia():
    if not IA_DISPONIVEL: return None, None, None, None, None
    try:
        # Limpa qualquer modelo Keras fantasma na memória antes de carregar um novo
        K.clear_session()
        
        scaler = joblib.load("scaler_motronic.pkl")
        modelo = load_model("cerebro_motronic_autoencoder.keras")
        pipeline = MotronicDataPipeline(target_freq_hz=6)
        mestre = MecanicoEspecialista_Motronic154()
        biblioteca = BibliotecaDefeitosDTW()
        return scaler, modelo, pipeline, mestre, biblioteca
    except Exception as e:
        st.error(f"Falha ao carregar pesos da IA: {e}")
        return None, None, None, None, None


# ==============================================================================
# INTERFACE PRINCIPAL (SISTEMA DE ROTEAMENTO DINÂMICO SPA)
# ==============================================================================

# Fluxo 1: Nenhum Log Selecionado (Mostra apenas o Banco de Dados)
if st.session_state.log_selecionado is None:
    
    st.markdown("<h3 style='text-align: left; color: #4F4F4F; margin-bottom: 20px;'>Visualizador de LOG's Motronic 1.5.4 DashBoard</h3>", unsafe_allow_html=True)
    
    st.subheader("🌐 Banco de Dados da Comunidade")
    st.info(" **Dica:** Para uma melhor experiência de análise e visualização dos gráficos de telemetria, recomendamos o uso de um computador.")
    st.write("👇 Clique no botão à esquerda da linha de registro do Log que deseja carregar para iniciar a análise.")
        
    df_publicos = carregar_lista_logs_publicos()
    
    if not df_publicos.empty:
        # Cria uma coluna temporária convertendo o texto para Data real (considerando o dia na frente)
        df_publicos['Temp_Date'] = pd.to_datetime(df_publicos['Data/Hora'], dayfirst=True, errors='coerce')
        
        # Ordena usando a data real e depois exclui a coluna temporária para não aparecer na tabela
        df_publicos = df_publicos.sort_values(by="Temp_Date", ascending=False).drop(columns=['Temp_Date'])
        
        event = st.dataframe(

            df_publicos,
            column_order=["Data/Hora", "Duração", "Usuário", "Veículo", "Comentário", "Obs_Moderador"],
            column_config={
                "Data/Hora": st.column_config.TextColumn("Data de Registro", width=130),
                "Duração": st.column_config.TextColumn("Duração do Registro", width=130),
                "Usuário": st.column_config.TextColumn("Enviado por", width=150),
                "Veículo": st.column_config.TextColumn("Modelo", width=250),
                "Comentário": st.column_config.TextColumn("Observações do Utilizador", width=750),
                "Obs_Moderador": None, "ID": None, "Status_Geral": None, "Tipo_Trajeto": None,
                "F_Engasgo": None, "F_Partida": None, "F_Potencia": None,
                "F_MarchaLenta": None, "F_Apagando": None, "F_Consumo": None, "ID_Arquivo": None
            },
            hide_index=True,
            width="stretch", 
            on_select="rerun",
            selection_mode="single-row",
            height=550
        )
        
        # Ação Automática ao Clicar na Tabela
        if len(event.selection.rows) > 0:
            idx = event.selection.rows[0]
            linha_selecionada = df_publicos.iloc[idx]
            
            st.session_state.log_selecionado = f"https://drive.google.com/uc?export=download&id={linha_selecionada['ID_Arquivo']}"
            st.session_state.nome_log_selecionado = "Arquivo da Comunidade" 
            st.rerun() 
            
    else:
        st.warning("Nenhum log público foi encontrado ou a base de dados encontra-se vazia.")

    # Uploader Local (Padrão Multec)
    if ENABLE_LOCAL_UPLOAD:
        st.markdown("---")
        st.subheader("📂 Abrir Arquivo Local")
        arquivo_local = st.file_uploader("Selecione o arquivo .TXT gerado pelo Aplicativo Motronic DashBoard", type=["txt"])
        
        if arquivo_local:
            try:
                conteudo = arquivo_local.getvalue().decode('utf-8', errors='ignore')
                linhas = [l for l in conteudo.split('\n') if l.strip()]
                
                if not linhas:
                    st.error("❌ O arquivo selecionado está vazio.")
                else:
                    ultima_linha = linhas[-1].split('|')
                    if len(ultima_linha) < len(COLUNAS):
                        st.error("❌ Arquivo incompatível! Este log não possui a quantidade correta de colunas do Motronic 1.5.4.")
                    else:
                        st.session_state.log_selecionado = conteudo
                        st.session_state.nome_log_selecionado = arquivo_local.name
                        st.rerun()
            except Exception as e:
                st.error("❌ Erro ao tentar ler a assinatura do arquivo. Arquivo corrompido.")

# Fluxo 2: Log Selecionado (Abre APENAS o Painel de Análise)
else:
    # Título e Botão de Voltar na mesma linha
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown("<h3 style='text-align: left; color: #4F4F4F; margin-top: 0px; margin-bottom: 20px;'>Visualizador de LOG's Motronic 1.5.4 DashBoard</h3>", unsafe_allow_html=True)
    with col_btn:
        st.button("⬅️ Voltar à Comunidade", on_click=limpar_selecao, use_container_width=True)
        
    resultado_carregamento = carregar_dados(st.session_state.log_selecionado, COLUNAS, st.session_state.nome_log_selecionado)
    
    if resultado_carregamento is not None and resultado_carregamento[0] is not None and not resultado_carregamento[0].empty:
        df, nome_final = resultado_carregamento
        versao_dash = str(df["Versao_HW"].iloc[-1])

        aba1, aba2, aba3, aba4, aba5 = st.tabs([
            "📊 Visão Geral", 
            "📈 Telemetria (Gráficos)", 
            "⚠️ Diagnóstico", 
            "📋 Dados Brutos",
            "📖 Glossário"
        ])

        # ABA 1: VISÃO GERAL
        with aba1:
            st.success(f"Log carregado: **{nome_final}** (Dashboard v{versao_dash} | {len(df)} registros)")
            try:
                alpha = str(df["AlphaCode"].iloc[-1]).strip()
                gm_code = str(df["Codigo_GM"].iloc[-1]).strip()
                modelo_veiculo = ALPHACODE_MAP.get(alpha, "Modelo não mapeado")
                nome_modulo = f"{modelo_veiculo} (Módulo GM: {gm_code} | AlphaCode: {alpha})"
            except:
                nome_modulo = "Módulo Desconhecido"
                
            st.info(f"**Identificação da ECU:** {nome_modulo}")
            
            st.subheader("Resumo do Percurso")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RPM Máximo", f"{df['RPM'].max():.0f} RPM")
            col2.metric("Temp Máxima Água", f"{df['CTS (°C)'].max():.0f} °C")
            col3.metric("MAP Máximo", f"{df['MAP (kPa)'].max():.1f} kPa")
            col4.metric("Velocidade Máxima", f"{df['VSS (km/h)'].max():.0f} km/h")

            st.markdown("---")
            st.subheader("Médias de Funcionamento")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Tensão Média Bateria", f"{df['Bateria (V)'].mean():.2f} V")
            col_b.metric("Avanço Médio", f"{df['Avanço (°)'].mean():.1f} °")
            col_c.metric("Sonda Lambda Média", f"{df['Sonda (mV)'].mean():.0f} mV")
            col_d.metric("MAP Médio", f"{df['MAP (kPa)'].mean():.1f} kPa")

        # ABA 2: GRÁFICOS DE TELEMETRIA
        with aba2:
            # 1. Defina o que você NÃO quer mostrar
            excluir_analog = ["IAC_Integrador", "Sonda_BLM_Lenta", "Sonda_BLM_Parcial"]
            excluir_flags = ["Flag_VSS", "Flag_RPM", "Flag_Diag_Rqst", "Flag_Bomba_Ar", "Flag_MotorCil", "Flag_Transmissao", "Flag_Imob_Rec", "Flag_Imob_Act"]  

            # 2. Filtra as listas originais
            colunas_analogicas = list(LIMITES_SENSORES.keys())
            colunas_flags = [c for c in df.columns if c.startswith("Flag_") and c not in excluir_flags]
    
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                selecionados_analog = st.multiselect("Sensores Analógicos:", options=colunas_analogicas, default=["RPM", "MAP (kPa)", "TPS (%)", "VSS (km/h)", "CTS (°C)"])
            with col_sel2:
                selecionados_flags = st.multiselect("Sinais Digitais / Flags (ON/OFF):", options=colunas_flags, default=["Flag_Knock", "Flag_AC_Embreagem"])

            if selecionados_analog or selecionados_flags:
                fig = go.Figure()
                cores = px.colors.qualitative.Plotly
                layout_updates = {}
                
                tem_analog = len(selecionados_analog) > 0
                tem_flags = len(selecionados_flags) > 0
                
                if tem_analog:
                    for idx, sensor in enumerate(selecionados_analog):
                        axis_name = f"y{idx + 1}"
                        fig.add_trace(go.Scattergl(x=df['Tempo_Relogio'], y=df[sensor], name=sensor, mode='lines', line=dict(color=cores[idx % len(cores)]), yaxis=axis_name))
                        vmin, vmax = LIMITES_SENSORES.get(sensor, (df[sensor].min(), df[sensor].max()))
                        axis_key = f"yaxis{idx + 1}" if idx > 0 else "yaxis"
                        layout_updates[axis_key] = dict(range=[vmin, vmax], overlaying="y" if idx > 0 else None, visible=False, fixedrange=True)

                if tem_flags:
                    flag_axis_idx = len(selecionados_analog) + 1 if tem_analog else 1
                    axis_name_flag = f"y{flag_axis_idx}"
                    axis_key_flag = f"yaxis{flag_axis_idx}"
                    
                    def hex_to_rgba(hex_color, opacity=0.15):
                        hex_color = hex_color.lstrip('#')
                        try:
                            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            return f"rgba({r}, {g}, {b}, {opacity})"
                        except:
                            return f"rgba(100, 100, 100, {opacity})"
                    
                    for f_idx, flag in enumerate(selecionados_flags):
                        cor_idx = (len(selecionados_analog) + f_idx) % len(cores)
                        cor_hex = cores[cor_idx]
                        cor_transparente = hex_to_rgba(cor_hex, 0.15)
                        
                        valores_numericos = pd.to_numeric(df[flag], errors='coerce').fillna(0)
                        
                        y_plot = valores_numericos * 1.0 
                        
                        fig.add_trace(go.Scatter(
                            x=df['Tempo_Relogio'], 
                            y=y_plot, 
                            name=flag, 
                            mode='lines', 
                            line_shape='hv', 
                            line=dict(color=cor_hex, width=2),
                            fill='tozeroy',
                            fillcolor=cor_transparente,
                            customdata=valores_numericos.astype(int), 
                            hovertemplate=f"<b>{flag}</b>: %{{customdata}}<extra></extra>", 
                            yaxis=axis_name_flag
                        ))
                    
                    layout_updates[axis_key_flag] = dict(range=[0.0, 1.0], overlaying="y" if tem_analog else None, visible=False, fixedrange=True)

                fig.update_layout(
                    **layout_updates, 
                    height=600, 
                    hovermode="x unified", 
                    template="plotly_dark", 
                    margin=dict(l=20, r=20, t=50, b=20), 
                    title="Gráficos do arquivo LOG"
                )
                
                tempo_inicial = df['Tempo_Relogio'].min()
                range_inicial = [tempo_inicial, min(tempo_inicial + pd.Timedelta(minutes=1), df['Tempo_Relogio'].max())]
                
                fig.update_xaxes(
                    title_text="Tempo (hh:mm:ss)", 
                    tickformat="%H:%M:%S", 
                    hoverformat="%H:%M:%S.%L", 
                    range=range_inicial, 
                    rangeslider=dict(visible=True, thickness=0.05)
                )

                st.plotly_chart(fig, width="stretch")
                
                # Liberar memória do gráfico logo após renderizar
                del fig
                gc.collect()

        # ABA 3: DIAGNÓSTICO E INTELIGÊNCIA ARTIFICIAL
        with aba3:
            st.subheader("Módulo de Diagnóstico e Análise de Falhas")
            
            # ---------------------------------------------------------
            # 1. SISTEMA ORIGINAL DA ECU
            # ---------------------------------------------------------
            st.markdown("### Falhas Registradas na ECU")
            falhas_encontradas = {}
                
            for idx in range(8):
                col_code = f"Err_Code_{idx}"
                col_stat = f"Err_Stat_{idx}"
                
                # Procura todos os códigos de erro únicos que apareceram na coluna e que sejam > 0
                codigos_ativos = df[df[col_code] > 0][col_code].unique()
                
                for cod in codigos_ativos:
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
                st.success("Nenhum código de falha registado na memória da ECU. Sistema OK.")

            st.markdown("---")
            
            # ---------------------------------------------------------
            # 2. SISTEMA DE IA NEURO-SIMBÓLICO
            # ---------------------------------------------------------
            if ENABLE_AI_DIAGNOSIS:
                st.markdown("### Diagnóstico Avançado Utilizando Inteligência Artificial (IA)")
                
                if not IA_DISPONIVEL:
                    st.warning(f"O módulo de IA não está disponível neste servidor. Erro interno: {ERRO_CARREGAMENTO_IA}")
                else:
                    if st.button("🔍 Executar a Análise com IA", type="primary"):
                        with st.spinner("Iniciando os modelos matemáticos e a avaliando o Log..."):
                            
                            # Variáveis para limpar no finally
                            df_cru_ia = None
                            df_alvo = None
                            fig_ia = None
                            
                            try:
                                scaler, modelo, pipeline, mestre, biblioteca = carregar_cerebro_ia()
                                
                                if modelo is None:
                                    st.error("Falha ao carregar o Cérebro Neural. Operação cancelada.")
                                else:
                                    # LÓGICA DE IA (Adaptar no backend para features do Motronic)
                                    if isinstance(st.session_state.log_selecionado, str) and st.session_state.log_selecionado.startswith("http"):
                                        resposta = requests.get(st.session_state.log_selecionado)
                                        texto_cru = resposta.text
                                    else:
                                        texto_cru = st.session_state.log_selecionado
                                        
                                    linhas_validas = [l for l in texto_cru.split('\n') if len(l.split('|')) == len(COLUNAS_SCANNER)]
                                    df_cru_ia = pd.read_csv(io.StringIO('\n'.join(linhas_validas)), sep="|", header=None, names=COLUNAS_SCANNER)
                                    
                                    df_alvo = pipeline.processar_log(df_cru_ia)
                                    
                                    # OTIMIZAÇÃO: Downcast float32
                                    float_cols = df_alvo.select_dtypes(include=['float64']).columns
                                    df_alvo[float_cols] = df_alvo[float_cols].astype('float32')
                                    
                                    # Extração de Features Temporais - Adaptado para Sensores Motronic
                                    df_alvo['Sonda_Diff'] = df_alvo['Sonda (mV)'].diff().fillna(0).abs().astype('float32')
                                    df_alvo['TPS_Diff_Abs'] = df_alvo['TPS (%)'].diff().fillna(0).abs().astype('float32')
                                    df_alvo['RPM_Diff_Abs'] = df_alvo['RPM'].diff().fillna(0).abs().astype('float32')
                                    df_alvo['Bateria_Diff_Abs'] = df_alvo['Bateria (V)'].diff().fillna(0).abs().astype('float32')
                                    df_alvo['MAP_V_Diff_Abs'] = df_alvo['MAP (V)'].diff().fillna(0).abs().astype('float32')
                                    df_alvo['MAP_kPa_Diff_Abs'] = df_alvo['MAP (kPa)'].diff().fillna(0).abs().astype('float32')
                                    df_alvo['CTS_V_Diff_Abs'] = df_alvo['CTS (V)'].diff().fillna(0).abs().astype('float32')
                                    df_alvo['CTS_C_Diff_Abs'] = df_alvo['CTS (°C)'].diff().fillna(0).abs().astype('float32')
                                    df_alvo['IAT_C_Diff_Abs'] = df_alvo['IAT (°C)'].diff().fillna(0).abs().astype('float32')
                                    df_alvo['TPS_V_Diff_Abs'] = df_alvo['TPS (V)'].diff().fillna(0).abs().astype('float32')
                                    
                                    limites_por_estado = {'Idle': 3.5, 'Cruise': 4.0, 'Decel': 4.5, 'WOT': 5.0, 'Warmup': 6.0}
                                    limite_global_mad = 4.0

                                    dados_normalizados = scaler.transform(df_alvo[COLUNAS_IA])
                                    dados_reconstruidos = modelo.predict(dados_normalizados, verbose=0)
                                    
                                    erros_individuais_brutos = np.power(dados_normalizados - dados_reconstruidos, 2)
                                    df_erros_individuais = pd.DataFrame(erros_individuais_brutos, columns=COLUNAS_IA, index=df_alvo.index)
                                    
                                    df_alvo['Erro_IA_Pura'] = np.mean(erros_individuais_brutos, axis=1).astype('float32')
                                    df_alvo['Limite_MAD_Estado'] = df_alvo['Estado_Motor'].map(limites_por_estado).fillna(limite_global_mad).astype('float32')

                                    diagnosticos, sensores_culpados_brutos, grau_severidade = [], [], []
                                    for index, linha in df_alvo.iterrows():
                                        erro_ia = linha['Erro_IA_Pura']
                                        limite_ia = linha['Limite_MAD_Estado']
                                        diag, sensor = mestre.auditar_diagnostico_ia(linha)
                                        
                                        if diag != "Normal":
                                            diagnosticos.append(diag)
                                            sensores_culpados_brutos.append(sensor)
                                            grau_severidade.append(limite_ia * 3) 
                                        elif erro_ia > limite_ia:
                                            diagnosticos.append("Anomalia Sistémica/Estatística (IA)")
                                            sensores_culpados_brutos.append("IA_Genérica") 
                                            grau_severidade.append(erro_ia)
                                        else:
                                            diagnosticos.append("Normal")
                                            sensores_culpados_brutos.append("Nenhum")
                                            grau_severidade.append(0)

                                    df_alvo['Severidade_Final'] = np.array(grau_severidade, dtype=np.float32)
                                    df_alvo['Diagnostico_Texto'] = diagnosticos
                                    df_alvo['Culpado_Bruto'] = sensores_culpados_brutos
                                    df_alvo['Culpado_Final'] = df_alvo['Culpado_Bruto']
                                    
                                    # Crivo de Falsos Positivos Motronic
                                    mask_ia = df_alvo['Culpado_Bruto'] == 'IA_Genérica'
                                    if mask_ia.any():
                                        max_sensors = df_erros_individuais.loc[mask_ia, SENSORES_CAUSA_RAIZ].idxmax(axis=1)
                                        max_erros = df_erros_individuais.loc[mask_ia, SENSORES_CAUSA_RAIZ].max(axis=1)
                                        valid_ia_mask = mask_ia & (max_erros > 6.0)
                                        
                                        carga_real = (df_alvo['TPS (%)'] >= 2.0) | (df_alvo['VSS (km/h)'] >= 2)
                                        falso_tps = valid_ia_mask & (max_sensors == 'TPS (%)') & (df_alvo['TPS_Diff_Abs'] < 30.0)
                                        falso_tps_v = valid_ia_mask & (max_sensors == 'TPS (V)') & (df_alvo['TPS_V_Diff_Abs'] < 1.5)
                                        falso_rpm = valid_ia_mask & (max_sensors == 'RPM') & (df_alvo['RPM_Diff_Abs'] < 300)
                                        falso_bat = valid_ia_mask & (max_sensors == 'Bateria (V)') & (df_alvo['Bateria_Diff_Abs'] < 0.5)
                                        falso_map_v = valid_ia_mask & (max_sensors == 'MAP (V)') & (df_alvo['MAP_V_Diff_Abs'] < 1.5) & carga_real
                                        falso_map_kpa = valid_ia_mask & (max_sensors == 'MAP (kPa)') & (df_alvo['MAP_kPa_Diff_Abs'] < 30.0) & carga_real
                                        falso_cts_v = valid_ia_mask & (max_sensors == 'CTS (V)') & (df_alvo['CTS_V_Diff_Abs'] < 0.2)
                                        falso_cts_c = valid_ia_mask & (max_sensors == 'CTS (°C)') & (df_alvo['CTS_C_Diff_Abs'] < 2.0)
                                        falso_sonda = valid_ia_mask & (max_sensors == 'Sonda (mV)') & (df_alvo['Sonda_Diff'] < 200)
                                        
                                        invalidos = falso_tps | falso_rpm | falso_bat | falso_map_v | falso_map_kpa | falso_cts_v | falso_cts_c | falso_tps_v | falso_sonda
                                        valid_ia_mask = valid_ia_mask & ~invalidos
                                        invalid_ia_mask = mask_ia & ~valid_ia_mask
                                        
                                        df_alvo.loc[valid_ia_mask, 'Culpado_Final'] = max_sensors[valid_ia_mask]
                                        df_alvo.loc[invalid_ia_mask, 'Culpado_Final'] = "Nenhum"
                                        df_alvo.loc[invalid_ia_mask, 'Culpado_Bruto'] = "Nenhum"
                                        df_alvo.loc[invalid_ia_mask, 'Diagnostico_Texto'] = "Normal"
                                        df_alvo.loc[invalid_ia_mask, 'Severidade_Final'] = 0.0

                                    FREQ_HZ = 6
                                    frames_persistencia = max(2, int(FREQ_HZ * 0.4)) 
                                    anomalia_instantanea = df_alvo['Severidade_Final'] > df_alvo['Limite_MAD_Estado']
                                    df_alvo['Falha_Confirmada'] = anomalia_instantanea.rolling(window=frames_persistencia, min_periods=1).min() > 0

                                    margem = FREQ_HZ * 2 
                                    n_start, n_end = min(margem, len(df_alvo)), min(margem, len(df_alvo))
                                    df_alvo.iloc[:n_start, df_alvo.columns.get_loc('Falha_Confirmada')] = False
                                    df_alvo.iloc[-n_end:, df_alvo.columns.get_loc('Falha_Confirmada')] = False

                                    falhas_confirmadas = df_alvo[df_alvo['Falha_Confirmada']]
                                    picos_falha = len(falhas_confirmadas)
                                    
                                    texto_laudo_llm = ""
                                    assinatura_dtw = ""
                                    sensores_para_grafico = []

                                    if picos_falha > 0:
                                        st.error(f"A IA detetou anomalias reais! ({picos_falha} frames confirmados, ~{picos_falha/FREQ_HZ:.1f} segundos)")
                                        
                                        falhas_fisicas = falhas_confirmadas[falhas_confirmadas['Culpado_Bruto'] != "IA_Genérica"]
                                        falhas_ia = falhas_confirmadas[falhas_confirmadas['Culpado_Bruto'] == "IA_Genérica"]
                                        
                                        if len(falhas_fisicas) > 0:
                                            principal = falhas_fisicas['Diagnostico_Texto'].value_counts().index[0]
                                            culpados = falhas_fisicas['Culpado_Final'].unique().tolist()
                                            st.warning(f"**Diagnóstico Físico:** {principal}")
                                            st.warning(f"**Sensores Culpados:** {culpados}")
                                            texto_laudo_llm = principal
                                            
                                            sensores_para_grafico.extend(culpados)
                                            
                                            df_recorte = df_alvo[df_alvo['Falha_Confirmada']].copy()
                                            diag_dtw, distancia = biblioteca.classificar_anomalia(df_recorte, culpados)
                                            assinatura_dtw = diag_dtw
                                            st.info(f"**Análise de Curva (DTW):** {diag_dtw}")
                                            
                                            # Limpar df temporário
                                            del df_recorte
                                        
                                        if len(falhas_ia) > 0:
                                            principal_ia = falhas_ia['Culpado_Final'].value_counts().index[0]
                                            st.warning(f"**Causa Raiz Estatística (IA):** Anomalia centrada em {principal_ia}")
                                            
                                            if len(falhas_fisicas) == 0:
                                                texto_laudo_llm = f"Desvio matemático grave focado no sensor {principal_ia}"
                                                assinatura_dtw = "Anomalia Não Mapeada"
                                            
                                            for sensor in falhas_ia['Culpado_Final'].unique():
                                                if sensor not in sensores_para_grafico and len(sensores_para_grafico) < 4:
                                                    sensores_para_grafico.append(sensor)
                                    else:
                                        st.success("A IA aprovou este log. Nenhuma anomalia grave ou desvio estatístico confirmado no motor.")
                                        texto_laudo_llm = "Nenhum problema encontrado. O motor está a funcionar perfeitamente dentro das tolerâncias físicas e estatísticas."
                                        assinatura_dtw = "Nenhuma"

                                    sensores_para_grafico = list(dict.fromkeys(sensores_para_grafico))[:4]

                                    # RENDERIZAÇÃO DO RELATÓRIO GRÁFICO DA IA
                                    st.markdown("#### Relatório Gráfico Detalhado:")
                                    
                                    if 'RTM (s)' in df_alvo.columns:
                                        df_alvo['Tempo_Relogio'] = pd.to_datetime(df_alvo['RTM (s)'], unit='s')
                                    else:
                                        df_alvo['Tempo_Relogio'] = df_alvo.index
                                    tempo_plot = df_alvo['Tempo_Relogio']
                                    
                                    num_paineis = 2 + len(sensores_para_grafico)
                                    titulos_paineis = ["Visão Geral do Motor (RPM & TPS)"] 
                                    titulos_paineis.extend([f"Monitorização de Falha: {s}" for s in sensores_para_grafico])
                                    titulos_paineis.append("Avaliação Cruzada IA (Sensores + Tensões + Flags)")

                                    fig_ia = make_subplots(
                                        rows=num_paineis, cols=1, 
                                        shared_xaxes=True, 
                                        vertical_spacing=0.08,
                                        subplot_titles=titulos_paineis,
                                        specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (num_paineis - 1)
                                    )
                                    
                                    leg_1 = "legend"
                                    fig_ia.add_trace(go.Scatter(x=tempo_plot, y=df_alvo['RPM'], name='RPM', line=dict(color='#1f77b4', width=2), legend=leg_1), row=1, col=1, secondary_y=False)
                                    fig_ia.add_trace(go.Scatter(x=tempo_plot, y=df_alvo['TPS (%)'], name='TPS (%)', line=dict(color='#2ca02c', width=1.5), opacity=0.7, legend=leg_1), row=1, col=1, secondary_y=True)

                                    estados_cores = {
                                        'Idle': 'rgba(0, 255, 255, 0.15)', 'Cruise': 'rgba(128, 128, 128, 0.15)', 
                                        'WOT': 'rgba(255, 0, 0, 0.15)', 'Decel': 'rgba(0, 0, 255, 0.15)', 'Warmup': 'rgba(255, 0, 255, 0.15)'
                                    }
                                    for estado, cor in estados_cores.items():
                                        onde = df_alvo['Estado_Motor'] == estado
                                        if onde.any():
                                            y_bg = np.where(onde, 7500, 0)
                                            fig_ia.add_trace(go.Scatter(x=tempo_plot, y=y_bg, fill='tozeroy', mode='none', fillcolor=cor, name=f'Estado: {estado}', hoverinfo='skip', line_shape='hv', legend=leg_1), row=1, col=1, secondary_y=False)

                                    fig_ia.update_yaxes(title_text="RPM", title_font=dict(color="#1f77b4", size=11, family="Arial Black"), range=[0, 7500], row=1, col=1, secondary_y=False)
                                    fig_ia.update_yaxes(title_text="TPS (%)", title_font=dict(color="#2ca02c", size=11, family="Arial Black"), range=[0, 100], row=1, col=1, secondary_y=True)

                                    for i, sensor in enumerate(sensores_para_grafico):
                                        row = i + 2
                                        leg_s = f"legend{row}"
                                        fig_ia.add_trace(go.Scatter(x=tempo_plot, y=df_alvo[sensor], name=sensor, line=dict(color='darkorange', width=2), legend=leg_s), row=row, col=1)

                                        falha_sensor = (df_alvo['Falha_Confirmada'] & (df_alvo['Culpado_Final'] == sensor)).rolling(window=FREQ_HZ, center=True, min_periods=1).max() > 0
                                        if falha_sensor.any():
                                            y_max = LIMITES_SENSORES.get(sensor, (df_alvo[sensor].min(), df_alvo[sensor].max() * 1.1))[1]
                                            if pd.isna(y_max) or y_max == 0: y_max = 100
                                            y_bg_sensor = np.where(falha_sensor, y_max, 0)
                                            fig_ia.add_trace(go.Scatter(x=tempo_plot, y=y_bg_sensor, fill='tozeroy', mode='none', fillcolor='rgba(255,0,0,0.3)', name=f'Alvo Culpado', hoverinfo='skip', line_shape='hv', legend=leg_s), row=row, col=1)
                                        
                                        fig_ia.update_yaxes(title_text=sensor, title_font=dict(color="darkorange", size=11, family="Arial Black"), row=row, col=1)

                                    row_ia = num_paineis
                                    leg_ia = f"legend{row_ia}"
                                    fig_ia.add_trace(go.Scatter(x=tempo_plot, y=df_alvo['Severidade_Final'], name='Erro Reconstrução', line=dict(color='white', width=1.5), legend=leg_ia), row=row_ia, col=1)
                                    
                                    mad_medio = df_alvo['Limite_MAD_Estado'].mean()
                                    linha_mad_constante = np.full(len(tempo_plot), mad_medio)
                                    
                                    fig_ia.add_trace(go.Scatter(x=tempo_plot, y=linha_mad_constante, name='Threshold Médio (MAD)', line=dict(color='red', width=2, dash='dash'), legend=leg_ia), row=row_ia, col=1)

                                    falha_geral_visual = df_alvo['Falha_Confirmada'].rolling(window=FREQ_HZ, center=True, min_periods=1).max() > 0
                                    if falha_geral_visual.any():
                                        fig_ia.add_trace(go.Scatter(x=tempo_plot, y=np.where(falha_geral_visual, linha_mad_constante, np.nan), line=dict(width=0), showlegend=False, hoverinfo='skip'), row=row_ia, col=1)
                                        fig_ia.add_trace(go.Scatter(x=tempo_plot, y=np.where(falha_geral_visual, df_alvo['Severidade_Final'], np.nan), fill='tonexty', mode='none', fillcolor='rgba(255,0,0,0.4)', name='Falha Sistêmica', hoverinfo='skip', legend=leg_ia), row=row_ia, col=1)

                                    fig_ia.update_yaxes(title_text="Gravidade", title_font=dict(color="white", size=11, family="Arial Black"), row=row_ia, col=1)

                                    layout_updates = {
                                        "height": 250 + (220 * (num_paineis - 1)), 
                                        "template": "plotly_dark",
                                        "margin": dict(l=20, r=150, t=60, b=20), 
                                        "hovermode": "x unified",
                                        "dragmode": False, 
                                    }

                                    for r in range(1, num_paineis + 1):
                                        leg_key = f"legend{r}" if r > 1 else "legend"
                                        y_pos = 1.0 - ((r - 1) * (1.0 / num_paineis)) - (0.5 / num_paineis)
                                        layout_updates[leg_key] = dict(
                                            y=y_pos, yanchor="middle", x=1.02, xanchor="left",
                                            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                                            bordercolor="rgba(255,255,255,0.2)", borderwidth=1
                                        )

                                    fig_ia.update_layout(**layout_updates)
                                    fig_ia.update_yaxes(fixedrange=True) 

                                    for r in range(1, num_paineis + 1):
                                        title = "Tempo Real do Log (hh:mm:ss)" if r == num_paineis else None
                                        fig_ia.update_xaxes(title_text=title, tickformat="%H:%M:%S", hoverformat="%H:%M:%S.%L", fixedrange=True, row=r, col=1)

                                    st.plotly_chart(fig_ia, width="stretch", config={'scrollZoom': False})

                                    # --- RESPOSTA HUMANIZADA (LLM - GOOGLE GEMINI) ADAPTADA PARA MOTRONIC ---
                                    if ENABLE_LLM_EXPLANATION and picos_falha > 0:
                                        st.markdown("---")
                                        st.markdown("### Laudo Técnico do Mecânico Virtual:")
                                        with st.spinner("Gerando Laudo Técnico..."):
                                            try:
                                                alpha_code = str(df["AlphaCode"].iloc[-1]).strip()
                                                gm_code = str(df["Codigo_GM"].iloc[-1]).strip()
                                                modelo_veiculo = ALPHACODE_MAP.get(alpha_code, "Modelo não mapeado")
                                                descricao_modulo = f"Veículo: {modelo_veiculo} | Código GM: {gm_code} | AlphaCode: {alpha_code}"

                                                chave_api = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
                                                if not chave_api:
                                                    try:
                                                        chave_api = st.secrets.get("GEMINI_API_KEY", "")
                                                    except Exception:
                                                        pass

                                                chave_api = chave_api.strip(' "\'\n\r')

                                                if chave_api:
                                                    prompt = f"""
                                                    Sintoma no motor (Injeção Motronic 1.5.4): {texto_laudo_llm}
                                                    
                                                    Aja como um sistema automatizado de diagnóstico mecânico. 
                                                    A sua ÚNICA função é retornar as possíveis causas e os itens que devem ser verificados.

                                                    Módulo ECU Analisado: {descricao_modulo}
                                                    Sistema de Injeção: Bosch Motronic 1.5.4 (MPFI - Multiponto).
                                                    
                                                    Lembre-se das características do Motronic 1.5.4:
                                                    - É um sistema multiponto (4 bicos injetores), podendo pulsar em banco ou sequencial (dependendo do modelo exato).
                                                    - Possui Sonda Lambda (sensor de oxigênio).
                                                    - Utiliza Sensor MAP e sensor IAT integrado, mas o sistema baseia-se muito na pressão do coletor.
                                                    - Possui sensor de detonação (Knock sensor).
                                                    - Usa Atuador de Marcha Lenta tipo motor de passo (IAC).
                                                    - O sinal de rotação vem de uma Roda Fônica (sensor de rotação magnético).
                                                    - Bobina de Ignição dupla.
                                                    - Possui válvula EGR dependendo do motor/ano e Canister.
                                                    
                                                    REGRAS OBRIGATÓRIAS (Falhar não é opção):
                                                    1. PROIBIDO usar saudações, introduções ou conclusões (ex: "Olá", "Com base...", "Aqui estão").
                                                    2. PROIBIDO descrever o que o condutor está a sentir.
                                                    3. A sua resposta DEVE começar EXATAMENTE com a linha: "Avaliação Técnica e o que verificar primeiro:"
                                                    4. Faça uma descrição das possíveis causas, levando em consideração todas as informações técnicas fornecidas do veículo e liste logo abaixo uma lista de verificação em formato de bullet points, sendo muito objetivo e usando negrito nas peças.
                                                    """
                                                   
                                                    if NOVO_SDK_GENAI:
                                                        client = genai.Client(api_key=chave_api)
                                                        resposta_llm = client.models.generate_content(
                                                            model='gemini-2.5-flash',
                                                            contents=prompt
                                                        )
                                                        st.info(resposta_llm.text)
                                                    else:
                                                        genai_old.configure(api_key=chave_api)
                                                        modelos_permitidos = [m.name for m in genai_old.list_models() if 'generateContent' in m.supported_generation_methods]
                                                        
                                                        if not modelos_permitidos:
                                                            raise Exception("A sua API Key conectou com sucesso, mas não tem acesso a nenhum modelo de geração de texto.")
                                                            
                                                        modelo_escolhido = modelos_permitidos[0]
                                                        for nome in modelos_permitidos:
                                                            if 'flash' in nome.lower():
                                                                modelo_escolhido = nome
                                                                break
                                                                
                                                        modelo_escolhido = modelo_escolhido.replace('models/', '')
                                                        llm_model = genai_old.GenerativeModel(modelo_escolhido)
                                                        resposta_llm = llm_model.generate_content(prompt)
                                                        
                                                        st.info(resposta_llm.text)
                                                        
                                                    st.caption("⚠️ *Nota: A Inteligência Artificial pode cometer erros de interpretação. Confirme sempre o diagnóstico com um profissional qualificado.*")
                                                    
                                                else:
                                                    st.warning("⚠️ **Falta a Chave API no Servidor:** A chave não foi carregada pelo Python. **Solução:** Faça um **Redeploy** manual na Railway.")
                                            except Exception as e_llm:
                                                st.error(f"⚠️ **Falha de Comunicação com a IA da Google!**\n\n**O erro relatado foi:** `{e_llm}`")

                            except Exception as err:
                                st.error(f"❌ Ocorreu um erro inesperado durante a análise de IA: {err}")
                            finally:
                                # OTIMIZAÇÃO: Excluir os grandes objetos de memória explicitamente antes do garbage collector
                                if df_cru_ia is not None:
                                    del df_cru_ia
                                if df_alvo is not None:
                                    del df_alvo
                                if fig_ia is not None:
                                    del fig_ia
                                if 'dados_normalizados' in locals():
                                    del dados_normalizados
                                if 'dados_reconstruidos' in locals():
                                    del dados_reconstruidos
                                    
                                # Limpar a sessão do Keras/TensorFlow para liberar memória de tensores
                                if IA_DISPONIVEL:
                                    K.clear_session()
                                    
                                # força a liberação de memória
                                gc.collect()

        # ABA 4: DADOS BRUTOS
        with aba4:
            st.subheader("Tabela de Dados Brutos")
            st.dataframe(df.drop(columns=["Tempo_Relogio", "RTM_Continuo"]), width="stretch", height=500)

        # ABA 5: GLOSSÁRIO (Estruturado em Colunas e Tabela)
        with aba5:
            st.subheader("📖 Glossário de Parâmetros Motronic 1.5.4")
            st.markdown("Consulta rápida de todos os parâmetros lidos e processados pelo sistema.")
            
            col_g1, col_g2, col_g3 = st.columns(3)
            
            with col_g1:
                st.markdown("#### 🌡️ Sensores Analógicos e Medidas")
                st.markdown("""
                | Parâmetro | Significado |
                | :--- | :--- |
                | **RTM (s)** | Tempo de funcionamento em segundos. |
                | **ID_Modulo** | Identificação do Módulo. |
                | **MAP (V / kPa)** | Pressão do coletor de admissão (Volts e kPa). |
                | **CTS (V / °C)** | Temperatura da água do motor (Volts e Graus). |
                | **IAT (V / °C)** | Temperatura do ar de admissão (Volts e Graus). |
                | **TPS (V / %)** | Posição da borboleta de aceleração (Volts e %). |
                | **Bateria (V)** | Tensão de alimentação (Bateria/Alternador). |
                | **Sonda (mV)** | Tensão da sonda lambda (Mistura Rica/Pobre). |
                | **RPM** | Rotação atual do motor. |
                | **VSS (km/h)** | Velocidade do veículo. |
                """)

            with col_g2:
                st.markdown("#### ⚙️ Parâmetros Calculados e Atuadores")
                st.markdown("""
                | Parâmetro | Significado |
                | :--- | :--- |
                | **RPM_Alvo** | Rotação alvo estipulada pela ECU para lenta. |
                | **Tempo_Inj_Banco (ms)** | Tempo de injeção por banco. |
                | **Tempo_Inj_Ciclo (ms)** | Tempo de injeção por ciclo. |
                | **Avanço (°)** | Avanço/Ponto de ignição em graus. |
                | **Atraso_Detonacao (%)** | Retardo da ignição por detonação. |
                | **Canister (%)** | Abertura da purga do Canister. |
                | **EGR (%)** | Abertura da válvula EGR. |
                | **Tempo_Carga (ms)** | Tempo de carga do motor. |
                | **Vazao_Ar_Atual (Kg/h)** | Vazão de ar atual calculada. |
                | **Vazao_Ar_Alvo (Kg/h)** | Vazão de ar alvo para marcha lenta. |
                | **IAC (Passos)** | Posição do motor de passo da marcha lenta. |
                | **IAC_BLM (Passos)** | *Block Learn Multiplier* do IAC. |
                | **IAC_Integrador** | Fator integrador do IAC. |
                | **Sonda_Integrador** | Integrador da Sonda Lambda (Correção curta). |
                | **Sonda_BLM_Lenta** | Aprendizado de mistura em Lenta. |
                | **Sonda_BLM_Parcial** | Aprendizado de mistura em carga Parcial. |
                | **Consumo Médio / Inst.**| Cálculos de Consumo (km/L e L/h). |
                | **Distancia_Total (km)** | Distância percorrida. |
                | **AlphaCode / Cod. GM** | Identificações e calibração da ECU. |
                | **NumFalhas / Versao_HW**| Total de erros registrados e versão do log. |
                """)

            with col_g3:
                st.markdown("#### 🚩 Flags (Sinais Digitais / Status)")
                st.markdown("""
                | Flag / Sinal | Condição de Acionamento |
                | :--- | :--- |
                | **Flag_VSS** | Sinal do sensor de velocidade ativo. |
                | **Flag_RPM** | Sinal do sensor de rotação ativo. |
                | **Flag_ParkNeutral** | Sensor de Câmbio Park/Neutral. |
                | **Flag_TorqueCtrl** | Controle de torque ativo. |
                | **Flag_TPS_Lenta** | Borboleta fechada (Lenta reconhecida). |
                | **Flag_TPS_Plena** | WOT (Aceleração total reconhecida). |
                | **Flag_AC_Pressao** | Sensor de alta pressão do A/C. |
                | **Flag_AC_Botao** | Botão de acionamento do A/C ligado. |
                | **Flag_Diag_Rqst** | Modo de diagnóstico solicitado. |
                | **Flag_Malha_Fechada**| Sonda lambda assumiu leitura ativa. |
                | **Flag_EGR_Ativa** | Válvula EGR acionada. |
                | **Flag_Knock** | Detecção de batida de pino (Sensor Detonação). |
                | **Flag_AC_Embreagem** | Relé da embreagem do A/C ativado. |
                | **Flag_Bomba_Comb**| Relé da bomba acionado. |
                | **Flag_Check_Engine**| Luz de anomalia do painel acesa. |
                | **Flag_Mistura** | Indicação de mistura rica/pobre imediata. |
                | **Flag_MotorCil** | Flag de configuração do tipo de motor. |
                | **Flag_Transmissao** | Flag de configuração da transmissão. |
                | **Flag_Imob_Rec** | Sinal receptor do Imobilizador ativo. |
                | **Flag_Imob_Act** | Imobilizador ativo/bloqueando a ECU. |
                | **Flag_Ventoinha** | Acionamento do eletroventilador. |
                """)
                
            st.markdown("---")
            st.markdown("#### ⚠️ Códigos de Erros Mapeados (DTCs)")
            dtc_df = pd.DataFrame(list(DTC_TABLE.items()), columns=["Código da Falha", "Descrição da Avaria ECU"])

            # Ajuste fino: definimos larguras específicas e removemos o use_container_width
            st.dataframe(
                dtc_df, 
                hide_index=True, 
                use_container_width=False, # Definimos como False para não forçar a largura total
                column_config={
                    "Código da Falha": st.column_config.NumberColumn("Código", width=100),
                    "Descrição da Avaria ECU": st.column_config.TextColumn("Descrição da Avaria", width="600"),
                }
            )
