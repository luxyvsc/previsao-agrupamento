import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import requests


# Cache para carregar modelos apenas uma vez
@st.cache_resource
def load_encoder():
    try:
        return joblib.load('encoder.pkl')
    except Exception as e:
        st.error(f'Erro ao carregar encoder: {e}')
        return None

@st.cache_resource
def load_scaler():
    try:
        return joblib.load('scaler.pkl')
    except Exception as e:
        st.error(f'Erro ao carregar scaler: {e}')
        return None

@st.cache_resource
def load_kmeans():
    try:
        return joblib.load('kmeans.pkl')
    except Exception as e:
        st.error(f'Erro ao carregar kmeans: {e}')
        return None

encoder = load_encoder()
scaler = load_scaler()
kmeans = load_kmeans()


st.title('Grupos de interesse para marketing')
st.write("""
Neste projeto, aplicamos o algoritmo de clusterização K-means para identificar e prever agrupamentos de interesses de usuários, com o objetivo de direcionar campanhas de marketing de forma mais eficaz.
Através dessa análise, conseguimos segmentar o público em bolhas de interesse, permitindo a criação de campanhas personalizadas e mais assertivas, com base nos padrões de comportamento e preferências de cada grupo.
""")


up_file = st.file_uploader('Escolha um arquivo CSV para realizar a previsão', type='csv')


def validar_dados(df):
    """Valida se o DataFrame possui as colunas esperadas."""
    colunas_esperadas = ['sexo']
    for col in colunas_esperadas:
        if col not in df.columns:
            return False, f'Coluna obrigatória ausente: {col}'
    return True, ''

def processar_prever(df):
    """Processa o DataFrame e retorna os clusters."""
    try:
        valido, msg = validar_dados(df)
        if not valido:
            st.error(msg)
            return None
        encoded_sexo = encoder.transform(df[['sexo']])
        encoded_df = pd.DataFrame(encoded_sexo, columns=encoder.get_feature_names_out(['sexo']))
        dados = pd.concat([df.drop('sexo', axis=1), encoded_df], axis=1)
        dados_escalados = scaler.transform(dados)
        cluster = kmeans.predict(dados_escalados)
        return cluster
    except Exception as e:
        st.error(f'Erro ao processar dados: {e}')
        return None


def gerar_descricao_hf(df, token: str, model_id: str = "google/flan-t5-large") -> str:
    """Gera descrição dos grupos usando a Hugging Face Inference API com fallback de modelos serverless."""

    # Reduz prompt e evita payload muito grande
    prompt = (
        "Você é um especialista em análise de dados. Recebeu um DataFrame com os seguintes grupos identificados: "
        f"{sorted(set(map(int, pd.Series(df['grupos']).unique())))}\n"
        "Abaixo está uma amostra (até 3 linhas) dos dados de cada grupo. Gere uma breve descrição para cada grupo, com padrões e interesses.\n"
    )
    for g in sorted(pd.Series(df['grupos']).unique()):
        amostra = df[df['grupos'] == g].head(3).to_dict(orient='records')
        prompt += f"\nGrupo {int(g)} (amostra): {amostra}\n"
    prompt += "\nResponda em português, em parágrafos curtos com títulos por grupo."

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Lista de modelos serverless seguros (ordem de tentativa)
    candidates = [model_id, "google/flan-t5-base", "MBZUAI/LaMini-Flan-T5-248M", "google/flan-t5-large"]

    def _call(model: str):
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 350, "return_full_text": False, "temperature": 0.2}}
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        return resp

    last_error = None
    for model in candidates:
        try:
            resp = _call(model)
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    if 'generated_text' in result[0]:
                        return result[0]['generated_text']
                    if 'error' in result[0]:
                        last_error = result[0]['error']
                        # Erros de provider: tente próximo modelo
                        if "Providers" in last_error or "permissions" in last_error:
                            continue
                if isinstance(result, dict):
                    if 'generated_text' in result:
                        return result['generated_text']
                    if 'error' in result:
                        last_error = result['error']
                        if "Providers" in last_error or "permissions" in last_error:
                            continue
                # Se retornou algo diferente mas sem erro explícito
                return str(result)
            else:
                last_error = resp.text
                if "Providers" in last_error or "permissions" in last_error:
                    continue
        except Exception as e:
            last_error = str(e)
            continue

    return f"Erro Hugging Face: {last_error or 'Falha ao gerar descrição.'}"


if up_file is not None:
    try:
        df = pd.read_csv(up_file)
        st.success('Arquivo carregado com sucesso!')
        cluster = processar_prever(df)
        if cluster is not None:
            df.insert(0, 'grupos', cluster)

            # Geração automática de descrição via Hugging Face (se HF_TOKEN estiver configurado)
            st.markdown('### Descrição dos Grupos via Hugging Face')
            hf_token = (st.secrets.get('HF_TOKEN') if hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets else os.environ.get('HF_TOKEN'))
            if hf_token:
                # Permitir override do modelo por secret/variável de ambiente e escolha na UI
                default_model = (st.secrets.get('MODEL_ID') if hasattr(st, 'secrets') and 'MODEL_ID' in st.secrets else os.environ.get('MODEL_ID', 'google/flan-t5-large'))
                safe_models = ['google/flan-t5-large', 'google/flan-t5-base', 'MBZUAI/LaMini-Flan-T5-248M']
                try:
                    default_index = safe_models.index(default_model) if default_model in safe_models else 0
                except ValueError:
                    default_index = 0
                selected_model = st.selectbox('Modelo para descrição (com fallback automático):', options=safe_models, index=default_index)
                with st.spinner(f'Consultando Hugging Face ({selected_model})...'):
                    descricao = gerar_descricao_hf(df, hf_token, selected_model)
                # Renderização com bom contraste em qualquer tema (sem fundo customizado)
                st.markdown(descricao)
            else:
                st.warning('Defina o token da Hugging Face (HF_TOKEN) nas variáveis de ambiente ou em st.secrets para gerar a descrição automática.')

            st.write('Visualização dos resultados (10 primeiros registros):')
            st.dataframe(df.head(10), use_container_width=True)

            # Visualização gráfica da distribuição dos grupos
            st.markdown('**Distribuição dos grupos:**')
            st.bar_chart(df['grupos'].value_counts().sort_index())

            # Permite ao usuário escolher o nome do arquivo de download
            nome_arquivo = st.text_input('Nome do arquivo para download:', value='Grupos_interesse.csv')
            csv = df.to_csv(index=False)
            st.download_button(label='Baixar resultados completos', data=csv, file_name=nome_arquivo, mime='text/csv')
    except Exception as e:
        st.error(f'Erro ao ler o arquivo: {e}')