import streamlit as st
import pandas as pd
import joblib
import os


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
st.caption('Ou use o arquivo de exemplo da raiz do projeto: novas_entradas.csv')
usar_exemplo = st.button('Usar exemplo (novas_entradas.csv)')


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


df = None
fonte = None
if up_file is not None:
    try:
        df = pd.read_csv(up_file)
        fonte = 'upload'
    except Exception as e:
        st.error(f'Erro ao ler o arquivo enviado: {e}')
elif usar_exemplo:
    exemplo_path = 'novas_entradas.csv'
    if os.path.exists(exemplo_path):
        try:
            df = pd.read_csv(exemplo_path)
            fonte = 'exemplo'
        except Exception as e:
            st.error(f'Erro ao ler o arquivo de exemplo: {e}')
    else:
        st.error('Arquivo de exemplo novas_entradas.csv não encontrado na raiz do projeto.')

if df is not None:
    st.success('Arquivo carregado com sucesso!' + (" (exemplo)" if fonte == 'exemplo' else ''))
    cluster = processar_prever(df)
    if cluster is not None:
        df.insert(0, 'grupos', cluster)

        # Descrição via IA removida temporariamente

        st.write('Visualização dos resultados (10 primeiros registros):')
        st.dataframe(df.head(10), use_container_width=True)

        # Visualização gráfica da distribuição dos grupos
        st.markdown('**Distribuição dos grupos:**')
        st.bar_chart(df['grupos'].value_counts().sort_index())

        # Permite ao usuário escolher o nome do arquivo de download
        nome_arquivo = st.text_input('Nome do arquivo para download:', value='Grupos_interesse.csv')
        csv = df.to_csv(index=False)
        st.download_button(label='Baixar resultados completos', data=csv, file_name=nome_arquivo, mime='text/csv')