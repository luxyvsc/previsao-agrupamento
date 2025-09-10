# Grupos de interesse para marketing

Aplicação Streamlit que carrega modelos (encoder, scaler e k-means) para prever clusters de interesse a partir de um CSV e gera uma descrição automática dos grupos via Hugging Face Inference API.

## Funcionalidades
- Upload de CSV e previsão de cluster com modelos pré-treinados (`encoder.pkl`, `scaler.pkl`, `kmeans.pkl`).
- Validação de colunas essenciais (ex.: `sexo`).
- Visualização de amostra dos resultados e distribuição dos grupos.
- Geração opcional de descrição dos grupos via Hugging Face (token em secrets/env).
- Download do CSV anotado com a coluna `grupos`.

## Estrutura
- `App.py`: aplicação Streamlit.
- `encoder.pkl`, `scaler.pkl`, `kmeans.pkl`: artefatos do modelo (devem estar na raiz).
- `requirements.txt`: dependências do projeto.
- `runtime.txt`: versão do Python para deploy (Streamlit Cloud).

## Executar localmente
1. Crie e ative um ambiente virtual.
2. Instale dependências:
   - Caso encontre erro ao instalar `pandas` no Windows local, use Python 3.11.
3. Rode o app:
   ```bash
   streamlit run App.py
   ```

## Integração Hugging Face (opcional)
- Crie um token em https://huggingface.co/settings/tokens (tipo: `read`).
- Localmente, crie `.streamlit/secrets.toml` com:
  ```toml
  [default]
  HF_TOKEN = "hf_SEU_TOKEN"
  MODEL_ID = "google/flan-t5-large"
  ```
  ```

## Observações
- Não versione segredos ou ambientes virtuais. O `.gitignore` já inclui `venv/`, `.streamlit/secrets.toml`, arquivos temporários e CSVs locais.
- Se trocar o esquema de dados, atualize a validação e o pipeline de pré-processamento.
