# Grupos de interesse para marketing

Aplicação Streamlit que prevê clusters (K-means) a partir de um CSV e gera uma descrição automática dos grupos via Hugging Face Inference API.

## Funcionalidades
- Upload de CSV e previsão com modelos pré-treinados (`encoder.pkl`, `scaler.pkl`, `kmeans.pkl`).
- Validação de colunas essenciais (ex.: `sexo`).
- Visualização (amostra + distribuição por grupos).
- Descrição automática dos grupos (Hugging Face) com seletor de modelo na UI e fallback entre modelos serverless gratuitos.
- Download do CSV anotado com a coluna `grupos`.

## Requisitos
- Python 3.11 (o projeto inclui `runtime.txt` para o Streamlit Cloud).
- Artefatos na raiz: `encoder.pkl`, `scaler.pkl`, `kmeans.pkl`.

## Executar localmente (Windows PowerShell)
```powershell
# 1) Ambiente virtual
py -m venv venv
./venv/Scripts/Activate

# 2) Dependências
py -m pip install --upgrade pip
py -m pip install -r .\requirements.txt

# 3) (Opcional) Secrets locais
#   .streamlit/secrets.toml:
#   [default]
#   HF_TOKEN = "hf_SEU_TOKEN"
#   MODEL_ID = "google/flan-t5-large"

# 4) Rodar o app
py -m streamlit run .\App.py
```

Dicas locais:
- Se `pandas` falhar na instalação, prefira Python 3.11 e pip atualizado. Alternativa: `py -m pip install --only-binary=:all: pandas`.

## Hugging Face – modelos e tokens
- Gere um token com permissão `read`: https://huggingface.co/settings/tokens
- Modelos serverless gratuitos suportados (sem “Inference Providers”):
   - `google/flan-t5-large` (padrão)
   - `google/flan-t5-base`
   - `MBZUAI/LaMini-Flan-T5-248M`
- O app possui seletor de modelo na UI e fallback automático se houver erro de providers/permissões.

## Solução de problemas
- Erro: `This authentication method does not have sufficient permissions to call Inference Providers ...`
   - Use os modelos acima. O app já faz fallback; você também pode trocar no seletor da UI.
- Resposta lenta na primeira chamada: “model loading”; aguarde e tente de novo.
- Descrição vazia: verifique se `HF_TOKEN` está definido nos Secrets/env no deploy.

## Estrutura do projeto
- `App.py`: aplicação Streamlit.
- `encoder.pkl`, `scaler.pkl`, `kmeans.pkl`: pipeline de predição.
- `requirements.txt`: dependências.
- `runtime.txt`: versão do Python.
- `.gitignore`: ignora `venv/`, `.streamlit/secrets.toml`, CSVs locais, etc.
- `README.md`: este arquivo.

## Segurança
- Não versione segredos ou ambientes virtuais.
- Caso um token seja exposto, revogue e gere outro na Hugging Face.
