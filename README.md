# Grupos de interesse para marketing

Aplicação Streamlit que prevê clusters (K-means) a partir de um CSV e gera uma descrição automática dos grupos via Hugging Face Inference API.

## Funcionalidades
- Upload de CSV e previsão com modelos pré-treinados (`encoder.pkl`, `scaler.pkl`, `kmeans.pkl`).
- Validação de colunas essenciais (ex.: `sexo`).
- Visualização (amostra + distribuição por grupos).
- (Temporariamente removido) Descrição via IA. O app foca apenas na previsão, visualização e download. Podemos reativar depois.
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

## (Pausado) IA para descrição
Esta funcionalidade foi removida por enquanto para simplificar e evitar erros de permissão. 

## Solução de problemas
- Instalação no Windows: use Python 3.11 e pip atualizado se `pandas` falhar na instalação.

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
