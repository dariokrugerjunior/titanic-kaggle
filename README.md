# 🚢 Titanic — Previsão de Sobrevivência

Solução completa para a competição [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) do Kaggle.

## 📊 Resultado Final

| Métrica | Valor |
|---|---|
| Acurácia CV (5 folds) | **84.62%** |
| Modelo vencedor | **XGBoost** |

---

## 🗂️ Estrutura do projeto

```
titanic-kaggle/
│
├── notebook_melhorado.ipynb   # Pipeline completo
├── submissao.csv              # Arquivo de submissão gerado
├── curva_rfecv.png            # Curva de seleção de atributos
├── importancia_atributos.png  # Importância dos atributos (XGBoost)
└── README.md                  # Este arquivo
```

---

## 🔧 Pipeline

### 1. Engenharia de atributos
Criação de novas variáveis a partir dos dados brutos:

| Atributo | Descrição |
|---|---|
| `TamanhoFamilia` | Número total de membros da família a bordo |
| `Sozinho` | Flag: passageiro viajou sozinho? |
| `GrupoFamilia` | Categoria: sozinho / pequena / grande |
| `Titulo` | Título extraído do nome (Mr, Mrs, Miss, Raro...) |
| `TemCabine` | Flag: possui cabine registrada? |
| `DeckCabine` | Letra do deck da cabine (A–G) |
| `FreqBilhete` | Número de passageiros com o mesmo bilhete |
| `TarifaPorPessoa` | Tarifa dividida pelo grupo do bilhete |
| `FaixaIdade` | Faixa etária (criança / adolescente / adulto / meia-idade / idoso) |
| `FaixaTarifa` | Quartil da tarifa (baixa / média / alta / muito-alta) |

### 2. Tratamento de nulos
- **Idade**: preenchida com a mediana por `Pclass` + `Sexo` (calculada apenas no treino)
- **Porto de embarque**: preenchido com a moda do treino
- **Tarifa**: preenchida com a mediana do treino

### 3. Seleção de atributos (RFECV)
Eliminação recursiva de atributos com validação cruzada (`RFECV`) usando XGBoost como estimador base. Remove automaticamente atributos redundantes ou ruidosos.

### 4. Ajuste de hiperparâmetros
`GridSearchCV` com `StratifiedKFold(5)` aplicado individualmente em:
- 🌲 Floresta Aleatória (`RandomForestClassifier`)
- ⚡ XGBoost (`XGBClassifier`)
- 💡 LightGBM (`LGBMClassifier`)

### 5. Ensemble
Dois métodos de combinação avaliados:
- **Votação suave**: média das probabilidades dos 3 modelos
- **Empilhamento**: RF + XGB + LGB como base, Regressão Logística como meta-modelo

---

## 📈 Comparação dos modelos

| Modelo | Acurácia CV | Desvio Padrão |
|---|---|---|
| **XGBoost** | **84.62%** | ±1.23% |
| Empilhamento | 84.28% | ±1.80% |
| Votação suave | 83.61% | ±1.77% |
| Floresta Aleatória | 83.28% | ±1.17% |
| LightGBM | 82.72% | ±1.64% |

---

## 🏆 Top atributos (XGBoost)

| Atributo | Importância |
|---|---|
| `Titulo_Mr` | 46.1% |
| `Pclass` | 9.5% |
| `Titulo_Raro` | 8.1% |
| `FreqBilhete` | 7.9% |
| `TemCabine` | 7.4% |

> **Insight principal:** `Titulo_Mr` domina com quase metade do poder preditivo — homens adultos foram os que mais morreram pela regra "mulheres e crianças primeiro".

---

## 🛠️ Tecnologias utilizadas

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-lightblue?logo=pandas)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)

---

## 🚀 Como reproduzir

1. Clone o repositório:
```bash
git clone https://github.com/SEU_USUARIO/titanic-kaggle.git
```

2. Instale as dependências:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
```

3. Faça o download dos dados na [página da competição](https://www.kaggle.com/competitions/titanic/data) e coloque na pasta `/kaggle/input/titanic/`

4. Execute o notebook `notebook_melhorado.ipynb`
