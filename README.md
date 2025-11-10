# Sistema de Detecção e Reconhecimento de Letreiros de Ônibus


---

## Artefatos do Projeto

### 1. Códigos-Fonte Principais

#### `main.py`
Script principal do sistema. Coordena todo o pipeline: preparação do dataset, treinamento/carregamento do modelo YOLO, validação, processamento OCR e comparação de resultados entre os três engines.

**Parâmetros:**
- `--config`: Caminho para arquivo de configuração YAML (padrão: `config.yaml`)

**Saídas geradas:**
- Logs em `./logs/`
- Métricas e comparações no terminal
- Imagens de acertos e erros em `./resultados_ocr/`

---


### 2. Módulos do Sistema (`src/`)

#### `src/dataset_manager.py`
Gerenciador de dataset YOLO. Responsável por:
- Validar estrutura do dataset (train/valid/test)
- Criar arquivo `data.yaml` automaticamente
- Carregar ground truth annotations
- Fornecer estatísticas do dataset

**Principais funções:**
- `verify_dataset()`: Valida estrutura de diretórios
- `get_dataset_stats()`: Retorna contagem de imagens por split
- `load_ground_truth_boxes()`: Carrega anotações YOLO
- `get_test_images()`: Lista imagens do conjunto de teste

---

#### `src/ocr_manager.py`
Gerenciador centralizado de engines de OCR. Implementa pattern Strategy para suportar múltiplos engines de forma modular.

**Classes implementadas:**
- `OCREngine`: Classe abstrata base
- `TesseractOCR`: Engine Tesseract OCR
- `EasyOCREngine`: Engine EasyOCR
- `OpenOCREngine`: Engine OpenOCR (SVTRv2)
- `OCRManager`: Gerenciador principal que coordena os engines

**Configuração:** Via seção `ocr` do arquivo `config.yaml`

---

#### `src/text_similarity.py`
Implementação de algoritmos de similaridade textual usando Levenshtein Distance (Edit Distance). Utilizado para comparar texto extraído por OCR com ground truth.

**Algoritmo:** Programação Dinâmica - Complexidade O(n*m)

**Funções principais:**
- `levenshtein_distance()`: Calcula distância bruta (número de edições necessárias)
- `normalized_similarity()`: Retorna similaridade normalizada (0-100)
- `partial_similarity()`: Encontra melhor match em substrings
- `find_best_match()`: Busca melhor candidato em lista de opções

---

#### `src/utils.py`
Funções utilitárias para processamento de imagens, cálculo de métricas, logging e sanitização.

**Funções principais:**
- `safe_open_image()`: Abertura segura de imagens com tratamento de erros
- `sanitize_filename()`: Remove caracteres inválidos para nomes de arquivo (Windows-safe)
- `calculate_iou()`: Calcula Intersection over Union entre bounding boxes
- `setup_logging()`: Configura sistema de logging com timestamp
- `MetricsCollector`: Classe para agregação e cálculo de métricas

---

### 3. Arquivos de Configuração

#### `config.yaml`
Arquivo principal de configuração do sistema. Centraliza todos os parâmetros configuráveis.

**Seções:**
- **paths**: Diretórios base, dataset, modelo, saídas
- **training**: Parâmetros YOLO (epochs, batch, imgsz, device, confidence, etc.)
- **ocr**: Configurações de Tesseract, EasyOCR e OpenOCR
- **analysis**: Thresholds de IoU e similaridade textual
- **classes**: Lista de 271 rotas de ônibus de Curitiba

**Ajustes necessários:** Modificar `paths.base_dir` conforme seu ambiente

---

### 6. Dataset

#### `dataset_yolo/`
Dataset no formato YOLO com estrutura train/valid/test.

**Estrutura:**
```
dataset_yolo/
├── train/
│   ├── images/    # 898 imagens de treino
│   └── labels/    # Anotações .txt formato YOLO
├── valid/
│   ├── images/    # 224 imagens de validação
│   └── labels/
├── test/
│   ├── images/    # 280 imagens de teste
│   └── labels/
└── data.yaml      # Gerado automaticamente pelo sistema
```

**Formato das anotações:**
```
<class_id> <x_center> <y_center> <width> <height>
```
Valores normalizados (0.0 a 1.0)

**Split:** 64% treino, 16% validação, 20% teste
**Total:** 1402 imagens
**Classes:** 271 rotas de ônibus de Curitiba

---

#### `yolov8n.pt`
Modelo YOLOv8 Nano pré-treinado. Usado como base para transfer learning.

**Download:** https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

---

### 4. Diretórios de Saída (Gerados Automaticamente)

#### `onibus_tcc_treino_letreiros/`
Diretório principal de saída do treinamento YOLO.

**Conteúdo:**
```
letreiros_seg/
├── weights/
│   ├── best.pt          # Melhor modelo durante treinamento
│   └── last.pt          # Último checkpoint
├── results.csv          # Métricas por época
├── results.png          # Gráficos de treinamento
├── confusion_matrix.png # Matriz de confusão
├── F1_curve.png
├── P_curve.png          # Curva de precisão
├── R_curve.png          # Curva de revocação
└── PR_curve.png         # Precisão-revocação
```

---

#### `logs/`
Logs de execução com timestamp. Contém registro completo de todas as operações, métricas, warnings e erros.

**Uso:** Rastreamento de execução, debugging.

---

#### `resultados_ocr/`
Resultados organizados por engine de OCR. Contém imagens de acertos e erros separadas por tipo.

**Estrutura:**
```
resultados_ocr/
├── acertos_ocr_tesseract/           # OCR reconheceu corretamente
├── acertos_ocr_easyocr/
├── acertos_ocr_openocr/
├── erros_ocr_tesseract/             # OCR reconheceu incorretamente
├── erros_ocr_easyocr/
├── erros_ocr_openocr/
├── erros_classificacao_tesseract/   # YOLO classificou classe errada
├── erros_classificacao_easyocr/
└── erros_classificacao_openocr/
```

**Formato dos nomes:**
```
<imagem_original>_real_<classe_verdadeira>_ocr_<texto_ocr>.jpg
```

**Uso:** Análise qualitativa de erros, identificação de padrões de falha

---

#### `imagens_erro_leitura/`
Imagens corrompidas ou com erro de leitura.

**Conteúdo:** Cópias de imagens que falharam ao serem abertas pelo sistema

**Formato:** `erro_<nome_original>.jpg`

**Uso:** Identificar e corrigir problemas no dataset

---

#### `dataset_yolo/temp_crops/`
Crops temporários de letreiros durante processamento OCR.

**Conteúdo:**
- `crop_tesseract.jpg`, `crop_easyocr.jpg`, `crop_openocr.jpg`
- `*_empty_*.jpg`: Amostras onde OCR retornou texto vazio

**Limpeza:** Sobrescritos a cada execução

---

### 5. Bibliotecas Externas

#### `OpenOCR/`
Biblioteca OpenOCR (Scene Text Recognition - SVTRv2).

**Função:** Fornece engine de OCR baseado em deep learning

**Modelo:** SVTRv2 (state-of-the-art para reconhecimento de texto em cenas)

**Tamanho:** ~500 MB

**Usado por:** `src/ocr_manager.py` (classe `OpenOCREngine`)

**Opcional:** Sistema funciona apenas com Tesseract/EasyOCR se OpenOCR não estiver disponível

**Instalação:**
```bash
git clone https://github.com/Topdu/OpenOCR
cd OpenOCR
pip install -r requirements.txt
```

---
