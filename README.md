
# SISTEMA DE DETECÇÃO E RECONHECIMENTO DE LETREIROS DE ÔNIBUS

ESTRUTURA DE DIRETÓRIOS E ARTEFATOS


1. CÓDIGOS-FONTE PRINCIPAIS
   -------------------------

   bus_sign_system.py
      Descrição: Script principal do sistema. Coordena todas as
                 etapas do pipeline: preparação do dataset, treinamento/
                 carregamento do modelo YOLO, validação, processamento OCR e
                 comparação de resultados.
      Parâmetros: --config [caminho_config.yaml]
                  Opcional. Define o arquivo de configuração a ser usado.
                  Padrão: config.yaml
      Execução: python bus_sign_system.py
                python bus_sign_system.py --config custom_config.yaml
      Saídas: Gera logs, métricas, imagens de erro e comparações em
              ./logs/, ./resultados_ocr/, ./misdetected/, ./misocr/

   tcc2.py
      Descrição: Código original (legado) do sistema. Mantido para referência
                 e comparação. Contém toda a lógica em um único arquivo
                 monolítico (~1450 linhas).
      Parâmetros: Nenhum (configurações hardcoded no código)
      Execução: python tcc2.py
      Nota: Recomenda-se usar bus_sign_system.py (versão refatorada)


2. MÓDULOS AUXILIARES (ARQUITETURA MODULAR)

   dataset_manager.py
      Descrição: Gerenciador de dataset YOLO. Responsável por validar a
                 estrutura do dataset, criar arquivo data.yaml, carregar
                 ground truths e fornecer estatísticas.
      Funções principais:
         - verify_dataset(): Valida estrutura train/valid/test
         - get_dataset_stats(): Retorna contagem de imagens por split
         - load_ground_truth_boxes(): Carrega anotações YOLO
         - get_test_images(): Lista imagens do conjunto de teste

   ocr_manager.py
      Descrição: Gerenciador centralizado de múltiplos engines de OCR.
                 Implementa pattern Strategy para suportar Tesseract, EasyOCR
                 e OpenOCR de forma modular e extensível.
      Classes:
         - OCREngine: Classe abstrata base
         - TesseractOCR: Engine Tesseract OCR
         - EasyOCREngine: Engine EasyOCR
         - OpenOCREngine: Engine OpenOCR (SVTRv2)
         - OCRManager: Gerenciador principal
      Configuração: Via config.yaml seção 'ocr'

   text_similarity.py
      Descrição: Implementação de algoritmos de similaridade textual usando
                 Levenshtein Distance (Edit Distance). Utilizado para comparar
                 texto extraído por OCR com ground truth.
      Algoritmo: Programação Dinâmica - O(n*m) onde n,m são tamanhos das strings
      Métricas:
         - levenshtein_distance(): Distância bruta (número de edições)
         - normalized_similarity(): Similaridade normalizada 0-100
         - partial_similarity(): Melhor match em substrings
         - find_best_match(): Busca melhor candidato em lista

   utils.py
      Descrição: Funções utilitárias gerais para processamento de imagens,
                 cálculo de IoU, sanitização de nomes de arquivo, logging e
                 coleta de métricas.
      Funções principais:
         - safe_open_image(): Abertura segura de imagens com tratamento de erros
         - sanitize_filename(): Remove caracteres inválidos para Windows
         - calculate_iou(): Calcula Intersection over Union entre boxes
         - match_predictions_to_ground_truth(): Matching usando IoU greedy
         - MetricsCollector: Classe para agregação de métricas


3. ARQUIVOS DE CONFIGURAÇÃO

   config.yaml
      Descrição: Arquivo principal de configuração do sistema. Define caminhos,
                 parâmetros de treinamento YOLO, configurações de OCR, thresholds
                 de análise e lista de 271 classes (rotas de ônibus).
      Seções:
         - paths: Diretórios base e de saída
         - training: Parâmetros YOLO (epochs, batch, imgsz, device, etc.)
         - ocr: Configurações de Tesseract, EasyOCR e OpenOCR
         - analysis: Thresholds de IoU e similaridade
         - classes: 271 nomes de rotas de ônibus de Curitiba
      Modificações: Ajuste paths.base_dir conforme seu ambiente

   requirements.txt
      Descrição: Lista de dependências Python do projeto.
      Instalação: pip install -r requirements.txt
      Principais bibliotecas:
         - ultralytics: YOLOv8
         - pytesseract: Tesseract OCR
         - easyocr: EasyOCR
         - opencv-python: Processamento de imagens
         - PyYAML: Parsing de configurações


4. DOCUMENTAÇÃO

   README.md
      Descrição: Documentação completa do projeto em formato Markdown. Inclui
                 instruções de instalação, configuração, execução, estrutura
                 do projeto, métricas coletadas e troubleshooting.
      Conteúdo: Pré-requisitos, instalação, configuração, fluxo de execução,
                métricas, comparação código original vs refatorado

   exemplo_uso.py
      Descrição: Exemplos práticos de uso do sistema. Demonstra diferentes
                 casos de uso: execução completa, apenas OCR, comparação de
                 engines, uso de módulos individuais, etc.
      Exemplos incluídos:
         1. Execução básica completa
         2. Apenas OCR (modelo pré-treinado)
         3. Comparar engines de OCR
         4. Similaridade de texto
         5. Gerenciador de dataset
         6. OCR em imagem única
         7. Criar configuração customizada
      Execução: python exemplo_uso.py


5. DATASET 
   ./dataset_yolo/
      Descrição: Dataset no formato YOLO com estrutura train/valid/test.
      Estrutura esperada:
         dataset_yolo/
         ├── train/
         │   ├── images/    (imagens de treino)
         │   └── labels/    (anotações .txt YOLO)
         ├── valid/
         │   ├── images/    (imagens de validação)
         │   └── labels/    (anotações .txt YOLO)
         ├── test/
         │   ├── images/    (imagens de teste)
         │   └── labels/    (anotações .txt YOLO)
         └── data.yaml      (gerado automaticamente pelo sistema)

      Formato das anotações YOLO:
         <class_id> <x_center> <y_center> <width> <height>
         Valores normalizados (0.0 a 1.0)

      Classes: 271 rotas de ônibus (definidas em config.yaml)

   yolov8n.pt
      Descrição: Modelo YOLOv8 Nano pré-treinado (base para transfer learning).
      Download: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
      Uso: Base para treinamento inicial


6. DIRETÓRIOS DE SAÍDA (GERADOS AUTOMATICAMENTE)
   ./onibus_tcc_treino_letreiros/
      Descrição: Diretório principal de saída do treinamento YOLO.
      Conteúdo:
         letreiros_seg/
         ├── weights/
         │   ├── best.pt          (melhor modelo durante treinamento)
         │   └── last.pt          (último checkpoint)
         ├── results.csv          (métricas por época)
         ├── results.png          (gráficos de treinamento)
         ├── confusion_matrix.png (matriz de confusão)
         ├── F1_curve.png
         ├── P_curve.png          (curva de precisão)
         ├── R_curve.png          (curva de revocação)
         └── PR_curve.png         (precisão-revocação)

   ./logs/
      Descrição: Logs de execução do sistema com timestamp.
      Conteúdo: Registro completo de todas as operações, métricas, warnings e erros
      Uso: Rastreamento de execução, debugging, auditoria

   ./resultados_ocr/
      Descrição: Resultados organizados por engine de OCR.
      Estrutura:
         erros_ocr_tesseract/        (OCR incorreto - Tesseract)
         erros_ocr_easyocr/          (OCR incorreto - EasyOCR)
         erros_ocr_openocr/          (OCR incorreto - OpenOCR)
         erros_classificacao_tesseract/  (Classe YOLO errada - Tesseract)
         erros_classificacao_easyocr/    (Classe YOLO errada - EasyOCR)
         erros_classificacao_openocr/    (Classe YOLO errada - OpenOCR)
      Formato dos arquivos:
         <imagem_original>_real_<classe_verdadeira>_ocr_<texto_ocr>.jpg
      Uso: Análise qualitativa de erros, identificação de padrões de falha

   ./misdetected/
      Descrição: Erros de detecção do modelo YOLO.
      Conteúdo:
         - missed_gt: Ground truths não detectados (falsos negativos)
         - false_positive: Detecções sem ground truth correspondente
         - class_mismatch: Detecções com classe incorreta
         misdetected_log.csv: Log estruturado de todos os erros
      Formato CSV: [image, type, pred_cls, pred_box, gt_cls, gt_box, iou, note]

   ./misocr/
      Descrição: Erros específicos de OCR (texto reconhecido incorretamente).
      Conteúdo: Crops de letreiros onde OCR falhou
         misocr_log.csv: Log estruturado dos erros de OCR
      Formato CSV: [image, true_cls, ocr_text, best_match, similarity, crop_path, note]

   ./imagens_erro_leitura/
      Descrição: Imagens corrompidas ou com erro de leitura.
      Conteúdo: Cópias de imagens que falharam ao serem abertas
      Formato: erro_<nome_original>.jpg
      Uso: Identificar problemas no dataset

   ./dataset_yolo/temp_crops/
      Descrição: Crops temporários de letreiros durante processamento OCR.
      Conteúdo:
         - crop_tesseract.jpg, crop_easyocr.jpg, crop_openocr.jpg
         - *_empty_sample_*.jpg: Amostras onde OCR retornou vazio
      Limpeza: Sobrescritos a cada execução


7. DIRETÓRIOS ADICIONAIS

   ./src/
      Descrição: Módulos internos do sistema (arquitetura modular).
      Função: Contém todo o código-fonte organizado em componentes reutilizáveis
      Conteúdo:
         - dataset_manager.py: Gerenciamento de dataset YOLO
         - ocr_manager.py: Gerenciamento de engines OCR (Tesseract/EasyOCR/OpenOCR)
         - text_similarity.py: Algoritmos de similaridade textual (Levenshtein)
         - utils.py: Funções auxiliares gerais
         - __init__.py: Configuração do módulo Python
      Importado por: main.py

   ./docs/
      Descrição: Documentação completa do projeto.
      Função: Armazena toda a documentação técnica e acadêmica
      Conteúdo:
         - README.md: Documentação técnica detalhada
         - readme.txt: Especificações para entrega do TCC (este arquivo)
         - REFATORACAO_RESUMO.md: Análise comparativa do projeto

   ./examples/
      Descrição: Exemplos práticos de uso do sistema.
      Função: Demonstra diferentes formas de utilizar os módulos
      Conteúdo:
         - exemplo_uso.py: 7 exemplos de uso (básico, OCR isolado, etc.)
      Execução: python examples/exemplo_uso.py

   ./OpenOCR/
      Descrição: Biblioteca OpenOCR (Scene Text Recognition - SVTRv2).
      Tamanho: ~500 MB
      Função: Fornece engine de OCR baseado em deep learning
      Modelo: SVTRv2 (state-of-the-art para reconhecimento de texto em cenas)
      Usado por: src/ocr_manager.py (classe OpenOCREngine)
      Opcional: Sistema funciona apenas com Tesseract/EasyOCR se OpenOCR não estiver disponível


8. MÉTRICAS E RESULTADOS
   

   MÉTRICAS DE DETECÇÃO (YOLO):
      - mAP50-95: Mean Average Precision (IoU 0.5 a 0.95)
      - mAP50: Mean Average Precision (IoU 0.5)
      - mAP75: Mean Average Precision (IoU 0.75)
      - Precisão média (mp)
      - Revocação média (mr)
      Localização: Exibidas no terminal e salvas em ./logs/

   MÉTRICAS DE OCR:
      Para cada engine (Tesseract, EasyOCR, OpenOCR):
      - Total de letreiros analisados
      - Acertos exatos: OCR idêntico ao ground truth
      - Acertos aproximados: Similaridade Levenshtein >= 80%
      - Acurácia exata (%): exact_matches / total * 100
      - Acurácia total (%): (exact + approx) / total * 100
      - Taxa de sucesso OCR: detections_with_ocr / total_detections * 100
      - OCR vazios: Contagem de crops sem texto reconhecido
      Localização: Logs + tabela comparativa no terminal

   TABELA COMPARATIVA FINAL:
      Compara side-by-side os três engines de OCR em:
      - Total analisados
      - Acertos exatos
      - Acurácia exata (%)
      - Acurácia total (%)
      - Taxa de sucesso (%)
      Identifica automaticamente o melhor engine


9. FLUXO DE EXECUÇÃO DETALHADO

   Etapa 1: Preparação do Dataset
      - Verifica estrutura train/valid/test
      - Cria/valida data.yaml
      - Exibe estatísticas
      - Valida integridade dos arquivos

   Etapa 2: Treinamento/Carregamento do Modelo YOLO
      - Verifica se modelo treinado existe em ./onibus_tcc_treino_letreiros/
      - Se SIM: Carrega modelo existente 
      - Se NÃO: Treina novo modelo com parâmetros de config.yaml
         * Carrega yolov8n.pt como base
         * Treina por N epochs com early stopping (patience)
         * Salva checkpoints best.pt e last.pt
         * Gera gráficos e matrizes de confusão

   Etapa 3: Validação do Modelo
      - Executa validação no conjunto de teste
      - Calcula métricas mAP, precisão, revocação
      - Registra resultados no log

   Etapa 4: Processamento OCR Multi-Engine
      Para cada engine habilitado (Tesseract, EasyOCR, OpenOCR):
      a) Inicializa engine (se disponível)
      b) Para cada imagem de teste:
         - Detecta letreiros com YOLO (conf >= threshold)
         - Recorta região do letreiro
         - Aplica OCR no crop
         - Compara texto OCR com ground truth (Levenshtein)
         - Classifica como acerto exato/aproximado/erro
         - Salva crops de erro e acerto em diretórios específicos
      c) Agrega métricas por engine
      d) Registra resultados

   Etapa 5: Comparação Final
      - Gera tabela comparativa dos três engines
      - Identifica melhor engine por métrica
      - Exibe recomendação final
      - Salva relatório completo no log

   Etapa 6: Finalização
      - Exibe sumário de diretórios de saída
      - Registra caminho do log completo
      - Retorna código de sucesso/erro


10. PARÂMETROS DE CONFIGURAÇÃO IMPORTANTES

   TREINAMENTO YOLO (config.yaml -> training):
      epochs: 150              (número de épocas de treinamento)
      patience: 15             (early stopping após N épocas sem melhora)
      imgsz: 640               (tamanho de entrada da imagem)
      batch: 16                (tamanho do batch)
      device: 0                (GPU - 0, 1, 2... ou 'cpu')
      confidence: 0.4          (threshold de confiança para detecções)
      single_cls: true         (tratar todas as classes como "letreiro")

   OCR TESSERACT (config.yaml -> ocr.tesseract):
      psm: 7                   (Page Segmentation Mode - linha única)
      lang: eng                (idioma do Tesseract)
      enabled: true            (habilitar/desabilitar engine)

   OCR EASYOCR (config.yaml -> ocr.easyocr):
      languages: [pt, en]      (idiomas suportados)
      gpu: true                (usar GPU se disponível)
      enabled: true

   OCR OPENOCR (config.yaml -> ocr.openocr):
      config_path: configs/rec/svtrv2/svtrv2_ch.yml
      backend: torch           (backend PyTorch)
      enabled: true

   ANÁLISE (config.yaml -> analysis):
      iou_threshold: 0.5       (IoU mínimo para matching pred-GT)
      similarity_threshold: 80 (% mínimo Levenshtein para acerto aproximado)
      max_empty_samples: 10    (máximo de samples vazios para salvar)


11. DEPENDÊNCIAS E INSTALAÇÃO

    PYTHON: 3.8 ou superior

    BIBLIOTECAS PYTHON:
       pip install -r requirements.txt

       Principais:
       - ultralytics >= 8.0.0     (YOLOv8)
       - numpy >= 1.21.0
       - opencv-python >= 4.5.0
       - Pillow >= 9.0.0
       - PyYAML >= 6.0
       - tqdm >= 4.65.0
       - pytesseract >= 0.3.10    (Tesseract OCR)
       - easyocr >= 1.7.0         (EasyOCR)

    TESSERACT OCR (Opcional):
       Windows: https://github.com/UB-Mannheim/tesseract/wiki
       Instalar em: C:\Program Files\Tesseract-OCR

    OPENOCR:
       git clone https://github.com/Topdu/OpenOCR
       cd OpenOCR
       pip install -r requirements.txt

    GPU (Opcional, mas recomendado):
       CUDA Toolkit + cuDNN
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118


12. INSTRUÇÕES DE USO

    EXECUÇÃO BÁSICA:
       1. Ajustar config.yaml (paths.base_dir)
       2. Preparar dataset em ./dataset_yolo/
       3. Baixar yolov8n.pt
       4. Executar: python bus_sign_system.py

    APENAS OCR (MODELO JÁ TREINADO):
       Modificar bus_sign_system.py para pular treinamento ou
       executar exemplo_uso.py -> Opção 2

    PROCESSAR APENAS UM ENGINE:
       Desabilitar engines não desejados em config.yaml:
       ocr:
         tesseract:
           enabled: true
         easyocr:
           enabled: false
         openocr:
           enabled: false

