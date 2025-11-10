# -*- coding: utf-8 -*-
"""
Sistema de Detecção e Reconhecimento de Letreiros de Ônibus
Versão Refatorada - Arquitetura Modular

Autor: TCC - Sistema de Visão Computacional
"""

import os
import csv
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm  # type: ignore

from src.dataset_manager import DatasetManager
from src.ocr_manager import OCRManager
from src.text_similarity import TextSimilarity
from src.utils import (
    safe_open_image, sanitize_filename, setup_logging,
    create_directory_structure, MetricsCollector
)

logger = logging.getLogger(__name__)


class BusSignRecognitionSystem:
    """Sistema principal de detecção e reconhecimento de letreiros"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa o sistema.

        Args:
            config_path: Caminho para o arquivo de configuração
        """
        self.config = self._load_config(config_path)
        self.base_dir = self.config['paths']['base_dir']

        # Configurar logging
        log_dir = os.path.join(self.base_dir, self.config['paths']['output']['logs'])
        self.log_path = setup_logging(log_dir)

        # Inicializar componentes
        self._setup_directories()
        self._initialize_components()

        logger.info("=" * 80)
        logger.info("Sistema de Detecção e Reconhecimento de Letreiros de Ônibus")
        logger.info("=" * 80)
        logger.info(f"Diretório base: {self.base_dir}")
        logger.info(f"Log salvo em: {self.log_path}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configurações do arquivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Arquivo de configuração não encontrado: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Erro ao parsear arquivo YAML: {e}")
            raise

    def _setup_directories(self):
        """Cria estrutura de diretórios necessária"""
        output_dirs = [
            self.config['paths']['output']['logs'],
            self.config['paths']['output']['temp_crops'],
            self.config['paths']['output']['images_error'],
            self.config['paths']['output']['misdetected'],
            self.config['paths']['output']['misocr'],
            self.config['paths']['output']['results_ocr']
        ]

        for dir_path in output_dirs:
            full_path = os.path.join(self.base_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)

    def _initialize_components(self):
        """Inicializa componentes do sistema"""
        # Dataset Manager
        dataset_dir = os.path.join(self.base_dir, self.config['paths']['dataset_dir'])
        self.dataset_manager = DatasetManager(dataset_dir, self.config)

        # OCR Manager
        openocr_dir = os.path.join(self.base_dir, self.config['paths']['openocr_dir'])
        self.ocr_manager = OCRManager(
            self.config['ocr'],
            openocr_dir if os.path.exists(openocr_dir) else None
        )

        # Text Similarity
        self.text_similarity = TextSimilarity()

        # YOLO Model
        self.model = None

    def prepare_dataset(self):
        """Prepara e valida o dataset"""
        logger.info("\n" + "=" * 80)
        logger.info("PREPARAÇÃO DO DATASET")
        logger.info("=" * 80)

        self.dataset_manager.log_dataset_info()
        self.dataset_manager.ensure_data_yaml()

    def train_or_load_model(self) -> bool:
        """
        Treina o modelo YOLO ou carrega modelo existente.

        Returns:
            True se o modelo foi carregado/treinado com sucesso
        """
        from ultralytics import YOLO

        logger.info("\n" + "=" * 80)
        logger.info("MODELO YOLO")
        logger.info("=" * 80)

        # Caminhos
        model_path = os.path.join(self.base_dir, self.config['paths']['model_path'])
        training_dir = os.path.join(self.base_dir, self.config['paths']['output']['training'])
        trained_model_path = os.path.join(training_dir, 'letreiros_seg', 'weights', 'best.pt')

        # Verificar se modelo treinado existe
        if os.path.exists(trained_model_path):
            logger.info("Modelo treinado encontrado!")
            logger.info(f"Carregando de: {trained_model_path}")
            logger.info("Pulando etapa de treinamento (economizando tempo!)")
            self.model = YOLO(trained_model_path)
            return True

        # Treinar novo modelo
        logger.info("Iniciando treinamento YOLO...")
        logger.info(f"Modelo base: {model_path}")

        if not os.path.exists(model_path):
            logger.error(f"Modelo base não encontrado: {model_path}")
            logger.error("Baixe o YOLOv8n: https://github.com/ultralytics/ultralytics")
            return False

        self.model = YOLO(model_path)

        # Configurações de treinamento
        train_config = self.config['training']
        data_yaml = self.dataset_manager.data_yaml

        logger.info("\nConfigurações de treinamento:")
        for key, value in train_config.items():
            logger.info(f"   {key}: {value}")

        # Treinar
        logger.info("\nTreinamento iniciado (pode levar várias horas)...")

        results = self.model.train(
            data=data_yaml,
            epochs=train_config['epochs'],
            patience=train_config['patience'],
            imgsz=train_config['imgsz'],
            batch=train_config['batch'],
            device=train_config['device'],
            single_cls=train_config['single_cls'],
            pretrained=train_config['pretrained'],
            verbose=train_config['verbose'],
            project=training_dir,
            name='letreiros_seg'
        )

        logger.info("Treinamento finalizado!")
        logger.info(f"Modelo salvo em: {trained_model_path}")

        return True

    def validate_model(self):
        """Valida o modelo no conjunto de teste"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDAÇÃO DO MODELO")
        logger.info("=" * 80)

        metrics = self.model.val(data=self.dataset_manager.data_yaml)

        logger.info("\nResultados:")
        logger.info(f"   mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"   mAP50: {metrics.box.map50:.4f}")
        logger.info(f"   mAP75: {metrics.box.map75:.4f}")
        logger.info(f"   Precisão: {metrics.box.mp:.4f}")
        logger.info(f"   Revocação: {metrics.box.mr:.4f}")

    def process_ocr(self, engine_name: str) -> Dict[str, Any]:
        """
        Processa OCR em todas as imagens de teste usando um engine específico.

        Args:
            engine_name: Nome do engine ('tesseract', 'easyocr', 'openocr')

        Returns:
            Dicionário com métricas do processamento
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"OCR COM {engine_name.upper()}")
        logger.info("=" * 80)

        # Verificar disponibilidade do engine
        engine = self.ocr_manager.get_engine(engine_name)
        if not engine or not engine.is_available():
            logger.warning(f"Engine {engine_name} não disponível. Pulando...")
            return {}

        # Diretórios de saída
        results_dir = os.path.join(self.base_dir, self.config['paths']['output']['results_ocr'])
        wrong_ocr_dir = os.path.join(results_dir, f"erros_ocr_{engine_name}")
        wrong_class_dir = os.path.join(results_dir, f"erros_classificacao_{engine_name}")
        correct_ocr_dir = os.path.join(results_dir, f"acertos_ocr_{engine_name}")
        temp_crops_dir = os.path.join(self.base_dir, self.config['paths']['output']['temp_crops'])

        os.makedirs(wrong_ocr_dir, exist_ok=True)
        os.makedirs(wrong_class_dir, exist_ok=True)
        os.makedirs(correct_ocr_dir, exist_ok=True)

        # Métricas
        metrics = MetricsCollector()
        similarity_threshold = self.config['analysis']['similarity_threshold']
        max_empty_samples = self.config['analysis']['max_empty_samples']

        # Classes
        class_names = self.dataset_manager.class_names

        # Processar imagens
        test_images = self.dataset_manager.get_test_images()
        logger.info(f"Processando {len(test_images)} imagens de teste...")

        for img_name in tqdm(test_images, desc=f"OCR {engine_name}"):
            img_path = self.dataset_manager.get_test_image_path(img_name)

            # Detectar letreiros
            results = self.model.predict(
                source=img_path,
                conf=self.config['training']['confidence'],
                verbose=False
            )

            metrics.update(images_processed=1)
            has_detection = False

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    metrics.update(total_detections=1)
                    has_detection = True

                    # Extrair informações da detecção
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    true_label = class_names[class_id]

                    # Recortar letreiro
                    img = safe_open_image(img_path)
                    if img is None:
                        continue

                    cropped = img.crop((x1, y1, x2, y2))
                    tmp_path = os.path.join(temp_crops_dir, f"crop_{engine_name}.jpg")
                    cropped.save(tmp_path)

                    # Aplicar OCR
                    ocr_text = self.ocr_manager.recognize(tmp_path, engine_name)

                    if not ocr_text:
                        metrics.update(ocr_empty_count=1)
                        # Salvar amostras de OCR vazio
                        if metrics.ocr_empty_count <= max_empty_samples:
                            sample_path = os.path.join(
                                temp_crops_dir,
                                f"{engine_name}_empty_{metrics.ocr_empty_count}.jpg"
                            )
                            cropped.save(sample_path)
                        continue

                    metrics.update(detections_with_ocr=1)

                    # Encontrar melhor match
                    best_match, score = self.text_similarity.find_best_match(
                        ocr_text,
                        class_names
                    )

                    metrics.update(total=1)

                    # Verificar acertos
                    is_exact = (ocr_text == true_label.upper())
                    is_approx = (score >= similarity_threshold)

                    if is_exact:
                        metrics.update(exact_matches=1)
                    elif is_approx:
                        metrics.update(approx_matches=1)

                    # Salvar resultados (acertos e erros)
                    safe_ocr_text = sanitize_filename(ocr_text)
                    save_name = f"{Path(img_name).stem}_real_{true_label}_ocr_{safe_ocr_text}.jpg"

                    # Salvar ACERTOS (exatos ou aproximados)
                    if is_exact or is_approx:
                        cropped.save(os.path.join(correct_ocr_dir, save_name))

                    # Salvar ERROS OCR
                    if not is_exact and not is_approx:
                        cropped.save(os.path.join(wrong_ocr_dir, save_name))

                    # Salvar ERROS de classificação YOLO
                    if best_match.upper() != true_label.upper():
                        cropped.save(os.path.join(wrong_class_dir, save_name))

            if has_detection:
                metrics.update(images_with_detections=1)

        # Log métricas
        metrics.log_summary(logger, f"MÉTRICAS OCR ({engine_name.upper()})")

        if metrics.ocr_empty_count > 0:
            logger.warning(f"\n{metrics.ocr_empty_count} letreiros tiveram OCR vazio!")
            logger.warning(f"   Verifique amostras em: {temp_crops_dir}")

        logger.info(f"\nImagens salvas em:")
        logger.info(f"   - Acertos: {correct_ocr_dir}")
        logger.info(f"   - Erros OCR: {wrong_ocr_dir}")
        logger.info(f"   - Erros Classificação: {wrong_class_dir}")

        return metrics.to_dict()

    def compare_ocr_engines(self, metrics_dict: Dict[str, Dict[str, Any]]):
        """
        Compara resultados de diferentes engines de OCR.

        Args:
            metrics_dict: Dicionário com métricas de cada engine
        """
        logger.info("\n" + "=" * 80)
        logger.info("COMPARAÇÃO DE ENGINES OCR")
        logger.info("=" * 80)

        if not metrics_dict:
            logger.info("Nenhuma métrica disponível para comparação")
            return

        # Tabela comparativa
        logger.info("\n┌────────────────────────────┬────────────┬────────────┬────────────┐")
        logger.info("│ Métrica                    │ Tesseract  │ EasyOCR    │ OpenOCR    │")
        logger.info("├────────────────────────────┼────────────┼────────────┼────────────┤")

        # Função auxiliar para formatar valores
        def get_metric(engine, metric, default=0):
            return metrics_dict.get(engine, {}).get(metric, default)

        # Comparar métricas
        engines = ['tesseract', 'easyocr', 'openocr']

        # Total analisados
        totals = [get_metric(e, 'total') for e in engines]
        logger.info(f"│ Total analisados           │ {totals[0]:10d} │ {totals[1]:10d} │ {totals[2]:10d} │")

        # Acertos exatos
        exacts = [get_metric(e, 'exact_matches') for e in engines]
        logger.info(f"│ Acertos exatos             │ {exacts[0]:10d} │ {exacts[1]:10d} │ {exacts[2]:10d} │")

        # Acurácia exata
        exact_accs = [get_metric(e, 'exact_acc', 0.0) for e in engines]
        logger.info(f"│ Acurácia exata (%)         │ {exact_accs[0]:9.2f}% │ {exact_accs[1]:9.2f}% │ {exact_accs[2]:9.2f}% │")

        # Acurácia total
        total_accs = [get_metric(e, 'approx_acc', 0.0) for e in engines]
        logger.info(f"│ Acurácia total (%)         │ {total_accs[0]:9.2f}% │ {total_accs[1]:9.2f}% │ {total_accs[2]:9.2f}% │")

        logger.info("└────────────────────────────┴────────────┴────────────┴────────────┘")

        # Determinar melhor engine
        best_acc = max(total_accs)
        if best_acc > 0:
            best_idx = total_accs.index(best_acc)
            best_engine = engines[best_idx].upper()
            logger.info(f"\n🏆 Melhor desempenho: {best_engine} ({best_acc:.2f}%)")

    def run(self):
        """Executa o sistema completo"""
        try:
            # 1. Preparar dataset
            self.prepare_dataset()

            # 2. Treinar ou carregar modelo
            if not self.train_or_load_model():
                logger.error("Falha ao carregar/treinar modelo. Encerrando.")
                return

            # 3. Validar modelo
            self.validate_model()

            # 4. Processar OCR com diferentes engines
            all_metrics = {}

            available_engines = self.ocr_manager.get_available_engines()
            logger.info(f"\nEngines OCR disponíveis: {list(available_engines.keys())}")

            for engine_name in available_engines.keys():
                metrics = self.process_ocr(engine_name)
                if metrics:
                    all_metrics[engine_name] = metrics

            # 5. Comparar resultados
            self.compare_ocr_engines(all_metrics)

            # 6. Finalizar
            logger.info("\n" + "=" * 80)
            logger.info("PROCESSAMENTO COMPLETO!")
            logger.info("=" * 80)
            logger.info(f"Resultados salvos em: {self.base_dir}")
            logger.info(f"Log completo: {self.log_path}")

        except Exception as e:
            logger.error(f"Erro durante execução: {e}", exc_info=True)
            raise


def main():
    """Função principal"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sistema de Detecção e Reconhecimento de Letreiros de Ônibus"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Caminho para arquivo de configuração'
    )

    args = parser.parse_args()

    # Criar e executar sistema
    system = BusSignRecognitionSystem(config_path=args.config)
    system.run()


if __name__ == '__main__':
    main()
