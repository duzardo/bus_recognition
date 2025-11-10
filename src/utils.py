# -*- coding: utf-8 -*-
"""
Utilitários Gerais
Funções auxiliares para processamento de imagens, arquivos e análise
"""

import os
import shutil
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


def safe_open_image(image_path: str, error_dir: Optional[str] = None) -> Optional[Image.Image]:
    """
    Abre uma imagem de forma segura.
    Se falhar, salva cópia em pasta de erros e retorna None.

    Args:
        image_path: Caminho para a imagem
        error_dir: Diretório para salvar imagens com erro (opcional)

    Returns:
        Imagem PIL ou None se houver erro
    """
    try:
        img = Image.open(image_path)
        # Tenta carregar a imagem para verificar se está corrompida
        img.load()
        return img
    except Exception as e:
        logger.error(f"Erro ao abrir imagem {image_path}: {e}")

        # Copiar imagem com erro para pasta de erros
        if error_dir:
            try:
                os.makedirs(error_dir, exist_ok=True)
                error_filename = f"erro_{os.path.basename(image_path)}"
                error_path = os.path.join(error_dir, error_filename)
                shutil.copy2(image_path, error_path)
                logger.info(f"   Imagem com erro copiada para: {error_path}")
            except Exception as copy_error:
                logger.error(f"   Erro ao copiar imagem: {copy_error}")

        return None


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Remove ou substitui caracteres inválidos para nomes de arquivo no Windows.

    Caracteres inválidos: < > : " / \\ | ? *
    Também remove espaços extras e limita o tamanho.

    Args:
        text: Texto a ser sanitizado
        max_length: Tamanho máximo do nome

    Returns:
        Texto seguro para usar em nome de arquivo
    """
    if not text:
        return "empty"

    # Substituir caracteres inválidos por underscore
    invalid_chars = '<>:"/\\|?*{}'
    for char in invalid_chars:
        text = text.replace(char, '_')

    # Remover espaços extras e limitar tamanho
    text = text.strip()
    text = '_'.join(text.split())  # Substituir espaços por underscore

    # Limitar tamanho do nome
    if len(text) > max_length:
        text = text[:max_length]

    return text if text else "empty"


def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """
    Calcula Intersection over Union (IoU) entre duas bounding boxes.

    Args:
        boxA: Bounding box no formato [x1, y1, x2, y2]
        boxB: Bounding box no formato [x1, y1, x2, y2]

    Returns:
        Valor de IoU (0.0 a 1.0)
    """
    # Coordenadas da interseção
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Área da interseção
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    # Áreas das boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def match_predictions_to_ground_truth(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int, float]], set, set]:
    """
    Faz o matching entre predições e ground truths usando IoU.

    Args:
        predictions: Lista de predições com formato {'cls': int, 'bbox': [x1,y1,x2,y2]}
        ground_truths: Lista de ground truths com mesmo formato
        iou_threshold: Threshold mínimo de IoU para considerar um match

    Returns:
        Tupla contendo:
        - Lista de matches (pred_idx, gt_idx, iou_score)
        - Set de índices de predições usadas
        - Set de índices de ground truths usadas
    """
    if len(predictions) == 0 or len(ground_truths) == 0:
        return [], set(), set()

    # Criar matriz de IoU
    iou_matrix = np.zeros((len(predictions), len(ground_truths)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            iou_matrix[i, j] = calculate_iou(pred['bbox'], gt['bbox'])

    # Greedy matching
    matches = []
    used_preds = set()
    used_gts = set()

    while True:
        if iou_matrix.size == 0:
            break

        # Encontrar maior IoU
        idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = iou_matrix[idx]

        if max_iou <= 0:
            break

        pi, gj = idx

        if max_iou >= iou_threshold:
            matches.append((pi, gj, float(max_iou)))
            used_preds.add(pi)
            used_gts.add(gj)
            # Marcar linha e coluna como usadas
            iou_matrix[pi, :] = -1
            iou_matrix[:, gj] = -1
        else:
            break

    return matches, used_preds, used_gts


def setup_logging(log_dir: str, log_name: str = "execucao") -> str:
    """
    Configura o sistema de logging (console + arquivo).

    Args:
        log_dir: Diretório para salvar os logs
        log_name: Nome base do arquivo de log

    Returns:
        Caminho para o arquivo de log criado
    """
    from datetime import datetime

    os.makedirs(log_dir, exist_ok=True)

    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'{log_name}_{timestamp}.log'
    log_path = os.path.join(log_dir, log_filename)

    # Configurar handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output no terminal
            logging.FileHandler(log_path, encoding='utf-8')  # Salvar em arquivo
        ]
    )

    return log_path


def create_directory_structure(base_dir: str, subdirs: List[str]):
    """
    Cria estrutura de diretórios.

    Args:
        base_dir: Diretório base
        subdirs: Lista de subdiretórios a criar
    """
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Diretório criado/verificado: {path}")


def get_tesseract_paths() -> List[str]:
    """
    Retorna lista de caminhos comuns onde o Tesseract pode estar instalado no Windows.

    Returns:
        Lista de caminhos possíveis
    """
    username = os.getenv('USERNAME', '')
    return [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        f'C:\\Users\\{username}\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
    ]


class MetricsCollector:
    """Coletor de métricas para análise de OCR"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reseta todas as métricas"""
        self.total = 0
        self.exact_matches = 0
        self.approx_matches = 0
        self.images_processed = 0
        self.images_with_detections = 0
        self.total_detections = 0
        self.detections_with_ocr = 0
        self.ocr_empty_count = 0

    def update(self, **kwargs):
        """Atualiza métricas específicas"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, getattr(self, key) + value)

    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Calcula e retorna métricas de acurácia"""
        if self.total == 0:
            return {
                'exact_acc': 0.0,
                'approx_acc': 0.0,
                'success_rate': 0.0
            }

        exact_acc = (self.exact_matches / self.total) * 100
        approx_acc = ((self.exact_matches + self.approx_matches) / self.total) * 100

        success_rate = 0.0
        if self.total_detections > 0:
            success_rate = (self.detections_with_ocr / self.total_detections) * 100

        return {
            'exact_acc': exact_acc,
            'approx_acc': approx_acc,
            'success_rate': success_rate
        }

    def to_dict(self) -> Dict[str, Any]:
        """Converte métricas para dicionário"""
        metrics = self.get_accuracy_metrics()
        return {
            'total': self.total,
            'exact_matches': self.exact_matches,
            'approx_matches': self.approx_matches,
            'exact_acc': metrics['exact_acc'],
            'approx_acc': metrics['approx_acc'],
            'images_processed': self.images_processed,
            'images_with_detections': self.images_with_detections,
            'total_detections': self.total_detections,
            'detections_with_ocr': self.detections_with_ocr,
            'ocr_empty_count': self.ocr_empty_count
        }

    def log_summary(self, logger_instance: logging.Logger, title: str = "MÉTRICAS"):
        """Registra sumário das métricas no log"""
        logger_instance.info(f"\n{'=' * 80}")
        logger_instance.info(f"{title}")
        logger_instance.info(f"{'=' * 80}")

        if self.total > 0:
            metrics = self.get_accuracy_metrics()
            logger_instance.info(f"   Total de letreiros analisados: {self.total}")
            logger_instance.info(f"   Acertos exatos: {self.exact_matches} ({metrics['exact_acc']:.2f}%)")
            logger_instance.info(f"   Acertos aproximados: {self.approx_matches}")
            logger_instance.info(f"   Acurácia total: {metrics['approx_acc']:.2f}%")
            logger_instance.info(f"   Taxa de sucesso OCR: {metrics['success_rate']:.2f}%")
        else:
            logger_instance.info("   Nenhuma métrica disponível")
