# -*- coding: utf-8 -*-
"""
Gerenciador de Dataset
Responsável por preparar, validar e gerenciar o dataset YOLO
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class DatasetManager:
    """Gerencia o dataset YOLO para treinamento e teste"""

    def __init__(self, dataset_dir: str, config: Dict[str, Any]):
        """
        Inicializa o gerenciador de dataset.

        Args:
            dataset_dir: Diretório raiz do dataset
            config: Configurações do sistema (config.yaml)
        """
        self.dataset_dir = dataset_dir
        self.config = config
        self.class_names = config['classes']['names']
        self.num_classes = config['classes']['num_classes']

        # Diretórios
        self.train_dir = os.path.join(dataset_dir, 'train')
        self.val_dir = os.path.join(dataset_dir, 'valid')
        self.test_dir = os.path.join(dataset_dir, 'test')
        self.data_yaml = os.path.join(dataset_dir, 'data.yaml')

    def verify_dataset(self) -> bool:
        """
        Verifica se o dataset existe e está estruturado corretamente.

        Returns:
            True se o dataset existe, False caso contrário
        """
        required_dirs = [
            os.path.join(self.train_dir, 'images'),
            os.path.join(self.train_dir, 'labels'),
            os.path.join(self.val_dir, 'images'),
            os.path.join(self.val_dir, 'labels'),
            os.path.join(self.test_dir, 'images'),
            os.path.join(self.test_dir, 'labels')
        ]

        return all(os.path.exists(d) for d in required_dirs)

    def get_dataset_stats(self) -> Dict[str, int]:
        """
        Retorna estatísticas do dataset.

        Returns:
            Dicionário com contagem de imagens por split
        """
        stats = {}

        for split, split_dir in [
            ('train', self.train_dir),
            ('val', self.val_dir),
            ('test', self.test_dir)
        ]:
            img_dir = os.path.join(split_dir, 'images')
            if os.path.exists(img_dir):
                images = [
                    f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                ]
                stats[split] = len(images)
            else:
                stats[split] = 0

        return stats

    def create_data_yaml(self) -> str:
        """
        Cria o arquivo data.yaml necessário para o YOLO.

        Returns:
            Caminho para o arquivo data.yaml criado
        """
        yaml_content = {
            'train': os.path.join(self.dataset_dir, 'train', 'images'),
            'val': os.path.join(self.dataset_dir, 'valid', 'images'),
            'test': os.path.join(self.dataset_dir, 'test', 'images'),
            'nc': self.num_classes,
            'names': self.class_names,
            'task': 'detect'
        }

        with open(self.data_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Arquivo data.yaml criado em: {self.data_yaml}")
        return self.data_yaml

    def ensure_data_yaml(self) -> str:
        """
        Garante que o arquivo data.yaml existe.

        Returns:
            Caminho para o arquivo data.yaml
        """
        if not os.path.exists(self.data_yaml):
            logger.info("Criando arquivo data.yaml...")
            return self.create_data_yaml()
        else:
            logger.info(f"Arquivo data.yaml já existe: {self.data_yaml}")
            return self.data_yaml

    def get_test_images(self) -> List[str]:
        """
        Retorna lista de imagens no conjunto de teste.

        Returns:
            Lista de nomes de arquivos de imagem
        """
        test_img_dir = os.path.join(self.test_dir, 'images')
        if not os.path.exists(test_img_dir):
            logger.warning(f"Diretório de teste não encontrado: {test_img_dir}")
            return []

        images = [
            f for f in os.listdir(test_img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        return images

    def get_test_image_path(self, image_name: str) -> str:
        """Retorna o caminho completo de uma imagem de teste"""
        return os.path.join(self.test_dir, 'images', image_name)

    def get_test_label_path(self, image_name: str) -> str:
        """Retorna o caminho completo do arquivo de label de uma imagem"""
        label_name = Path(image_name).stem + '.txt'
        return os.path.join(self.test_dir, 'labels', label_name)

    def load_ground_truth_boxes(self, label_path: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
        """
        Carrega as bounding boxes do ground truth de um arquivo de label.

        Args:
            label_path: Caminho para o arquivo .txt de labels
            img_w: Largura da imagem
            img_h: Altura da imagem

        Returns:
            Lista de dicionários com {'cls': int, 'bbox': [x1, y1, x2, y2]}
        """
        boxes = []

        if not os.path.exists(label_path):
            return boxes

        try:
            with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls = int(float(parts[0]))
                    x_ctr = float(parts[1]) * img_w
                    y_ctr = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h

                    x1 = max(0, int(x_ctr - w/2))
                    y1 = max(0, int(y_ctr - h/2))
                    x2 = min(img_w, int(x_ctr + w/2))
                    y2 = min(img_h, int(y_ctr + h/2))

                    boxes.append({'cls': cls, 'bbox': [x1, y1, x2, y2]})

        except Exception as e:
            logger.error(f"Erro ao carregar labels de {label_path}: {e}")

        return boxes

    def log_dataset_info(self):
        """Registra informações sobre o dataset no log"""
        logger.info("=" * 80)
        logger.info("INFORMAÇÕES DO DATASET")
        logger.info("=" * 80)

        if self.verify_dataset():
            logger.info("Dataset encontrado e estruturado corretamente")
            stats = self.get_dataset_stats()
            logger.info(f"   Treino: {stats['train']} imagens")
            logger.info(f"   Validação: {stats['val']} imagens")
            logger.info(f"   Teste: {stats['test']} imagens")
            logger.info(f"   Classes: {self.num_classes}")
        else:
            logger.warning("Dataset não encontrado ou incompleto")
            logger.warning(f"Esperado em: {self.dataset_dir}")
