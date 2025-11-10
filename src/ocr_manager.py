# -*- coding: utf-8 -*-
"""
Gerenciador de OCR
Centraliza a lógica de OCR para Tesseract, EasyOCR e OpenOCR
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class OCREngine(ABC):
    """Classe abstrata base para engines de OCR"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def recognize(self, image_path: str) -> str:
        """
        Extrai texto de uma imagem.

        Args:
            image_path: Caminho para a imagem

        Returns:
            Texto extraído (em maiúsculas)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Verifica se o engine está disponível/instalado"""
        pass


class TesseractOCR(OCREngine):
    """Engine de OCR usando Tesseract"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tesseract = None
        self._initialize()

    def _initialize(self):
        """Inicializa e configura o Tesseract"""
        try:
            import pytesseract

            # Tentar encontrar Tesseract no Windows
            tesseract_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(
                    os.getenv('USERNAME', '')
                )
            ]

            for tess_path in tesseract_paths:
                if os.path.exists(tess_path):
                    pytesseract.pytesseract.tesseract_cmd = tess_path
                    logger.info(f"Tesseract encontrado em: {tess_path}")
                    self.tesseract = pytesseract
                    return

            # Se não encontrou, ainda tenta usar (pode estar no PATH)
            self.tesseract = pytesseract
            logger.warning("Tesseract não encontrado nos caminhos padrão. Usando PATH do sistema.")

        except ImportError:
            logger.warning("Biblioteca pytesseract não instalada. Execute: pip install pytesseract")
            self.tesseract = None

    def is_available(self) -> bool:
        return self.tesseract is not None

    def recognize(self, image_path: str) -> str:
        """Extrai texto usando Tesseract"""
        if not self.is_available():
            return ""

        try:
            img = Image.open(image_path)
            psm = self.config.get('psm', 7)
            lang = self.config.get('lang', 'eng')

            text = self.tesseract.image_to_string(
                img,
                lang=lang,
                config=f'--psm {psm}'
            )
            return text.strip().upper()
        except Exception as e:
            logger.debug(f"Erro no Tesseract OCR para {image_path}: {e}")
            return ""


class EasyOCREngine(OCREngine):
    """Engine de OCR usando EasyOCR"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.reader = None
        self._initialize()

    def _initialize(self):
        """Inicializa o EasyOCR"""
        try:
            import easyocr

            languages = self.config.get('languages', ['pt', 'en'])
            gpu = self.config.get('gpu', True)

            logger.info(f"Inicializando EasyOCR (idiomas: {languages}, GPU: {gpu})...")
            self.reader = easyocr.Reader(languages, gpu=gpu)
            logger.info("EasyOCR inicializado com sucesso!")

        except ImportError:
            logger.warning("Biblioteca easyocr não instalada. Execute: pip install easyocr")
            self.reader = None
        except Exception as e:
            logger.error(f"Erro ao inicializar EasyOCR: {e}")
            self.reader = None

    def is_available(self) -> bool:
        return self.reader is not None

    def recognize(self, image_path: str) -> str:
        """Extrai texto usando EasyOCR"""
        if not self.is_available():
            return ""

        try:
            import cv2

            # Ler imagem e converter para grayscale
            img = cv2.imread(image_path)
            if img is None:
                logger.debug(f"Não foi possível ler a imagem: {image_path}")
                return ""

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Aplicar OCR
            result = self.reader.readtext(gray, detail=0)
            text = " ".join(result).strip().upper()
            return text

        except Exception as e:
            logger.debug(f"Erro no EasyOCR para {image_path}: {e}")
            return ""


class OpenOCREngine(OCREngine):
    """Engine de OCR usando OpenOCR (SVTRv2)"""

    def __init__(self, config: Dict[str, Any], openocr_dir: str):
        super().__init__(config)
        self.openocr_dir = openocr_dir
        self.recognizer = None
        self._initialize()

    def _initialize(self):
        """Inicializa o OpenOCR"""
        try:
            if not os.path.exists(self.openocr_dir):
                logger.warning(f"Diretório OpenOCR não encontrado: {self.openocr_dir}")
                return

            # Adicionar ao path
            import sys
            if self.openocr_dir not in sys.path:
                sys.path.insert(0, self.openocr_dir)

            from tools.infer_rec import OpenRecognizer
            from tools.engine.config import Config

            # Configurar caminho do config
            config_path = os.path.join(self.openocr_dir, self.config.get('config_path'))
            if not os.path.exists(config_path):
                fallback = os.path.join(self.openocr_dir, self.config.get('fallback_config'))
                if os.path.exists(fallback):
                    config_path = fallback
                else:
                    logger.warning(f"Arquivo de configuração OpenOCR não encontrado: {config_path}")
                    return

            # Carregar configuração
            cfg_rec = Config(config_path).cfg
            cfg_rec['Global']['device'] = 'gpu' if self._has_gpu() else 'cpu'

            # Inicializar recognizer
            backend = self.config.get('backend', 'torch')
            self.recognizer = OpenRecognizer(config=cfg_rec, backend=backend)

            logger.info("OpenOCR inicializado com sucesso!")
            logger.info(f"   Device: {cfg_rec['Global']['device']}")
            logger.info(f"   Config: {os.path.basename(config_path)}")

        except ImportError as e:
            logger.warning(f"Biblioteca OpenOCR não disponível: {e}")
            self.recognizer = None
        except Exception as e:
            logger.error(f"Erro ao inicializar OpenOCR: {e}")
            self.recognizer = None

    def _has_gpu(self) -> bool:
        """Verifica se GPU está disponível"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return os.system('nvidia-smi') == 0

    def is_available(self) -> bool:
        return self.recognizer is not None

    def recognize(self, image_path: str) -> str:
        """Extrai texto usando OpenOCR"""
        if not self.is_available():
            return ""

        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                logger.debug(f"Não foi possível ler a imagem: {image_path}")
                return ""

            # Aplicar OCR
            result = self.recognizer(img_numpy=img, batch_num=1)

            if result and len(result) > 0:
                text = result[0].get('text', '').strip().upper()
                return text
            else:
                return ""

        except Exception as e:
            logger.debug(f"Erro no OpenOCR para {image_path}: {e}")
            return ""


class OCRManager:
    """Gerenciador centralizado de múltiplos engines de OCR"""

    def __init__(self, config: Dict[str, Any], openocr_dir: Optional[str] = None):
        """
        Inicializa o gerenciador de OCR.

        Args:
            config: Configurações de OCR do config.yaml
            openocr_dir: Diretório do OpenOCR (opcional)
        """
        self.config = config
        self.engines = {}

        # Inicializar engines conforme configuração
        if config.get('tesseract', {}).get('enabled', True):
            self.engines['tesseract'] = TesseractOCR(config.get('tesseract', {}))

        if config.get('easyocr', {}).get('enabled', True):
            self.engines['easyocr'] = EasyOCREngine(config.get('easyocr', {}))

        if config.get('openocr', {}).get('enabled', True) and openocr_dir:
            self.engines['openocr'] = OpenOCREngine(config.get('openocr', {}), openocr_dir)

    def get_engine(self, engine_name: str) -> Optional[OCREngine]:
        """Obtém um engine específico"""
        return self.engines.get(engine_name)

    def get_available_engines(self) -> Dict[str, OCREngine]:
        """Retorna apenas os engines disponíveis"""
        return {
            name: engine
            for name, engine in self.engines.items()
            if engine.is_available()
        }

    def recognize(self, image_path: str, engine_name: str) -> str:
        """
        Reconhece texto em uma imagem usando um engine específico.

        Args:
            image_path: Caminho para a imagem
            engine_name: Nome do engine ('tesseract', 'easyocr', 'openocr')

        Returns:
            Texto reconhecido
        """
        engine = self.get_engine(engine_name)
        if engine and engine.is_available():
            return engine.recognize(image_path)
        return ""
