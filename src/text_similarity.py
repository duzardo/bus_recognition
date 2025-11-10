# -*- coding: utf-8 -*-
"""
Módulo de Similaridade Textual
Implementa algoritmos de comparação de strings usando Edit Distance (Levenshtein)
"""


class TextSimilarity:
    """Classe para calcular similaridade entre textos usando Levenshtein Distance"""

    @staticmethod
    def levenshtein_distance(str1: str, str2: str) -> int:
        """
        Calcula a distância de Levenshtein (Edit Distance) entre duas strings.

        A distância de Levenshtein é o número mínimo de operações de edição
        (inserção, deleção ou substituição) necessárias para transformar uma
        string em outra.

        Args:
            str1: Primeira string
            str2: Segunda string

        Returns:
            Distância de Levenshtein entre as duas strings
        """
        # Converter para minúsculas para comparação case-insensitive
        str1 = str1.lower()
        str2 = str2.lower()

        len1, len2 = len(str1), len(str2)

        # Criar matriz de distâncias
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Inicializar primeira linha e coluna
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        # Preencher matriz usando programação dinâmica
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if str1[i-1] == str2[j-1] else 1

                dp[i][j] = min(
                    dp[i-1][j] + 1,      # Deleção
                    dp[i][j-1] + 1,      # Inserção
                    dp[i-1][j-1] + cost  # Substituição
                )

        return dp[len1][len2]

    @classmethod
    def normalized_similarity(cls, str1: str, str2: str) -> float:
        """
        Calcula a similaridade normalizada baseada na distância de Levenshtein.

        Retorna um valor entre 0 e 100, onde 100 significa strings idênticas
        e 0 significa strings completamente diferentes.

        Args:
            str1: Primeira string
            str2: Segunda string

        Returns:
            Similaridade normalizada (0-100)
        """
        if not str1 or not str2:
            return 0.0

        distance = cls.levenshtein_distance(str1, str2)
        max_len = max(len(str1), len(str2))

        if max_len == 0:
            return 100.0

        # Normalizar para escala 0-100
        similarity = (1 - distance / max_len) * 100
        return max(0.0, similarity)

    @classmethod
    def partial_similarity(cls, str1: str, str2: str) -> float:
        """
        Calcula a similaridade parcial de Levenshtein.

        Esta função verifica se a string menor está contida na string maior
        e calcula a melhor similaridade possível considerando todas as
        substrings da string maior do tamanho da string menor.

        Args:
            str1: Primeira string
            str2: Segunda string

        Returns:
            Melhor similaridade parcial encontrada (0-100)
        """
        if not str1 or not str2:
            return 0.0

        str1 = str1.lower()
        str2 = str2.lower()

        # Garantir que str1 é a string menor
        if len(str1) > len(str2):
            str1, str2 = str2, str1

        if len(str1) == 0:
            return 0.0

        # Se str1 está contido em str2, retornar 100
        if str1 in str2:
            return 100.0

        best_similarity = 0.0

        # Verificar todas as substrings de str2 com o mesmo tamanho de str1
        for i in range(len(str2) - len(str1) + 1):
            substring = str2[i:i + len(str1)]
            similarity = cls.normalized_similarity(str1, substring)
            best_similarity = max(best_similarity, similarity)

        # Também verificar a similaridade completa
        full_similarity = cls.normalized_similarity(str1, str2)
        best_similarity = max(best_similarity, full_similarity)

        return best_similarity

    @classmethod
    def find_best_match(cls, text: str, candidates: list, threshold: float = 0.0) -> tuple:
        """
        Encontra a melhor correspondência de um texto em uma lista de candidatos.

        Args:
            text: Texto a ser comparado
            candidates: Lista de textos candidatos
            threshold: Similaridade mínima para considerar um match (0-100)

        Returns:
            Tupla (melhor_candidato, pontuação_similaridade)
        """
        if not text or not candidates:
            return "", 0.0

        try:
            best_match = max(
                candidates,
                key=lambda x: cls.partial_similarity(text, x.upper())
            )
            score = cls.partial_similarity(text, best_match.upper())

            if score < threshold:
                return "", score

            return best_match, score
        except Exception:
            return "", 0.0
