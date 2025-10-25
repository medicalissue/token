import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union, Literal, List
from dataclasses import dataclass, field
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from collections import deque
import colorsys


@dataclass
class SimilarityResult:
    """토큰 유사도 분석 결과를 저장하는 데이터 클래스"""
    # 인접 토큰 유사도 (N, M) 형태, 각 방향별
    horizontal_right: np.ndarray  # 오른쪽 토큰과의 유사도
    horizontal_left: np.ndarray   # 왼쪽 토큰과의 유사도
    vertical_down: np.ndarray     # 아래쪽 토큰과의 유사도
    vertical_up: np.ndarray       # 위쪽 토큰과의 유사도

    # CLS 토큰 유사도
    cls_similarity: np.ndarray    # (N, M) 형태

    # 메타데이터
    grid_size: int
    num_patches: int

    # 패치 토큰 (클러스터 평균 계산용, optional)
    patch_tokens: Optional[torch.Tensor] = None  # (num_patches, hidden_dim)


@dataclass
class ClusterResult:
    """클러스터링 결과를 저장하는 데이터 클래스"""
    cluster_map: np.ndarray          # (grid_size, grid_size) 각 토큰의 클러스터 ID
    num_clusters: int                # 총 클러스터 개수
    cluster_sizes: Dict[int, int]    # 각 클러스터의 크기
    threshold: float                 # 사용된 임계값
    similarity_result: SimilarityResult  # 원본 유사도 결과


class TokenSimilarityAnalyzer:
    """CLIP 비전 토큰의 유사도를 분석하는 클래스"""

    def __init__(self,
                 model_type: Literal["clip-vit-large-patch14", "clip-vit-large-patch14-336"] = "clip-vit-large-patch14",
                 device: Optional[Union[str, int]] = None,
                 similarity_metric: Literal["cosine", "dot_product"] = "cosine",
                 remove_positional_embedding: bool = False):
        """
        Args:
            model_type: 사용할 CLIP 모델 타입
            device: 사용할 디바이스
            similarity_metric: 유사도 계산 방식 (cosine 또는 dot_product)
            remove_positional_embedding: 포지셔널 임베딩 제거 여부
        """
        self.model_type = model_type
        self.similarity_metric = similarity_metric
        self.remove_positional_embedding = remove_positional_embedding

        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device(device)

        # 모델 설정
        MODEL_CONFIGS = {
            "clip-vit-large-patch14": {
                "model_name": "openai/clip-vit-large-patch14",
                "grid_size": 16
            },
            "clip-vit-large-patch14-336": {
                "model_name": "openai/clip-vit-large-patch14-336",
                "grid_size": 24
            }
        }

        self.config = MODEL_CONFIGS[model_type]
        self.grid_size = self.config["grid_size"]

        print(f"Loading {model_type}...")
        print(f"Using device: {self.device}")
        print(f"Similarity metric: {similarity_metric}")
        print(f"Remove positional embedding: {remove_positional_embedding}")

        self.model = CLIPModel.from_pretrained(self.config["model_name"])
        self.processor = CLIPProcessor.from_pretrained(self.config["model_name"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 포지셔널 임베딩 저장 (나중에 복원용)
        if self.remove_positional_embedding:
            self.original_position_embedding = self.model.vision_model.embeddings.position_embedding.weight.data.clone()
            print("⚠️  Positional embedding will be removed during inference")

        print(f"Model loaded! Grid size: {self.grid_size}x{self.grid_size}")

    def get_preprocessed_image(self, image_path: str) -> Image.Image:
        """
        CLIP 전처리된 이미지를 반환합니다 (실제로 모델에 들어가는 이미지).

        Args:
            image_path: 이미지 파일 경로

        Returns:
            preprocessed_image: 전처리된 PIL Image (224x224 또는 336x336)
        """
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        # pixel_values에서 이미지 복원
        pixel_values = inputs['pixel_values'][0]  # (3, H, W)
        pixel_values = pixel_values.permute(1, 2, 0).numpy()

        # 정규화 해제
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        pixel_values = pixel_values * std + mean
        pixel_values = np.clip(pixel_values * 255, 0, 255).astype(np.uint8)

        preprocessed_image = Image.fromarray(pixel_values)
        return preprocessed_image

    def extract_tokens(self, image_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        이미지에서 CLS 토큰과 패치 토큰을 추출합니다.

        Args:
            image_path: 이미지 파일 경로

        Returns:
            cls_token: CLS 토큰 (hidden_dim,)
            patch_tokens: 패치 토큰들 (num_patches, hidden_dim)
        """
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 포지셔널 임베딩 제거 (일시적으로)
        if self.remove_positional_embedding:
            # 포지셔널 임베딩을 0으로 설정
            self.model.vision_model.embeddings.position_embedding.weight.data.zero_()

        # 비전 인코더를 통과
        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs['pixel_values'],
                output_hidden_states=True
            )

        # 포지셔널 임베딩 복원
        if self.remove_positional_embedding:
            self.model.vision_model.embeddings.position_embedding.weight.data.copy_(
                self.original_position_embedding
            )

        # 히든 스테이트: (batch_size, num_patches + 1, hidden_dim)
        hidden_states = vision_outputs.last_hidden_state[0]  # (num_patches + 1, hidden_dim)

        cls_token = hidden_states[0]  # (hidden_dim,)
        patch_tokens = hidden_states[1:]  # (num_patches, hidden_dim)

        return cls_token, patch_tokens

    def compute_similarity(self, tokens1: torch.Tensor, tokens2: torch.Tensor) -> torch.Tensor:
        """
        두 토큰 집합 간의 유사도를 계산합니다.

        Args:
            tokens1: (N, D) 형태의 토큰들
            tokens2: (N, D) 형태의 토큰들

        Returns:
            similarities: (N,) 형태의 유사도 값들
        """
        if self.similarity_metric == "cosine":
            # 코사인 유사도
            similarities = F.cosine_similarity(tokens1, tokens2, dim=-1)
        elif self.similarity_metric == "dot_product":
            # 내적
            similarities = (tokens1 * tokens2).sum(dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        return similarities

    def compute_neighbor_similarities(self, patch_tokens: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        격자 형태로 배열된 패치 토큰들의 상하좌우 인접 유사도를 계산합니다.

        각 방향별로 유사도를 계산하며, 가장자리 토큰은 해당 방향의 이웃이 없으므로 제외됩니다:
        - horizontal_right: 맨 오른쪽 열(grid[:, -1]) 제외 -> (grid_size, grid_size-1)
        - horizontal_left: 맨 왼쪽 열(grid[:, 0]) 제외 -> (grid_size, grid_size-1)
        - vertical_down: 맨 아래 행(grid[-1, :]) 제외 -> (grid_size-1, grid_size)
        - vertical_up: 맨 위 행(grid[0, :]) 제외 -> (grid_size-1, grid_size)

        Args:
            patch_tokens: (num_patches, hidden_dim) 형태의 패치 토큰들

        Returns:
            similarities: 각 방향별 유사도 딕셔너리
                - 'horizontal_right': 각 토큰(맨 오른쪽 제외)과 오른쪽 이웃의 유사도
                - 'horizontal_left': 각 토큰(맨 왼쪽 제외)과 왼쪽 이웃의 유사도
                - 'vertical_down': 각 토큰(맨 아래 제외)과 아래 이웃의 유사도
                - 'vertical_up': 각 토큰(맨 위 제외)과 위 이웃의 유사도
        """
        # 그리드 형태로 재배열: (grid_size, grid_size, hidden_dim)
        tokens_grid = patch_tokens.reshape(self.grid_size, self.grid_size, -1)

        # 각 방향별 유사도 계산
        similarities = {}

        # 오른쪽 (→): 각 토큰과 오른쪽 이웃과의 유사도
        # tokens_grid[:, :-1]은 맨 오른쪽 열 제외
        # tokens_grid[:, 1:]은 맨 왼쪽 열 제외
        # 결과: (grid_size, grid_size-1) - 각 행에서 인접 쌍의 유사도
        left_tokens = tokens_grid[:, :-1, :].reshape(-1, patch_tokens.shape[-1])
        right_tokens = tokens_grid[:, 1:, :].reshape(-1, patch_tokens.shape[-1])
        horizontal_right = self.compute_similarity(left_tokens, right_tokens)
        similarities['horizontal_right'] = horizontal_right.cpu().numpy().reshape(self.grid_size, self.grid_size - 1)

        # 왼쪽 (←): 동일한 유사도이지만 해석이 다름
        # horizontal_right[i, j]는 (i,j)와 (i,j+1)의 유사도
        # horizontal_left 관점에서는 (i,j+1)와 (i,j)의 유사도
        # 코사인 유사도는 대칭이므로 동일
        similarities['horizontal_left'] = similarities['horizontal_right']

        # 아래쪽 (↓): 각 토큰과 아래쪽 이웃과의 유사도
        # tokens_grid[:-1, :]는 맨 아래 행 제외
        # tokens_grid[1:, :]는 맨 위 행 제외
        # 결과: (grid_size-1, grid_size) - 각 열에서 인접 쌍의 유사도
        top_tokens = tokens_grid[:-1, :, :].reshape(-1, patch_tokens.shape[-1])
        bottom_tokens = tokens_grid[1:, :, :].reshape(-1, patch_tokens.shape[-1])
        vertical_down = self.compute_similarity(top_tokens, bottom_tokens)
        similarities['vertical_down'] = vertical_down.cpu().numpy().reshape(self.grid_size - 1, self.grid_size)

        # 위쪽 (↑): 동일한 유사도이지만 해석이 다름
        # vertical_down[i, j]는 (i,j)와 (i+1,j)의 유사도
        # vertical_up 관점에서는 (i+1,j)와 (i,j)의 유사도
        # 코사인 유사도는 대칭이므로 동일
        similarities['vertical_up'] = similarities['vertical_down']

        return similarities

    def compute_cls_similarity(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> np.ndarray:
        """
        모든 패치 토큰과 CLS 토큰 간의 유사도를 계산합니다.

        Args:
            cls_token: (hidden_dim,) 형태의 CLS 토큰
            patch_tokens: (num_patches, hidden_dim) 형태의 패치 토큰들

        Returns:
            cls_similarities: (grid_size, grid_size) 형태의 유사도 맵
        """
        # CLS 토큰을 패치 수만큼 복제
        cls_expanded = cls_token.unsqueeze(0).expand(patch_tokens.shape[0], -1)

        # 유사도 계산
        similarities = self.compute_similarity(patch_tokens, cls_expanded)

        # 그리드 형태로 재배열
        similarities_grid = similarities.cpu().numpy().reshape(self.grid_size, self.grid_size)

        return similarities_grid

    def analyze(self, image_path: str) -> SimilarityResult:
        """
        이미지의 토큰 유사도를 전체 분석합니다.

        Args:
            image_path: 이미지 파일 경로

        Returns:
            SimilarityResult: 분석 결과
        """
        print(f"\nAnalyzing image: {image_path}")

        # 토큰 추출
        cls_token, patch_tokens = self.extract_tokens(image_path)
        print(f"Extracted {patch_tokens.shape[0]} patch tokens")

        # 인접 유사도 계산
        print("Computing neighbor similarities...")
        neighbor_sims = self.compute_neighbor_similarities(patch_tokens)

        # CLS 유사도 계산
        print("Computing CLS token similarities...")
        cls_sim = self.compute_cls_similarity(cls_token, patch_tokens)

        result = SimilarityResult(
            horizontal_right=neighbor_sims['horizontal_right'],
            horizontal_left=neighbor_sims['horizontal_left'],
            vertical_down=neighbor_sims['vertical_down'],
            vertical_up=neighbor_sims['vertical_up'],
            cls_similarity=cls_sim,
            grid_size=self.grid_size,
            num_patches=patch_tokens.shape[0],
            patch_tokens=patch_tokens  # 클러스터 평균 계산용
        )

        print("✓ Analysis complete!")
        return result


class TokenClusterer:
    """BFS 기반 토큰 클러스터링 클래스"""

    def __init__(self,
                 threshold_mode: str = "mean",
                 threshold_value: Optional[float] = None,
                 use_cluster_mean: bool = False):
        """
        Args:
            threshold_mode: 임계값 결정 방식
                - "mean": 평균값
                - "mean+std": 평균 + 1 표준편차
                - "mean-std": 평균 - 1 표준편차
                - "mean+2std": 평균 + 2 표준편차
                - "mean-2std": 평균 - 2 표준편차
                - "median": 중앙값
                - "median+mad": 중앙값 + MAD (Median Absolute Deviation)
                - "median-mad": 중앙값 - MAD
                - "q1": 1사분위수 (25th percentile)
                - "q3": 3사분위수 (75th percentile)
                - "iqr_lower": Q1 - 1.5 * IQR
                - "iqr_upper": Q3 + 1.5 * IQR
                - "percentile": 백분위수 (threshold_value 필수)
                - "fixed": 고정값 (threshold_value 필수)
            threshold_value: "percentile" 또는 "fixed" 모드일 때 필요
            use_cluster_mean: True면 클러스터 평균과 비교, False면 인접 유사도 사용
        """
        self.threshold_mode = threshold_mode
        self.threshold_value = threshold_value
        self.use_cluster_mean = use_cluster_mean

    def compute_threshold(self, similarity_result: SimilarityResult) -> float:
        """
        유사도 통계를 기반으로 임계값을 계산합니다.

        모든 토큰 간의 유사도를 계산하여 전체 분포를 기반으로 임계값을 결정합니다.
        (클러스터링 시에는 상하좌우 인접만 병합)

        Args:
            similarity_result: 유사도 분석 결과

        Returns:
            threshold: 계산된 임계값
        """
        # 모든 토큰 간의 유사도를 계산
        if similarity_result.patch_tokens is None:
            # patch_tokens가 없으면 인접 유사도만 사용 (fallback)
            print("⚠️  Warning: patch_tokens not available, using neighbor similarities for threshold")
            all_sims = np.concatenate([
                similarity_result.horizontal_right.flatten(),
                similarity_result.vertical_down.flatten()
            ])
        else:
            # 모든 토큰 쌍의 유사도 계산
            patch_tokens = similarity_result.patch_tokens  # (num_patches, hidden_dim)
            num_patches = patch_tokens.shape[0]

            # 코사인 유사도 계산: (num_patches, num_patches)
            patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=1)
            similarity_matrix = torch.mm(patch_tokens_norm, patch_tokens_norm.t())

            # 대각선 제외 (자기 자신과의 유사도는 항상 1.0이므로 제외)
            mask = ~torch.eye(num_patches, dtype=torch.bool, device=similarity_matrix.device)
            all_sims = similarity_matrix[mask].cpu().numpy()

        mean_val = all_sims.mean()
        std_val = all_sims.std()
        median_val = np.median(all_sims)

        if self.threshold_mode == "mean":
            threshold = mean_val
        elif self.threshold_mode == "mean+std":
            threshold = mean_val + std_val
        elif self.threshold_mode == "mean-std":
            threshold = mean_val - std_val
        elif self.threshold_mode == "mean+2std":
            threshold = mean_val + 2 * std_val
        elif self.threshold_mode == "mean-2std":
            threshold = mean_val - 2 * std_val
        elif self.threshold_mode == "median":
            threshold = median_val
        elif self.threshold_mode == "median+mad":
            mad = np.median(np.abs(all_sims - median_val))
            threshold = median_val + mad
        elif self.threshold_mode == "median-mad":
            mad = np.median(np.abs(all_sims - median_val))
            threshold = median_val - mad
        elif self.threshold_mode == "q1":
            threshold = np.percentile(all_sims, 25)
        elif self.threshold_mode == "q3":
            threshold = np.percentile(all_sims, 75)
        elif self.threshold_mode == "iqr_lower":
            q1 = np.percentile(all_sims, 25)
            q3 = np.percentile(all_sims, 75)
            iqr = q3 - q1
            threshold = q1 - 1.5 * iqr
        elif self.threshold_mode == "iqr_upper":
            q1 = np.percentile(all_sims, 25)
            q3 = np.percentile(all_sims, 75)
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
        elif self.threshold_mode == "percentile":
            if self.threshold_value is None:
                raise ValueError("threshold_value must be specified for percentile mode")
            threshold = np.percentile(all_sims, self.threshold_value)
        elif self.threshold_mode == "fixed":
            if self.threshold_value is None:
                raise ValueError("threshold_value must be specified for fixed mode")
            threshold = self.threshold_value
        else:
            raise ValueError(f"Unknown threshold_mode: {self.threshold_mode}")

        return threshold

    def cluster_bfs(self, similarity_result: SimilarityResult, threshold: Optional[float] = None) -> ClusterResult:
        """
        BFS를 사용하여 인접한 토큰들을 클러스터링합니다.

        Args:
            similarity_result: 유사도 분석 결과
            threshold: 클러스터링 임계값 (None이면 자동 계산)

        Returns:
            ClusterResult: 클러스터링 결과
        """
        # 임계값 결정
        if threshold is None:
            threshold = self.compute_threshold(similarity_result)

        # 클러스터링 모드에 따라 분기
        if self.use_cluster_mean:
            print(f"\nClustering mode: Cluster Mean Comparison")
            print(f"Threshold: {threshold:.4f}")
            return self._cluster_bfs_with_mean(similarity_result, threshold)
        else:
            print(f"\nClustering mode: Pairwise Similarity")
            print(f"Threshold: {threshold:.4f}")
            return self._cluster_bfs_pairwise(similarity_result, threshold)

    def _cluster_bfs_pairwise(self, similarity_result: SimilarityResult, threshold: float) -> ClusterResult:
        """
        미리 계산된 인접 유사도를 사용한 BFS 클러스터링 (기존 방식)

        Args:
            similarity_result: 유사도 분석 결과
            threshold: 클러스터링 임계값

        Returns:
            ClusterResult: 클러스터링 결과
        """

        grid_size = similarity_result.grid_size
        cluster_map = np.full((grid_size, grid_size), -1, dtype=int)  # -1은 미방문
        current_cluster_id = 0

        # BFS로 클러스터링
        for start_row in range(grid_size):
            for start_col in range(grid_size):
                if cluster_map[start_row, start_col] != -1:
                    continue  # 이미 방문함

                # 새 클러스터 시작
                queue = deque([(start_row, start_col)])
                cluster_map[start_row, start_col] = current_cluster_id

                while queue:
                    row, col = queue.popleft()

                    # 4방향 이웃 확인 (상하좌우)
                    neighbors = []

                    # 오른쪽
                    if col < grid_size - 1:
                        sim = similarity_result.horizontal_right[row, col]
                        neighbors.append((row, col + 1, sim))

                    # 왼쪽
                    if col > 0:
                        sim = similarity_result.horizontal_right[row, col - 1]
                        neighbors.append((row, col - 1, sim))

                    # 아래
                    if row < grid_size - 1:
                        sim = similarity_result.vertical_down[row, col]
                        neighbors.append((row + 1, col, sim))

                    # 위
                    if row > 0:
                        sim = similarity_result.vertical_down[row - 1, col]
                        neighbors.append((row - 1, col, sim))

                    # 임계값 이상인 이웃만 같은 클러스터에 추가
                    for n_row, n_col, sim in neighbors:
                        if cluster_map[n_row, n_col] == -1 and sim >= threshold:
                            cluster_map[n_row, n_col] = current_cluster_id
                            queue.append((n_row, n_col))

                current_cluster_id += 1

        # 클러스터 크기 계산
        cluster_sizes = {}
        for cluster_id in range(current_cluster_id):
            cluster_sizes[cluster_id] = np.sum(cluster_map == cluster_id)

        result = ClusterResult(
            cluster_map=cluster_map,
            num_clusters=current_cluster_id,
            cluster_sizes=cluster_sizes,
            threshold=threshold,
            similarity_result=similarity_result
        )

        print(f"✓ Found {current_cluster_id} clusters")
        print(f"  Cluster sizes: min={min(cluster_sizes.values())}, "
              f"max={max(cluster_sizes.values())}, "
              f"mean={np.mean(list(cluster_sizes.values())):.1f}")

        return result

    def _cluster_bfs_with_mean(self, similarity_result: SimilarityResult, threshold: float) -> ClusterResult:
        """
        클러스터 평균 토큰과의 유사도를 사용한 BFS 클러스터링 (새로운 방식)

        각 이웃 토큰이 현재 클러스터에 속한 토큰들의 평균과 유사도를 계산하여
        임계값 이상이면 클러스터에 추가합니다.

        Args:
            similarity_result: 유사도 분석 결과 (여기서 patch_tokens를 가져옴)
            threshold: 클러스터링 임계값

        Returns:
            ClusterResult: 클러스터링 결과
        """
        grid_size = similarity_result.grid_size
        cluster_map = np.full((grid_size, grid_size), -1, dtype=int)  # -1은 미방문
        current_cluster_id = 0

        # 패치 토큰들을 그리드 형태로 재배열 (나중에 평균 계산용)
        # SimilarityResult에는 토큰이 없으므로, analyze 메서드에서 전달받아야 함
        # 임시로 similarity_result에 patch_tokens를 추가해야 함
        if not hasattr(similarity_result, 'patch_tokens'):
            raise ValueError("similarity_result must have patch_tokens for cluster mean mode")

        patch_tokens = similarity_result.patch_tokens  # (num_patches, hidden_dim)
        tokens_grid = patch_tokens.reshape(grid_size, grid_size, -1)  # (grid_size, grid_size, hidden_dim)

        # BFS로 클러스터링
        for start_row in range(grid_size):
            for start_col in range(grid_size):
                if cluster_map[start_row, start_col] != -1:
                    continue  # 이미 방문함

                # 새 클러스터 시작
                queue = deque([(start_row, start_col)])
                cluster_map[start_row, start_col] = current_cluster_id

                # 현재 클러스터에 속한 토큰들의 리스트
                cluster_tokens = [tokens_grid[start_row, start_col]]

                while queue:
                    row, col = queue.popleft()

                    # 현재 클러스터의 평균 토큰 계산
                    cluster_mean = torch.stack(cluster_tokens).mean(dim=0)  # (hidden_dim,)

                    # 4방향 이웃 확인 (상하좌우)
                    neighbors = []

                    # 오른쪽
                    if col < grid_size - 1:
                        neighbors.append((row, col + 1))

                    # 왼쪽
                    if col > 0:
                        neighbors.append((row, col - 1))

                    # 아래
                    if row < grid_size - 1:
                        neighbors.append((row + 1, col))

                    # 위
                    if row > 0:
                        neighbors.append((row - 1, col))

                    # 각 이웃과 클러스터 평균의 유사도 계산
                    for n_row, n_col in neighbors:
                        if cluster_map[n_row, n_col] == -1:
                            neighbor_token = tokens_grid[n_row, n_col]

                            # 코사인 유사도 계산
                            sim = torch.nn.functional.cosine_similarity(
                                cluster_mean.unsqueeze(0),
                                neighbor_token.unsqueeze(0),
                                dim=1
                            ).item()

                            # 임계값 이상이면 클러스터에 추가
                            if sim >= threshold:
                                cluster_map[n_row, n_col] = current_cluster_id
                                cluster_tokens.append(neighbor_token)
                                queue.append((n_row, n_col))

                current_cluster_id += 1

        # 클러스터 크기 계산
        cluster_sizes = {}
        for cluster_id in range(current_cluster_id):
            cluster_sizes[cluster_id] = np.sum(cluster_map == cluster_id)

        result = ClusterResult(
            cluster_map=cluster_map,
            num_clusters=current_cluster_id,
            cluster_sizes=cluster_sizes,
            threshold=threshold,
            similarity_result=similarity_result
        )

        print(f"✓ Found {current_cluster_id} clusters")
        print(f"  Cluster sizes: min={min(cluster_sizes.values())}, "
              f"max={max(cluster_sizes.values())}, "
              f"mean={np.mean(list(cluster_sizes.values())):.1f}")

        return result


class AdjacencyMerger:
    """인접 클러스터 감지 및 병합 클래스"""

    def __init__(self, threshold_mode: str = "dynamic", threshold_method: str = "mean-std",
                 threshold_value: Optional[float] = None, fixed_threshold: float = 0.8):
        """
        Args:
            threshold_mode: "dynamic" 또는 "fixed"
            threshold_method: 동적 threshold 계산 방식
            threshold_value: percentile/fixed 모드에서 필요한 값
            fixed_threshold: 고정 threshold (fixed 모드에서 사용)
        """
        self.threshold_mode = threshold_mode
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.fixed_threshold = fixed_threshold
        self.cls_similarity_threshold = fixed_threshold  # 기본값

    def find_adjacent_clusters(self, cluster_map: np.ndarray) -> List[Tuple[int, int]]:
        """
        서로 인접한 클러스터 쌍들을 찾습니다.

        Args:
            cluster_map: 클러스터 맵

        Returns:
            List[Tuple[int, int]]: 인접한 클러스터 ID 쌍들의 리스트
        """
        unique_clusters = np.unique(cluster_map)
        adjacent_pairs = set()  # 중복 방지를 위해 set 사용

        # 모든 좌표를 순회하며 인접한 클러스터 찾기
        for row in range(cluster_map.shape[0]):
            for col in range(cluster_map.shape[1]):
                current_cluster = cluster_map[row, col]

                # 4방향 이웃 확인
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_row, new_col = row + dr, col + dc

                    # 그리드 범위 내에 있고 다른 클러스터면
                    if (0 <= new_row < cluster_map.shape[0] and
                        0 <= new_col < cluster_map.shape[1]):
                        neighbor_cluster = cluster_map[new_row, new_col]

                        if current_cluster != neighbor_cluster:
                            # 정렬하여 쌍 추가 (중복 방지)
                            pair = tuple(sorted((current_cluster, neighbor_cluster)))
                            adjacent_pairs.add(pair)

        return list(adjacent_pairs)

    def compute_cluster_token_mean(self, cluster_result: ClusterResult) -> Dict[int, torch.Tensor]:
        """
        각 클러스터의 평균 토큰 벡터를 계산합니다.

        Args:
            cluster_result: 클러스터링 결과

        Returns:
            Dict[int, torch.Tensor]: 클러스터 ID -> 평균 토큰 벡터
        """
        cluster_token_means = {}

        if cluster_result.similarity_result.patch_tokens is None:
            print("  Warning: patch_tokens not available for token averaging")
            return cluster_token_means

        patch_tokens = cluster_result.similarity_result.patch_tokens  # (num_patches, hidden_dim)
        grid_size = cluster_result.cluster_map.shape[0]

        # 패치 토큰들을 그리드 형태로 재배열
        tokens_grid = patch_tokens.reshape(grid_size, grid_size, -1)  # (grid_size, grid_size, hidden_dim)

        for cluster_id in range(cluster_result.num_clusters):
            # 해당 클러스터의 좌표들
            coords = np.argwhere(cluster_result.cluster_map == cluster_id)
            if len(coords) == 0:
                continue

            # 해당 좌표들의 토큰들 모으기
            cluster_tokens = []
            for row, col in coords:
                cluster_tokens.append(tokens_grid[row, col])

            # 평균 토큰 계산
            if cluster_tokens:
                token_mean = torch.stack(cluster_tokens).mean(dim=0)
                cluster_token_means[cluster_id] = token_mean

        return cluster_token_means

    def compute_cluster_cls_similarity(self, cluster_result: ClusterResult) -> Dict[int, float]:
        """
        각 클러스터의 평균 CLS 토큰 유사도를 계산합니다.

        Args:
            cluster_result: 클러스터링 결과

        Returns:
            Dict[int, float]: 클러스터 ID -> 평균 CLS 유사도
        """
        cluster_cls_sims = {}

        for cluster_id in range(cluster_result.num_clusters):
            # 해당 클러스터의 좌표들
            coords = np.argwhere(cluster_result.cluster_map == cluster_id)
            if len(coords) == 0:
                continue

            # 해당 좌표들의 CLS 유사도 평균
            cls_values = []
            for row, col in coords:
                cls_values.append(cluster_result.similarity_result.cls_similarity[row, col])

            cluster_cls_sims[cluster_id] = np.mean(cls_values)

        return cluster_cls_sims

    def compute_dynamic_threshold(self, cls_sims: Dict[int, float]) -> float:
        """
        CLS 유사도 통계를 기반으로 동적 threshold를 계산합니다.

        Args:
            cls_sims: 클러스터 ID -> CLS 유사도

        Returns:
            float: 계산된 threshold
        """
        if not cls_sims:
            return self.fixed_threshold

        if self.threshold_method == "percentile":
            if self.threshold_value is None:
                raise ValueError("threshold_value must be specified for percentile mode")

            # 모든 클러스터 쌍의 CLS 유사도 차이를 계산
            sim_values = list(cls_sims.values())
            diffs = []
            for i in range(len(sim_values)):
                for j in range(i + 1, len(sim_values)):
                    diffs.append(abs(sim_values[i] - sim_values[j]))

            if not diffs:
                return self.fixed_threshold

            # CLS 유사도 차이들의 percentile을 threshold로 사용
            threshold = np.percentile(diffs, self.threshold_value)
        elif self.threshold_method == "fixed":
            threshold = self.fixed_threshold
        else:
            raise ValueError(f"Unknown threshold_method: {self.threshold_method}. Only 'percentile' and 'fixed' are supported.")

        # threshold를 [0.0, 1.0] 범위로 클램핑 (CLS 유사도는 0-1 범위)
        threshold = np.clip(threshold, 0.0, 1.0)

        return threshold

    def merge_adjacent_clusters(self, cluster_result: ClusterResult,
                               similarity_result: SimilarityResult) -> ClusterResult:
        """
        인접한 클러스터들을 CLS 유사도를 기반으로 병합합니다.

        Args:
            cluster_result: 클러스터링 결과
            similarity_result: 유사도 결과

        Returns:
            ClusterResult: 병합된 클러스터링 결과
        """
        # 인접한 클러스터 쌍 찾기
        adjacent_pairs = self.find_adjacent_clusters(cluster_result.cluster_map)

        if not adjacent_pairs:
            print("  No adjacent clusters found")
            return cluster_result

        print(f"  Found {len(adjacent_pairs)} adjacent cluster pairs")

        # 각 클러스터의 평균 CLS 유사도 계산
        cluster_cls_sims = self.compute_cluster_cls_similarity(cluster_result)

        # 동적 threshold 계산
        if self.threshold_mode == "dynamic":
            self.cls_similarity_threshold = self.compute_dynamic_threshold(cluster_cls_sims)
            print(f"  Computed dynamic threshold: {self.cls_similarity_threshold:.4f} (method: {self.threshold_method})")

            # CLS 유사도 통계 정보 출력
            sim_values = list(cluster_cls_sims.values())
            print(f"  CLS similarity stats: mean={np.mean(sim_values):.4f}, "
                  f"std={np.std(sim_values):.4f}, "
                  f"min={min(sim_values):.4f}, "
                  f"max={max(sim_values):.4f}")

        # 각 클러스터의 평균 토큰 벡터 계산
        cluster_token_means = self.compute_cluster_token_mean(cluster_result)

        # 병합할 클러스터 쌍 결정
        merge_pairs = []
        for cluster1, cluster2 in adjacent_pairs:
            cls_sim1 = cluster_cls_sims.get(cluster1, 0)
            cls_sim2 = cluster_cls_sims.get(cluster2, 0)

            # CLS 유사도 차이가 임계값보다 작으면 병합 후보
            cls_diff = abs(cls_sim1 - cls_sim2)
            if cls_diff <= self.cls_similarity_threshold:
                merge_pairs.append((cluster1, cluster2))
                print(f"    Adjacent clusters {cluster1} and {cluster2} "
                      f"(CLS similarity diff: {cls_diff:.4f}, threshold: {self.cls_similarity_threshold:.4f})")
            else:
                print(f"    Skip {cluster1}+{cluster2}: CLS similarity diff {cls_diff:.4f} > threshold {self.cls_similarity_threshold:.4f}")

        if not merge_pairs:
            print("  No adjacent clusters met similarity threshold for merging")
            return cluster_result

        # 병합 수행 (Union-Find 스타일)
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # 병합 관계 설정
        for cluster1, cluster2 in merge_pairs:
            union(cluster1, cluster2)

        # 새 클러스터 ID 할당
        unique_roots = {}
        new_cluster_map = np.zeros_like(cluster_result.cluster_map)
        next_id = 0

        for cluster_id in range(cluster_result.num_clusters):
            root = find(cluster_id)
            if root not in unique_roots:
                unique_roots[root] = next_id
                next_id += 1
            new_id = unique_roots[root]

            # 해당 클러스터의 모든 좌표에 새 ID 할당
            mask = (cluster_result.cluster_map == cluster_id)
            new_cluster_map[mask] = new_id

        # 새 클러스터 정보 계산
        new_num_clusters = next_id
        new_cluster_sizes = {}
        for new_id in range(new_num_clusters):
            new_cluster_sizes[new_id] = np.sum(new_cluster_map == new_id)

        print(f"  Merged {len(merge_pairs)} adjacent cluster pairs")
        print(f"  Cluster count: {cluster_result.num_clusters} -> {new_num_clusters}")

        return ClusterResult(
            cluster_map=new_cluster_map,
            num_clusters=new_num_clusters,
            cluster_sizes=new_cluster_sizes,
            threshold=cluster_result.threshold,
            similarity_result=cluster_result.similarity_result
        )


class ConnectivitySplitter:
    """Convex Hull 기반 클러스터 분할 클래스"""

    def __init__(self, min_split_size: int = 2, hull_threshold: float = 0.7):
        """
        Args:
            min_split_size: 분할할 최소 클러스터 크기
            hull_threshold: Convex Hull 채우기 비교 임계값 (낮을수록 더 민감하게 분할)
        """
        self.min_split_size = min_split_size
        self.hull_threshold = hull_threshold

    def compute_convex_hull_area(self, coords: List[Tuple[int, int]]) -> float:
        """
        좌표들의 Convex Hull 면적을 계산합니다.

        Args:
            coords: 좌표 리스트

        Returns:
            float: Convex Hull 면적
        """
        if len(coords) < 3:
            return float(len(coords))

        try:
            # scipy가 없는 경우를 대비한 fallback 구현
            from scipy.spatial import ConvexHull
            points = np.array(coords)
            hull = ConvexHull(points)
            return float(hull.volume)  # 2D에서는 volume이 area
        except ImportError:
            # 간단한 bounding box 면적으로 fallback
            coords_array = np.array(coords)
            min_coords = coords_array.min(axis=0)
            max_coords = coords_array.max(axis=0)
            return float(np.prod(max_coords - min_coords + 1))
        except Exception:
            # 오류 발생 시 bounding box 면적으로 fallback
            coords_array = np.array(coords)
            min_coords = coords_array.min(axis=0)
            max_coords = coords_array.max(axis=0)
            return float(np.prod(max_coords - min_coords + 1))

    def should_split_cluster(self, coords: List[Tuple[int, int]]) -> bool:
        """
        클러스터를 분할해야 하는지 결정합니다.

        Args:
            coords: 클러스터 좌표들

        Returns:
            bool: 분할하면 True
        """
        if len(coords) < self.min_split_size * 2:
            return False

        # 실제 클러스터 크기
        actual_size = len(coords)

        # Convex Hull 면적
        hull_area = self.compute_convex_hull_area(coords)

        # 채우기 비율: 실제 크기 / Convex Hull 면적
        fill_ratio = actual_size / hull_area

        print(f"    Cluster size: {actual_size}, Hull area: {hull_area:.1f}, "
              f"Fill ratio: {fill_ratio:.3f}")

        # 채우기 비율이 임계값보다 낮으면 분할 (구멍이 있거나 오목한 형태)
        return fill_ratio < self.hull_threshold

    def find_optimal_split(self, coords: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        클러스터의 최적 분할선을 찾습니다.

        Args:
            coords: 클러스터 좌표들

        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: 분할된 두 좌표 그룹
        """
        coords_array = np.array(coords)

        # 클러스터의 중심점 계산
        center = coords_array.mean(axis=0)

        # 수직 분할과 수평 분할 중 더 좋은 것 선택
        best_split1, best_split2 = None, None
        best_score = float('inf')

        # 수직 분할 (좌우)
        left_coords = []
        right_coords = []
        for coord in coords:
            if coord[1] < center[1]:  # col < center_col
                left_coords.append(coord)
            else:
                right_coords.append(coord)

        if len(left_coords) >= self.min_split_size and len(right_coords) >= self.min_split_size:
            # 분할 균형성 점수 (더 균등할수록 좋음)
            balance_score = abs(len(left_coords) - len(right_coords))
            if balance_score < best_score:
                best_score = balance_score
                best_split1, best_split2 = left_coords, right_coords

        # 수평 분할 (상하)
        top_coords = []
        bottom_coords = []
        for coord in coords:
            if coord[0] < center[0]:  # row < center_row
                top_coords.append(coord)
            else:
                bottom_coords.append(coord)

        if len(top_coords) >= self.min_split_size and len(bottom_coords) >= self.min_split_size:
            balance_score = abs(len(top_coords) - len(bottom_coords))
            if balance_score < best_score:
                best_score = balance_score
                best_split1, best_split2 = top_coords, bottom_coords

        # 최적 분할을 찾지 못한 경우, 대각선 분할 시도
        if best_split1 is None:
            # 주성분 분석 기반 분할 (scipy가 없는 경우 간단한 대각선)
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                pca.fit(coords_array)
                projected = pca.transform(coords_array)
                median_val = np.median(projected)

                group1, group2 = [], []
                for i, coord in enumerate(coords):
                    if projected[i] < median_val:
                        group1.append(coord)
                    else:
                        group2.append(coord)

                if len(group1) >= self.min_split_size and len(group2) >= self.min_split_size:
                    best_split1, best_split2 = group1, group2
            except ImportError:
                # 간단한 대각선 분할 (y = x 기준)
                diag1, diag2 = [], []
                for coord in coords:
                    if coord[0] + coord[1] < 2 * center.mean():
                        diag1.append(coord)
                    else:
                        diag2.append(coord)

                if len(diag1) >= self.min_split_size and len(diag2) >= self.min_split_size:
                    best_split1, best_split2 = diag1, diag2

        # 최적 분할을 찾지 못한 경우, 임의로 반으로 나눔
        if best_split1 is None:
            mid = len(coords) // 2
            best_split1 = coords[:mid]
            best_split2 = coords[mid:]

        return best_split1, best_split2

    def split_clusters(self, cluster_result: ClusterResult) -> ClusterResult:
        """
        모든 클러스터에 대해 Convex Hull 기반 분할을 수행합니다.

        Args:
            cluster_result: 클러스터링 결과

        Returns:
            ClusterResult: 분할된 클러스터링 결과
        """
        new_cluster_map = np.full_like(cluster_result.cluster_map, -1)
        current_cluster_id = 0
        total_splits = 0

        for cluster_id in range(cluster_result.num_clusters):
            # 현재 클러스터의 좌표들
            coords = np.argwhere(cluster_result.cluster_map == cluster_id)
            coord_list = [(int(row), int(col)) for row, col in coords]

            print(f"  Analyzing cluster {cluster_id} ({len(coord_list)} tokens)...")

            # 분할 여부 확인
            if self.should_split_cluster(coord_list):
                # 최적 분할 찾기
                split1, split2 = self.find_optimal_split(coord_list)

                # 분할 수행
                for row, col in split1:
                    new_cluster_map[row, col] = current_cluster_id
                current_cluster_id += 1

                for row, col in split2:
                    new_cluster_map[row, col] = current_cluster_id
                current_cluster_id += 1

                total_splits += 1
                print(f"    Split cluster {cluster_id} into {len(split1)} and {len(split2)} tokens")
            else:
                # 분할하지 않고 그대로 유지
                for row, col in coord_list:
                    new_cluster_map[row, col] = current_cluster_id
                current_cluster_id += 1

        # 새 클러스터 정보 계산
        new_num_clusters = current_cluster_id
        new_cluster_sizes = {}
        for new_id in range(new_num_clusters):
            new_cluster_sizes[new_id] = np.sum(new_cluster_map == new_id)

        print(f"  Convex Hull-based splitting: {cluster_result.num_clusters} -> {new_num_clusters} clusters")
        print(f"  Split {total_splits} clusters total")

        return ClusterResult(
            cluster_map=new_cluster_map,
            num_clusters=new_num_clusters,
            cluster_sizes=new_cluster_sizes,
            threshold=cluster_result.threshold,
            similarity_result=cluster_result.similarity_result
        )


class SimilarityVisualizer:
    """토큰 유사도 분석 결과를 시각화하는 클래스"""

    def __init__(self, figsize: Tuple[int, int] = (20, 16), cmap: str = "viridis", dpi: int = 150):
        """
        Args:
            figsize: 그림 크기
            cmap: 컬러맵
            dpi: 해상도
        """
        self.figsize = figsize
        self.cmap = cmap
        self.dpi = dpi

    def visualize_all(self, result: SimilarityResult, save_path: Optional[str] = None):
        """
        모든 유사도를 한 번에 시각화합니다.

        Args:
            result: SimilarityResult 객체
            save_path: 저장 경로
        """
        fig, axes = plt.subplots(3, 3, figsize=(self.figsize[0], self.figsize[1] * 1.2))

        # 첫 번째 행: 히트맵
        # 1. 오른쪽 방향 유사도
        im1 = axes[0, 0].imshow(result.horizontal_right, cmap=self.cmap, aspect='auto')
        axes[0, 0].set_title(f'Horizontal Similarity (→)\n({result.grid_size}×{result.grid_size-1})', fontsize=11)
        axes[0, 0].set_xlabel('Position', fontsize=9)
        axes[0, 0].set_ylabel('Row', fontsize=9)
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 2. 아래쪽 방향 유사도
        im2 = axes[0, 1].imshow(result.vertical_down, cmap=self.cmap, aspect='auto')
        axes[0, 1].set_title(f'Vertical Similarity (↓)\n({result.grid_size-1}×{result.grid_size})', fontsize=11)
        axes[0, 1].set_xlabel('Column', fontsize=9)
        axes[0, 1].set_ylabel('Position', fontsize=9)
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 3. CLS 토큰 유사도
        im3 = axes[0, 2].imshow(result.cls_similarity, cmap=self.cmap, aspect='equal')
        axes[0, 2].set_title(f'CLS Token Similarity\n({result.grid_size}×{result.grid_size})', fontsize=11)
        axes[0, 2].set_xlabel('Column', fontsize=9)
        axes[0, 2].set_ylabel('Row', fontsize=9)
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # 두 번째 행: 분포 히스토그램
        # 4. 가로 방향 유사도 분포
        h_data = result.horizontal_right.flatten()
        axes[1, 0].hist(h_data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 0].set_title('Horizontal Similarity Distribution', fontsize=11)
        axes[1, 0].set_xlabel('Similarity', fontsize=9)
        axes[1, 0].set_ylabel('Frequency', fontsize=9)
        axes[1, 0].axvline(h_data.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {h_data.mean():.4f}')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # 5. 세로 방향 유사도 분포
        v_data = result.vertical_down.flatten()
        axes[1, 1].hist(v_data, bins=50, edgecolor='black', alpha=0.7, color='darkorange')
        axes[1, 1].set_title('Vertical Similarity Distribution', fontsize=11)
        axes[1, 1].set_xlabel('Similarity', fontsize=9)
        axes[1, 1].set_ylabel('Frequency', fontsize=9)
        axes[1, 1].axvline(v_data.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {v_data.mean():.4f}')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        # 6. CLS 유사도 분포
        cls_data = result.cls_similarity.flatten()
        axes[1, 2].hist(cls_data, bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
        axes[1, 2].set_title('CLS Similarity Distribution', fontsize=11)
        axes[1, 2].set_xlabel('Similarity', fontsize=9)
        axes[1, 2].set_ylabel('Frequency', fontsize=9)
        axes[1, 2].axvline(cls_data.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {cls_data.mean():.4f}')
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)

        # 세 번째 행: 통합 분포 비교
        # 7. 모든 유사도 분포 오버레이
        axes[2, 0].hist(h_data, bins=50, alpha=0.5, label='Horizontal', color='steelblue', edgecolor='black')
        axes[2, 0].hist(v_data, bins=50, alpha=0.5, label='Vertical', color='darkorange', edgecolor='black')
        axes[2, 0].hist(cls_data, bins=50, alpha=0.5, label='CLS', color='forestgreen', edgecolor='black')
        axes[2, 0].set_title('All Similarities Distribution Overlay', fontsize=11)
        axes[2, 0].set_xlabel('Similarity', fontsize=9)
        axes[2, 0].set_ylabel('Frequency', fontsize=9)
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].grid(True, alpha=0.3)

        # 8. 통계 박스플롯
        data_for_box = [h_data, v_data, cls_data]
        bp = axes[2, 1].boxplot(data_for_box, labels=['Horizontal', 'Vertical', 'CLS'],
                                patch_artist=True, showmeans=True)
        colors = ['steelblue', 'darkorange', 'forestgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[2, 1].set_title('Similarity Statistics Comparison', fontsize=11)
        axes[2, 1].set_ylabel('Similarity', fontsize=9)
        axes[2, 1].grid(True, alpha=0.3, axis='y')

        # 9. 통계 테이블
        axes[2, 2].axis('off')
        stats_text = f"""
        Statistics Summary
        ═══════════════════════════════

        Horizontal (→):
          Mean:  {h_data.mean():.4f}
          Std:   {h_data.std():.4f}
          Min:   {h_data.min():.4f}
          Max:   {h_data.max():.4f}

        Vertical (↓):
          Mean:  {v_data.mean():.4f}
          Std:   {v_data.std():.4f}
          Min:   {v_data.min():.4f}
          Max:   {v_data.max():.4f}

        CLS Token:
          Mean:  {cls_data.mean():.4f}
          Std:   {cls_data.std():.4f}
          Min:   {cls_data.min():.4f}
          Max:   {cls_data.max():.4f}
        """
        axes[2, 2].text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
                       fontfamily='monospace', transform=axes[2, 2].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved combined visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_individual(self, result: SimilarityResult, output_dir: str):
        """
        각 유사도를 개별 파일로 저장합니다.

        Args:
            result: SimilarityResult 객체
            output_dir: 출력 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 오른쪽 방향
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(result.horizontal_right, cmap=self.cmap, aspect='auto')
        ax.set_title(f'Horizontal Similarity (→)', fontsize=14)
        ax.set_xlabel('Position')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / "horizontal_similarity.png", dpi=self.dpi, bbox_inches='tight')
        plt.close()

        # 2. 아래쪽 방향
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(result.vertical_down, cmap=self.cmap, aspect='auto')
        ax.set_title(f'Vertical Similarity (↓)', fontsize=14)
        ax.set_xlabel('Column')
        ax.set_ylabel('Position')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / "vertical_similarity.png", dpi=self.dpi, bbox_inches='tight')
        plt.close()

        # 3. CLS 토큰
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(result.cls_similarity, cmap=self.cmap, aspect='equal')
        ax.set_title(f'CLS Token Similarity', fontsize=14)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / "cls_similarity.png", dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved individual visualizations to {output_dir}")

    def visualize_clusters_on_image(self, cluster_result: ClusterResult, image_path: str,
                                    analyzer: 'TokenSimilarityAnalyzer',
                                    save_path: Optional[str] = None, line_width: int = 8):
        """
        이미지 위에 클러스터를 두꺼운 색상 테두리로 시각화합니다.
        각 패치의 4개 변을 모두 그려서 겹치는 부분도 보이게 합니다.
        CLIP에 실제로 들어간 전처리된 이미지를 사용합니다.

        Args:
            cluster_result: 클러스터링 결과
            image_path: 원본 이미지 경로
            analyzer: TokenSimilarityAnalyzer 인스턴스 (전처리된 이미지 추출용)
            save_path: 저장 경로
            line_width: 테두리 두께 (기본값 8)
        """
        # CLIP 전처리된 이미지 사용 (실제로 모델에 들어간 이미지)
        image = analyzer.get_preprocessed_image(image_path)
        img_width, img_height = image.size

        # 그리드 크기
        grid_size = cluster_result.cluster_map.shape[0]
        patch_width = img_width / grid_size
        patch_height = img_height / grid_size

        # 클러스터별 매우 구분되는 색상 생성 (밝고 선명한 색상)
        num_clusters = cluster_result.num_clusters
        colors = []

        # 미리 정의된 구분되는 색상들
        predefined_colors = [
            (255, 0, 0),      # 빨강
            (0, 255, 0),      # 초록
            (0, 0, 255),      # 파랑
            (255, 255, 0),    # 노랑
            (255, 0, 255),    # 마젠타
            (0, 255, 255),    # 시안
            (255, 128, 0),    # 주황
            (128, 0, 255),    # 보라
            (255, 0, 128),    # 핑크
            (0, 255, 128),    # 민트
            (128, 255, 0),    # 라임
            (0, 128, 255),    # 하늘
            (255, 128, 128),  # 연한 빨강
            (128, 255, 128),  # 연한 초록
            (128, 128, 255),  # 연한 파랑
            (255, 255, 128),  # 연한 노랑
            (255, 128, 255),  # 연한 마젠타
            (128, 255, 255),  # 연한 시안
            (192, 64, 0),     # 갈색
            (64, 192, 192),   # 청록
        ]

        # 클러스터 수만큼 색상 생성
        for i in range(num_clusters):
            if i < len(predefined_colors):
                colors.append(predefined_colors[i])
            else:
                # 추가 색상이 필요하면 HSV로 생성
                hue = (i - len(predefined_colors)) / max(num_clusters - len(predefined_colors), 1)
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                colors.append(tuple(int(c * 255) for c in rgb))

        # RGBA 이미지로 변환하여 투명도 지원
        img_rgba = image.convert('RGBA')

        # 테두리만 그릴 오버레이 생성
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # 각 패치의 4개 변을 모두 그림 (겹치는 부분도 보이게)
        for row in range(grid_size):
            for col in range(grid_size):
                cluster_id = cluster_result.cluster_map[row, col]
                color = colors[cluster_id % len(colors)]

                x1 = int(col * patch_width)
                y1 = int(row * patch_height)
                x2 = int((col + 1) * patch_width)
                y2 = int((row + 1) * patch_height)

                # 4개 변을 모두 그림 (불투명하게)
                # 위쪽
                draw.line([(x1, y1), (x2, y1)], fill=color + (255,), width=line_width)

                # 아래쪽
                draw.line([(x1, y2), (x2, y2)], fill=color + (255,), width=line_width)

                # 왼쪽
                draw.line([(x1, y1), (x1, y2)], fill=color + (255,), width=line_width)

                # 오른쪽
                draw.line([(x2, y1), (x2, y2)], fill=color + (255,), width=line_width)

        # 이미지 합성
        img_rgba = image.convert('RGBA')
        result_img = Image.alpha_composite(img_rgba, overlay)
        result_img = result_img.convert('RGB')

        # 시각화 (matplotlib으로)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(result_img)
        ax.set_title(f'Token Clusters on Image\n{num_clusters} clusters (threshold={cluster_result.threshold:.4f})',
                    fontsize=14)
        ax.axis('off')

        # 범례 추가 (상위 10개 클러스터만)
        sorted_clusters = sorted(cluster_result.cluster_sizes.items(),
                               key=lambda x: x[1], reverse=True)[:10]

        legend_elements = []
        for cluster_id, size in sorted_clusters:
            color_norm = tuple(c / 255.0 for c in colors[cluster_id % len(colors)])
            legend_elements.append(mpatches.Patch(color=color_norm,
                                                 label=f'Cluster {cluster_id} (size={size})'))

        if len(cluster_result.cluster_sizes) > 10:
            legend_elements.append(mpatches.Patch(color='gray', alpha=0.3,
                                                 label=f'... and {num_clusters - 10} more'))

        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved cluster visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_cluster_map(self, cluster_result: ClusterResult, save_path: Optional[str] = None):
        """
        클러스터 맵을 히트맵으로 시각화합니다.
        이미지 오버레이와 동일한 색상을 사용합니다.

        Args:
            cluster_result: 클러스터링 결과
            save_path: 저장 경로
        """
        # 이미지 오버레이와 동일한 색상 생성
        num_clusters = cluster_result.num_clusters

        predefined_colors = [
            (255, 0, 0),      # 빨강
            (0, 255, 0),      # 초록
            (0, 0, 255),      # 파랑
            (255, 255, 0),    # 노랑
            (255, 0, 255),    # 마젠타
            (0, 255, 255),    # 시안
            (255, 128, 0),    # 주황
            (128, 0, 255),    # 보라
            (255, 0, 128),    # 핑크
            (0, 255, 128),    # 민트
            (128, 255, 0),    # 라임
            (0, 128, 255),    # 하늘
            (255, 128, 128),  # 연한 빨강
            (128, 255, 128),  # 연한 초록
            (128, 128, 255),  # 연한 파랑
            (255, 255, 128),  # 연한 노랑
            (255, 128, 255),  # 연한 마젠타
            (128, 255, 255),  # 연한 시안
            (192, 64, 0),     # 갈색
            (64, 192, 192),   # 청록
        ]

        colors_normalized = []
        for i in range(num_clusters):
            if i < len(predefined_colors):
                rgb = predefined_colors[i]
                colors_normalized.append(tuple(c / 255.0 for c in rgb))
            else:
                hue = (i - len(predefined_colors)) / max(num_clusters - len(predefined_colors), 1)
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                colors_normalized.append(rgb)

        # 커스텀 컬러맵 생성
        custom_cmap = ListedColormap(colors_normalized)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 1. 클러스터 맵 (커스텀 색상)
        im1 = axes[0].imshow(cluster_result.cluster_map, cmap=custom_cmap, aspect='equal',
                            vmin=0, vmax=num_clusters-1)
        axes[0].set_title(f'Cluster Map\n{cluster_result.num_clusters} clusters', fontsize=14)
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')

        # 컬러바 추가 (클러스터 ID 표시)
        cbar = plt.colorbar(im1, ax=axes[0], label='Cluster ID', ticks=range(min(num_clusters, 20)))
        if num_clusters <= 20:
            cbar.set_ticks(range(num_clusters))
            cbar.set_ticklabels([str(i) for i in range(num_clusters)])

        # 2. 클러스터 크기 분포
        cluster_sizes = list(cluster_result.cluster_sizes.values())
        axes[1].hist(cluster_sizes, bins=min(30, cluster_result.num_clusters),
                    edgecolor='black', alpha=0.7, color='steelblue')
        axes[1].set_title('Cluster Size Distribution', fontsize=14)
        axes[1].set_xlabel('Cluster Size (number of tokens)')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(np.mean(cluster_sizes), color='red', linestyle='--',
                       label=f'Mean: {np.mean(cluster_sizes):.1f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 통계 텍스트
        stats_text = f"""
        Total clusters: {cluster_result.num_clusters}
        Threshold: {cluster_result.threshold:.4f}
        Min size: {min(cluster_sizes)}
        Max size: {max(cluster_sizes)}
        Mean size: {np.mean(cluster_sizes):.1f}
        Std size: {np.std(cluster_sizes):.1f}
        """
        axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved cluster map visualization to {save_path}")
        else:
            plt.show()

        plt.close()
