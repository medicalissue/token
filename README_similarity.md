# CLIP Token Similarity Analyzer

CLIP 모델의 비전 토큰 간 유사도를 분석하고 시각화하는 도구입니다.

## 기능

1. **인접 토큰 유사도 계산**: 격자 형태로 배열된 토큰들의 상하좌우 이웃과의 코사인 유사도 계산
2. **CLS 토큰 유사도 계산**: 모든 패치 토큰과 CLS 토큰 간의 유사도 계산
3. **자동 시각화**: 모든 유사도를 히트맵 및 그래프로 시각화
4. **Hydra 설정 관리**: 모든 파라미터를 YAML 파일로 관리

## 설치

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
token/
├── configs/
│   ├── config.yaml                          # 메인 설정 파일
│   └── model/
│       ├── clip_vit_large_patch14.yaml      # 224 모델 설정
│       └── clip_vit_large_patch14_336.yaml  # 336 모델 설정
├── token_similarity_analyzer.py              # 분석 및 시각화 클래스
├── run_similarity_analysis.py                # Hydra 통합 메인 스크립트
└── outputs/                                  # 출력 디렉토리 (자동 생성)
```

## 사용법

### 기본 실행

```bash
# 기본 설정으로 실행
python run_similarity_analysis.py

# 다른 이미지 사용
python run_similarity_analysis.py image.path=your_image.jpg

# 336 모델 사용
python run_similarity_analysis.py model=clip_vit_large_patch14_336

# GPU 지정
python run_similarity_analysis.py device=0
python run_similarity_analysis.py device=cuda:1

# 여러 설정 동시 변경
python run_similarity_analysis.py image.path=my_image.jpg model=clip_vit_large_patch14_336 device=1
```

### Hydra 오버라이드

```bash
# 유사도 메트릭 변경
python run_similarity_analysis.py similarity.metric=dot_product

# 시각화 설정 변경
python run_similarity_analysis.py visualization.cmap=hot visualization.figsize=[24,20]

# 출력 디렉토리 변경
python run_similarity_analysis.py output.dir=my_results

# 개별 시각화만 저장
python run_similarity_analysis.py visualization.save_combined=false
```

### 멀티런 (여러 설정 동시 실행)

```bash
# 여러 이미지 동시 분석
python run_similarity_analysis.py --multirun image.path=image1.jpg,image2.jpg,image3.jpg

# 두 모델 비교
python run_similarity_analysis.py --multirun model=clip_vit_large_patch14,clip_vit_large_patch14_336

# 여러 GPU에서 병렬 실행
python run_similarity_analysis.py --multirun device=0,1,2
```

## 출력 결과

실행 후 `outputs/` 디렉토리에 다음 파일들이 생성됩니다:

```
outputs/
├── config.yaml                              # 사용된 설정
├── similarities.npz                         # 유사도 데이터 (NumPy)
├── similarity_analysis_combined.png         # 통합 시각화
└── individual/                              # 개별 시각화
    ├── horizontal_similarity.png            # 수평 방향 유사도
    ├── vertical_similarity.png              # 수직 방향 유사도
    └── cls_similarity.png                   # CLS 토큰 유사도
```

### 유사도 데이터 로드

```python
import numpy as np

# 저장된 유사도 데이터 로드
data = np.load("outputs/similarities.npz")

horizontal_right = data['horizontal_right']  # (N, M-1) 오른쪽 이웃 유사도
vertical_down = data['vertical_down']        # (N-1, M) 아래쪽 이웃 유사도
cls_similarity = data['cls_similarity']      # (N, M) CLS 토큰 유사도
grid_size = data['grid_size']                # 그리드 크기
```

## 설정 파일

### config.yaml

```yaml
# 모델 설정
defaults:
  - model: clip_vit_large_patch14  # 또는 clip_vit_large_patch14_336

# 이미지 경로
image:
  path: "sample_image.jpg"

# 디바이스 (null=자동, "cpu", "cuda", "cuda:0", 정수)
device: null

# 출력 설정
output:
  dir: "outputs"
  save_similarities: true
  save_visualizations: true
  format: "png"
  dpi: 150

# 유사도 계산 방식
similarity:
  metric: "cosine"  # cosine 또는 dot_product

# 시각화 설정
visualization:
  figsize: [20, 16]
  cmap: "viridis"
  save_individual: true
  save_combined: true
```

## 분석 결과 해석

### 1. Horizontal Similarity (→)
- 각 토큰과 오른쪽 이웃 토큰 간의 유사도
- 형태: (grid_size, grid_size-1)
- 높은 값: 좌우로 유사한 패턴/특징 존재

### 2. Vertical Similarity (↓)
- 각 토큰과 아래쪽 이웃 토큰 간의 유사도
- 형태: (grid_size-1, grid_size)
- 높은 값: 상하로 유사한 패턴/특징 존재

### 3. CLS Token Similarity
- 각 패치 토큰과 CLS 토큰 간의 유사도
- 형태: (grid_size, grid_size)
- 높은 값: 해당 패치가 전체 이미지를 대표하는 특징 포함

## Python 코드에서 직접 사용

```python
from token_similarity_analyzer import TokenSimilarityAnalyzer, SimilarityVisualizer

# Analyzer 초기화
analyzer = TokenSimilarityAnalyzer(
    model_type="clip-vit-large-patch14",
    device=0,
    similarity_metric="cosine"
)

# 분석 실행
result = analyzer.analyze("your_image.jpg")

# 결과 접근
print(f"Grid size: {result.grid_size}x{result.grid_size}")
print(f"Horizontal similarity shape: {result.horizontal_right.shape}")
print(f"CLS similarity mean: {result.cls_similarity.mean():.4f}")

# 시각화
visualizer = SimilarityVisualizer(figsize=(20, 16), cmap="viridis")
visualizer.visualize_all(result, save_path="my_analysis.png")
visualizer.visualize_individual(result, output_dir="my_results")
```

## 예시

### 예시 1: 기본 분석
```bash
python run_similarity_analysis.py image.path=cat.jpg
```

### 예시 2: 고해상도 모델로 GPU 1에서 실행
```bash
python run_similarity_analysis.py \
    model=clip_vit_large_patch14_336 \
    image.path=dog.jpg \
    device=1 \
    output.dir=results_dog
```

### 예시 3: 내적 유사도 사용
```bash
python run_similarity_analysis.py \
    similarity.metric=dot_product \
    visualization.cmap=hot
```

## 고급 사용법

### 커스텀 설정 파일 생성

```bash
# my_config.yaml 생성
cat > configs/my_config.yaml << EOF
defaults:
  - config
  - _self_

image:
  path: "my_special_image.jpg"

device: 0

output:
  dir: "my_outputs"
  dpi: 300

visualization:
  cmap: "plasma"
  figsize: [24, 20]
EOF

# 커스텀 설정으로 실행
python run_similarity_analysis.py --config-name=my_config
```

### 배치 처리

```python
from pathlib import Path
from token_similarity_analyzer import TokenSimilarityAnalyzer, SimilarityVisualizer

analyzer = TokenSimilarityAnalyzer("clip-vit-large-patch14", device=0)
visualizer = SimilarityVisualizer()

image_dir = Path("images")
output_dir = Path("batch_results")
output_dir.mkdir(exist_ok=True)

for image_path in image_dir.glob("*.jpg"):
    print(f"Processing {image_path.name}...")
    result = analyzer.analyze(str(image_path))

    out_path = output_dir / f"{image_path.stem}_analysis.png"
    visualizer.visualize_all(result, save_path=str(out_path))
```

## 문제 해결

### CUDA Out of Memory
- 더 작은 모델 사용: `model=clip_vit_large_patch14`
- CPU 사용: `device=cpu`

### Hydra working directory 문제
```bash
# 원본 디렉토리에서 실행
python run_similarity_analysis.py hydra.run.dir=.
```

## 참고

- CLIP 모델: [OpenAI CLIP](https://github.com/openai/CLIP)
- Hydra: [Hydra Documentation](https://hydra.cc)
