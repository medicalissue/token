# Quick Start Guide

CLIP 토큰 유사도 분석을 빠르게 시작하는 가이드입니다.

## 1단계: 설치

```bash
cd /Users/medicalissue/Desktop/token
pip install -r requirements.txt
```

## 2단계: 간단한 테스트 실행

### 방법 1: 테스트 스크립트 (가장 간단)

```bash
python test_similarity.py
```

이 명령어는:
- 자동으로 테스트 이미지 생성
- 토큰 유사도 분석 실행
- `test_outputs/` 디렉토리에 결과 저장

### 방법 2: Hydra로 실행

```bash
# 먼저 샘플 이미지 생성
python visualize_clip_tokens.py

# 유사도 분석 실행
python run_similarity_analysis.py
```

결과는 `outputs/` 디렉토리에 저장됩니다.

## 3단계: 자신의 이미지로 실행

```bash
# 테스트 스크립트 방식
python test_similarity.py  # 코드에서 image_path 수정 필요

# Hydra 방식 (권장)
python run_similarity_analysis.py image.path=your_image.jpg
```

## 주요 명령어

### 기본 사용

```bash
# 기본 실행 (224 모델, 자동 디바이스)
python run_similarity_analysis.py

# 다른 이미지
python run_similarity_analysis.py image.path=cat.jpg

# 336 모델 사용 (더 세밀한 분석)
python run_similarity_analysis.py model=clip_vit_large_patch14_336

# GPU 지정
python run_similarity_analysis.py device=0
python run_similarity_analysis.py device=cuda:1

# CPU 사용
python run_similarity_analysis.py device=cpu
```

### 고급 설정

```bash
# 내적 유사도 사용 (코사인 대신)
python run_similarity_analysis.py similarity.metric=dot_product

# 시각화 설정
python run_similarity_analysis.py visualization.cmap=hot visualization.figsize=[24,20]

# 출력 디렉토리 변경
python run_similarity_analysis.py output.dir=my_results

# 여러 설정 동시 변경
python run_similarity_analysis.py \
    image.path=dog.jpg \
    model=clip_vit_large_patch14_336 \
    device=1 \
    output.dir=dog_analysis \
    visualization.cmap=plasma
```

### 배치 실행

```bash
# 여러 이미지 동시 분석
python run_similarity_analysis.py --multirun image.path=img1.jpg,img2.jpg,img3.jpg

# 두 모델 비교
python run_similarity_analysis.py --multirun model=clip_vit_large_patch14,clip_vit_large_patch14_336
```

## 출력 결과

### 디렉토리 구조

```
outputs/  (또는 test_outputs/)
├── config.yaml                              # 사용된 설정
├── similarities.npz                         # 유사도 데이터
├── similarity_analysis_combined.png         # 통합 시각화 ⭐
└── individual/
    ├── horizontal_similarity.png            # 수평 유사도
    ├── vertical_similarity.png              # 수직 유사도
    └── cls_similarity.png                   # CLS 토큰 유사도
```

### 통합 시각화 내용

`similarity_analysis_combined.png` 파일에는 6개의 서브플롯이 포함됩니다:

1. **Horizontal Similarity (→)**: 오른쪽 이웃과의 유사도 히트맵
2. **Vertical Similarity (↓)**: 아래쪽 이웃과의 유사도 히트맵
3. **CLS Token Similarity**: 모든 패치와 CLS 토큰의 유사도 히트맵
4. **Mean Horizontal Similarity**: 행별 평균 수평 유사도 그래프
5. **Mean Vertical Similarity**: 열별 평균 수직 유사도 그래프
6. **CLS Similarity Distribution**: CLS 유사도 분포 히스토그램

## Python 코드에서 사용

```python
from token_similarity_analyzer import TokenSimilarityAnalyzer, SimilarityVisualizer

# 1. Analyzer 초기화
analyzer = TokenSimilarityAnalyzer(
    model_type="clip-vit-large-patch14",
    device=0,
    similarity_metric="cosine"
)

# 2. 분석 실행
result = analyzer.analyze("your_image.jpg")

# 3. 결과 확인
print(f"Grid: {result.grid_size}x{result.grid_size}")
print(f"Horizontal mean: {result.horizontal_right.mean():.4f}")
print(f"CLS mean: {result.cls_similarity.mean():.4f}")

# 4. 시각화
visualizer = SimilarityVisualizer()
visualizer.visualize_all(result, save_path="output.png")
```

## 설정 파일 커스터마이징

`configs/config.yaml`을 직접 수정하거나 새로운 설정 파일을 만들 수 있습니다:

```yaml
# configs/my_config.yaml
defaults:
  - config
  - model: clip_vit_large_patch14_336

image:
  path: "my_image.jpg"

device: 0

similarity:
  metric: "cosine"

visualization:
  cmap: "plasma"
  figsize: [24, 20]

output:
  dir: "my_results"
  dpi: 300
```

실행:
```bash
python run_similarity_analysis.py --config-name=my_config
```

## 문제 해결

### CUDA Out of Memory
```bash
# 더 작은 모델 사용
python run_similarity_analysis.py model=clip_vit_large_patch14

# CPU 사용
python run_similarity_analysis.py device=cpu
```

### 모델 다운로드 느림
처음 실행 시 Hugging Face에서 모델을 다운로드합니다. 이후에는 캐시된 모델을 사용합니다.

### 이미지 파일 경로 오류
절대 경로를 사용하거나 현재 디렉토리를 확인하세요:
```bash
python run_similarity_analysis.py image.path=/full/path/to/image.jpg
```

## 다음 단계

- 📖 전체 문서: [README_similarity.md](README_similarity.md)
- 🎨 시각화 예시: [visualize_clip_tokens.py](visualize_clip_tokens.py)
- ⚙️ 설정 파일: [configs/](configs/)

## 요약

```bash
# 가장 빠른 방법
python test_similarity.py

# 프로덕션 방법
python run_similarity_analysis.py image.path=your_image.jpg device=0
```

끝! 🎉
