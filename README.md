# CLIP Vision Token Visualizer

CLIP 모델의 비전 토큰을 이미지에 오버레이하여 시각화하는 도구입니다.

## 지원 모델

- `clip-vit-large-patch14` (224x224, 16x16 grid)
- `clip-vit-large-patch14-336` (336x336, 24x24 grid)

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 실행

```bash
python visualize_clip_tokens.py
```

### 코드에서 사용

```python
from visualize_clip_tokens import CLIPTokenVisualizer

# 모델 초기화 (224x224, 자동으로 GPU 선택)
visualizer = CLIPTokenVisualizer("clip-vit-large-patch14")

# 특정 GPU 지정
visualizer_gpu1 = CLIPTokenVisualizer("clip-vit-large-patch14", device=1)
visualizer_gpu2 = CLIPTokenVisualizer("clip-vit-large-patch14", device="cuda:2")

# CPU 사용
visualizer_cpu = CLIPTokenVisualizer("clip-vit-large-patch14", device="cpu")

# 토큰 그리드 시각화
visualizer.visualize_token_grid(
    "your_image.jpg",
    save_path="output_grid.png",
    show_indices=True,  # 패치 인덱스 표시
    alpha=0.4  # 그리드 투명도
)

# 토큰 어텐션 히트맵
visualizer.visualize_token_attention(
    "your_image.jpg",
    save_path="output_attention.png",
    cmap='viridis'
)

# 336x336 모델 사용
visualizer_336 = CLIPTokenVisualizer("clip-vit-large-patch14-336", device=0)
visualizer_336.visualize_token_grid("your_image.jpg", save_path="output_336.png")
```

### CLI에서 디바이스 지정

```bash
# GPU 0 사용
python example_custom.py my_image.jpg --device 0

# GPU 1 사용
python example_custom.py my_image.jpg clip-vit-large-patch14 --device cuda:1

# CPU 사용
python example_custom.py my_image.jpg --device cpu

# 두 모델 비교 (GPU 2 사용)
python example_custom.py my_image.jpg compare --device 2
```

## 출력 파일

- `token_grid_224.png` - 224 모델의 토큰 그리드 오버레이
- `token_attention_224.png` - 224 모델의 토큰 L2 norm 히트맵
- `token_grid_336.png` - 336 모델의 토큰 그리드 오버레이
- `token_attention_336.png` - 336 모델의 토큰 L2 norm 히트맵

## 기능

1. **토큰 그리드 시각화**: 이미지 위에 비전 패치를 그리드로 표시
2. **패치 인덱스**: 각 패치의 인덱스 번호 표시
3. **어텐션 히트맵**: 토큰의 L2 norm을 기반으로 한 히트맵
4. **모델 교체**: 설정으로 쉽게 다른 CLIP 모델 사용 가능
