import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from typing import Literal, Tuple
import os


class CLIPTokenVisualizer:
    """CLIP 비전 토큰을 시각화하는 클래스"""

    MODEL_CONFIGS = {
        "clip-vit-large-patch14": {
            "model_name": "openai/clip-vit-large-patch14",
            "image_size": 224,
            "patch_size": 14,
            "grid_size": 16  # 224 / 14 = 16
        },
        "clip-vit-large-patch14-336": {
            "model_name": "openai/clip-vit-large-patch14-336",
            "image_size": 336,
            "patch_size": 14,
            "grid_size": 24  # 336 / 14 = 24
        }
    }

    def __init__(self, model_type: Literal["clip-vit-large-patch14", "clip-vit-large-patch14-336"] = "clip-vit-large-patch14"):
        """
        Args:
            model_type: 사용할 CLIP 모델 타입
        """
        self.config = self.MODEL_CONFIGS[model_type]
        self.model_type = model_type

        print(f"Loading {model_type}...")
        self.model = CLIPModel.from_pretrained(self.config["model_name"])
        self.processor = CLIPProcessor.from_pretrained(self.config["model_name"])
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Image size: {self.config['image_size']}x{self.config['image_size']}")
        print(f"Patch size: {self.config['patch_size']}x{self.config['patch_size']}")
        print(f"Grid size: {self.config['grid_size']}x{self.config['grid_size']}")

    def extract_vision_tokens(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """
        이미지에서 비전 토큰을 추출합니다.

        Args:
            image_path: 이미지 파일 경로

        Returns:
            vision_tokens: 비전 토큰 텐서 (num_patches, hidden_dim)
            processed_image: 전처리된 이미지
        """
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 전처리
        inputs = self.processor(images=image, return_tensors="pt")

        # 비전 인코더를 통과
        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs['pixel_values'],
                output_hidden_states=True
            )

        # 마지막 히든 스테이트 가져오기 (CLS 토큰 제외)
        # Shape: (batch_size, num_patches + 1, hidden_dim)
        hidden_states = vision_outputs.last_hidden_state

        # CLS 토큰 제외하고 패치 토큰만 가져오기
        patch_tokens = hidden_states[:, 1:, :]  # (1, num_patches, hidden_dim)

        # 전처리된 이미지 복원
        pixel_values = inputs['pixel_values'][0]  # (3, H, W)
        pixel_values = pixel_values.permute(1, 2, 0).numpy()

        # 정규화 해제
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        pixel_values = pixel_values * std + mean
        pixel_values = np.clip(pixel_values * 255, 0, 255).astype(np.uint8)
        processed_image = Image.fromarray(pixel_values)

        return patch_tokens[0], processed_image

    def visualize_token_grid(self, image_path: str, save_path: str = None,
                            show_indices: bool = True, alpha: float = 0.3):
        """
        이미지에 토큰 그리드를 오버레이하여 시각화합니다.

        Args:
            image_path: 입력 이미지 경로
            save_path: 저장할 경로 (None이면 화면에 표시)
            show_indices: 패치 인덱스를 표시할지 여부
            alpha: 그리드 라인의 투명도
        """
        # 토큰 추출
        tokens, processed_image = self.extract_vision_tokens(image_path)

        # 원본 이미지 크기로 리사이즈
        display_size = 800
        aspect_ratio = processed_image.width / processed_image.height
        if aspect_ratio > 1:
            new_width = display_size
            new_height = int(display_size / aspect_ratio)
        else:
            new_height = display_size
            new_width = int(display_size * aspect_ratio)

        img_display = processed_image.resize((new_width, new_height), Image.LANCZOS)

        # 그리드 오버레이 생성
        overlay = Image.new('RGBA', img_display.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # 패치 크기 계산
        patch_width = new_width / self.config['grid_size']
        patch_height = new_height / self.config['grid_size']

        # 그리드 선 그리기
        line_color = (255, 0, 0, int(255 * alpha))
        line_width = 2

        # 세로 선
        for i in range(self.config['grid_size'] + 1):
            x = i * patch_width
            draw.line([(x, 0), (x, new_height)], fill=line_color, width=line_width)

        # 가로 선
        for i in range(self.config['grid_size'] + 1):
            y = i * patch_height
            draw.line([(0, y), (new_width, y)], fill=line_color, width=line_width)

        # 패치 인덱스 표시
        if show_indices:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=12)
            except:
                font = ImageFont.load_default()

            for i in range(self.config['grid_size']):
                for j in range(self.config['grid_size']):
                    patch_idx = i * self.config['grid_size'] + j
                    x = j * patch_width + patch_width / 2
                    y = i * patch_height + patch_height / 2

                    # 텍스트 배경
                    text = str(patch_idx)
                    bbox = draw.textbbox((x, y), text, font=font)
                    draw.rectangle(bbox, fill=(0, 0, 0, 180))
                    draw.text((x, y), text, fill=(255, 255, 0, 255),
                             font=font, anchor="mm")

        # 이미지 합성
        img_rgba = img_display.convert('RGBA')
        result = Image.alpha_composite(img_rgba, overlay)
        result = result.convert('RGB')

        # 결과 표시 또는 저장
        if save_path:
            result.save(save_path)
            print(f"Saved visualization to {save_path}")
        else:
            plt.figure(figsize=(12, 12))
            plt.imshow(result)
            plt.axis('off')
            plt.title(f"{self.model_type}\n"
                     f"Grid: {self.config['grid_size']}x{self.config['grid_size']} "
                     f"({self.config['grid_size']**2} patches)")
            plt.tight_layout()
            plt.show()

        return result, tokens

    def visualize_token_attention(self, image_path: str, save_path: str = None,
                                  cmap: str = 'viridis'):
        """
        토큰의 L2 norm을 히트맵으로 시각화합니다.

        Args:
            image_path: 입력 이미지 경로
            save_path: 저장할 경로
            cmap: 컬러맵
        """
        tokens, processed_image = self.extract_vision_tokens(image_path)

        # L2 norm 계산
        token_norms = torch.norm(tokens, dim=1).numpy()

        # 그리드 모양으로 재배열
        grid_size = self.config['grid_size']
        attention_map = token_norms.reshape(grid_size, grid_size)

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # 원본 이미지
        axes[0].imshow(processed_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 어텐션 맵
        im = axes[1].imshow(attention_map, cmap=cmap, interpolation='nearest')
        axes[1].set_title(f'Token L2 Norm Heatmap\n({grid_size}x{grid_size})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention visualization to {save_path}")
        else:
            plt.show()

        return attention_map


def main():
    """사용 예시"""
    # 테스트용 샘플 이미지 생성
    sample_image_path = "sample_image.jpg"

    if not os.path.exists(sample_image_path):
        print("Creating sample image...")
        # 간단한 샘플 이미지 생성
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        draw = ImageDraw.Draw(img)

        # 원 그리기
        draw.ellipse([100, 100, 400, 400], fill=(255, 200, 0), outline=(255, 255, 255), width=5)
        draw.ellipse([200, 150, 250, 200], fill=(0, 0, 0))  # 왼쪽 눈
        draw.ellipse([350, 150, 400, 200], fill=(0, 0, 0))  # 오른쪽 눈
        draw.arc([150, 200, 450, 400], start=0, end=180, fill=(255, 255, 255), width=5)  # 미소

        img.save(sample_image_path)
        print(f"Sample image saved to {sample_image_path}")

    # 모델 1: clip-vit-large-patch14
    print("\n" + "="*50)
    print("Testing clip-vit-large-patch14")
    print("="*50)
    visualizer_224 = CLIPTokenVisualizer("clip-vit-large-patch14")
    visualizer_224.visualize_token_grid(
        sample_image_path,
        save_path="token_grid_224.png",
        show_indices=True,
        alpha=0.4
    )
    visualizer_224.visualize_token_attention(
        sample_image_path,
        save_path="token_attention_224.png"
    )

    # 모델 2: clip-vit-large-patch14-336
    print("\n" + "="*50)
    print("Testing clip-vit-large-patch14-336")
    print("="*50)
    visualizer_336 = CLIPTokenVisualizer("clip-vit-large-patch14-336")
    visualizer_336.visualize_token_grid(
        sample_image_path,
        save_path="token_grid_336.png",
        show_indices=True,
        alpha=0.4
    )
    visualizer_336.visualize_token_attention(
        sample_image_path,
        save_path="token_attention_336.png"
    )

    print("\n✓ All visualizations completed!")
    print("Generated files:")
    print("  - token_grid_224.png")
    print("  - token_attention_224.png")
    print("  - token_grid_336.png")
    print("  - token_attention_336.png")


if __name__ == "__main__":
    main()
