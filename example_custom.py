"""
커스텀 사용 예시 스크립트
"""
from visualize_clip_tokens import CLIPTokenVisualizer
from PIL import Image
import sys


def visualize_with_model(image_path: str, model_type: str = "clip-vit-large-patch14"):
    """
    지정된 이미지를 특정 CLIP 모델로 시각화합니다.

    Args:
        image_path: 이미지 파일 경로
        model_type: "clip-vit-large-patch14" 또는 "clip-vit-large-patch14-336"
    """
    print(f"\n🎨 Visualizing with {model_type}")
    print("="*60)

    # Visualizer 초기화
    visualizer = CLIPTokenVisualizer(model_type)

    # 출력 파일명 생성
    base_name = image_path.rsplit('.', 1)[0]
    model_suffix = "224" if "336" not in model_type else "336"

    # 1. 토큰 그리드 시각화
    print("\n📊 Creating token grid visualization...")
    grid_output = f"{base_name}_grid_{model_suffix}.png"
    visualizer.visualize_token_grid(
        image_path,
        save_path=grid_output,
        show_indices=True,
        alpha=0.5
    )

    # 2. 어텐션 히트맵
    print("\n🔥 Creating attention heatmap...")
    attention_output = f"{base_name}_attention_{model_suffix}.png"
    visualizer.visualize_token_attention(
        image_path,
        save_path=attention_output,
        cmap='hot'
    )

    print(f"\n✓ Done! Generated files:")
    print(f"  - {grid_output}")
    print(f"  - {attention_output}")


def compare_models(image_path: str):
    """
    두 CLIP 모델의 결과를 비교합니다.

    Args:
        image_path: 이미지 파일 경로
    """
    print("\n🔍 Comparing both CLIP models...")
    print("="*60)

    # 224 모델
    visualize_with_model(image_path, "clip-vit-large-patch14")

    print("\n" + "-"*60 + "\n")

    # 336 모델
    visualize_with_model(image_path, "clip-vit-large-patch14-336")

    print("\n" + "="*60)
    print("✓ Comparison complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python example_custom.py <image_path> [model_type]")
        print()
        print("Examples:")
        print("  python example_custom.py my_image.jpg")
        print("  python example_custom.py my_image.jpg clip-vit-large-patch14")
        print("  python example_custom.py my_image.jpg clip-vit-large-patch14-336")
        print("  python example_custom.py my_image.jpg compare")
        print()
        print("Model types:")
        print("  - clip-vit-large-patch14 (224x224)")
        print("  - clip-vit-large-patch14-336 (336x336)")
        print("  - compare (both models)")
        sys.exit(1)

    image_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "clip-vit-large-patch14"

    # 이미지 존재 확인
    try:
        Image.open(image_path)
    except Exception as e:
        print(f"❌ Error: Cannot open image '{image_path}'")
        print(f"   {e}")
        sys.exit(1)

    # 실행
    if model_type == "compare":
        compare_models(image_path)
    else:
        visualize_with_model(image_path, model_type)
