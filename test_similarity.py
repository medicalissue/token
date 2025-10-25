"""
토큰 유사도 분석 테스트 스크립트
Hydra 없이 직접 실행 가능
"""
from token_similarity_analyzer import TokenSimilarityAnalyzer, SimilarityVisualizer
from PIL import Image, ImageDraw
from pathlib import Path


def create_test_image(save_path: str = "test_image.jpg"):
    """테스트용 샘플 이미지 생성"""
    print(f"Creating test image: {save_path}")

    # 512x512 이미지 생성
    img = Image.new('RGB', (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 여러 패턴 그리기
    # 1. 원형 (중앙)
    draw.ellipse([150, 150, 350, 350], fill=(255, 100, 100), outline=(0, 0, 0), width=3)

    # 2. 사각형 (좌상단)
    draw.rectangle([20, 20, 120, 120], fill=(100, 100, 255), outline=(0, 0, 0), width=3)

    # 3. 삼각형 (우상단)
    draw.polygon([(400, 20), (490, 120), (310, 120)], fill=(100, 255, 100), outline=(0, 0, 0), width=3)

    # 4. 줄무늬 (하단)
    for i in range(10):
        x = 50 + i * 40
        draw.rectangle([x, 400, x + 20, 490], fill=(200, 200, 0))

    img.save(save_path)
    print(f"✓ Test image saved to {save_path}")
    return save_path


def test_similarity_analysis():
    """유사도 분석 테스트"""
    print("\n" + "="*80)
    print("Token Similarity Analysis Test")
    print("="*80)

    # 테스트 이미지 생성
    image_path = create_test_image()

    # 출력 디렉토리 생성
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    # Analyzer 초기화
    print("\n" + "-"*80)
    print("Initializing TokenSimilarityAnalyzer...")
    print("-"*80)
    analyzer = TokenSimilarityAnalyzer(
        model_type="clip-vit-large-patch14",
        device=None,  # 자동 선택
        similarity_metric="cosine"
    )

    # 분석 실행
    print("\n" + "-"*80)
    print("Running analysis...")
    print("-"*80)
    result = analyzer.analyze(image_path)

    # 결과 출력
    print("\n" + "-"*80)
    print("Analysis Results:")
    print("-"*80)
    print(f"Grid size: {result.grid_size}x{result.grid_size}")
    print(f"Total patches: {result.num_patches}")
    print(f"\nHorizontal similarity shape: {result.horizontal_right.shape}")
    print(f"Vertical similarity shape: {result.vertical_down.shape}")
    print(f"CLS similarity shape: {result.cls_similarity.shape}")

    print(f"\nHorizontal similarity - Mean: {result.horizontal_right.mean():.4f}, "
          f"Std: {result.horizontal_right.std():.4f}")
    print(f"Vertical similarity   - Mean: {result.vertical_down.mean():.4f}, "
          f"Std: {result.vertical_down.std():.4f}")
    print(f"CLS similarity        - Mean: {result.cls_similarity.mean():.4f}, "
          f"Std: {result.cls_similarity.std():.4f}")

    # 시각화
    print("\n" + "-"*80)
    print("Creating visualizations...")
    print("-"*80)
    visualizer = SimilarityVisualizer(figsize=(20, 16), cmap="viridis", dpi=150)

    # 통합 시각화
    combined_path = output_dir / "test_combined.png"
    visualizer.visualize_all(result, save_path=str(combined_path))

    # 개별 시각화
    individual_dir = output_dir / "individual"
    visualizer.visualize_individual(result, output_dir=str(individual_dir))

    # 유사도 데이터 저장
    import numpy as np
    np.savez(
        output_dir / "test_similarities.npz",
        horizontal_right=result.horizontal_right,
        vertical_down=result.vertical_down,
        cls_similarity=result.cls_similarity,
        grid_size=result.grid_size
    )
    print(f"✓ Saved similarity data to {output_dir / 'test_similarities.npz'}")

    print("\n" + "="*80)
    print("✓ Test Complete!")
    print(f"✓ All outputs saved to: {output_dir.absolute()}")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {combined_path}")
    print(f"  - {individual_dir}/horizontal_similarity.png")
    print(f"  - {individual_dir}/vertical_similarity.png")
    print(f"  - {individual_dir}/cls_similarity.png")
    print(f"  - {output_dir}/test_similarities.npz")


if __name__ == "__main__":
    test_similarity_analysis()
