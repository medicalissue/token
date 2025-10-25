"""
토큰 유사도 분석 메인 스크립트 (Hydra 통합)
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
from token_similarity_analyzer import TokenSimilarityAnalyzer, SimilarityVisualizer, TokenClusterer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Hydra를 사용한 메인 실행 함수

    Args:
        cfg: Hydra configuration
    """
    print("="*80)
    print("Token Similarity Analysis")
    print("="*80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*80)

    # 출력 디렉토리 생성
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 설정 저장
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"\n✓ Saved configuration to {config_save_path}")

    # Analyzer 초기화
    print(f"\nInitializing TokenSimilarityAnalyzer...")
    analyzer = TokenSimilarityAnalyzer(
        model_type=cfg.model.type,
        device=cfg.device,
        similarity_metric=cfg.similarity.metric,
        remove_positional_embedding=cfg.similarity.remove_positional_embedding
    )

    # 분석 실행
    result = analyzer.analyze(cfg.image.path)

    # 유사도 데이터 저장
    if cfg.output.save_similarities:
        print("\n" + "="*80)
        print("Saving similarity data...")
        np.savez(
            output_dir / "similarities.npz",
            horizontal_right=result.horizontal_right,
            horizontal_left=result.horizontal_left,
            vertical_down=result.vertical_down,
            vertical_up=result.vertical_up,
            cls_similarity=result.cls_similarity,
            grid_size=result.grid_size,
            num_patches=result.num_patches
        )
        print(f"✓ Saved similarity data to {output_dir / 'similarities.npz'}")

        # 통계 정보 출력
        print("\nSimilarity Statistics:")
        print(f"  Horizontal (→) - Mean: {result.horizontal_right.mean():.4f}, "
              f"Std: {result.horizontal_right.std():.4f}, "
              f"Min: {result.horizontal_right.min():.4f}, "
              f"Max: {result.horizontal_right.max():.4f}")
        print(f"  Vertical (↓)   - Mean: {result.vertical_down.mean():.4f}, "
              f"Std: {result.vertical_down.std():.4f}, "
              f"Min: {result.vertical_down.min():.4f}, "
              f"Max: {result.vertical_down.max():.4f}")
        print(f"  CLS Token      - Mean: {result.cls_similarity.mean():.4f}, "
              f"Std: {result.cls_similarity.std():.4f}, "
              f"Min: {result.cls_similarity.min():.4f}, "
              f"Max: {result.cls_similarity.max():.4f}")

    # 클러스터링
    cluster_result = None
    if cfg.clustering.enabled:
        print("\n" + "="*80)
        print("Performing BFS clustering...")

        clusterer = TokenClusterer(
            threshold_mode=cfg.clustering.threshold_mode,
            threshold_value=cfg.clustering.threshold_value,
            use_cluster_mean=cfg.clustering.use_cluster_mean
        )

        cluster_result = clusterer.cluster_bfs(result)

        # 클러스터 데이터 저장
        if cfg.output.save_similarities:
            np.savez(
                output_dir / "clusters.npz",
                cluster_map=cluster_result.cluster_map,
                num_clusters=cluster_result.num_clusters,
                threshold=cluster_result.threshold
            )
            print(f"✓ Saved cluster data to {output_dir / 'clusters.npz'}")

    # 시각화
    if cfg.output.save_visualizations:
        print("\n" + "="*80)
        print("Creating visualizations...")

        visualizer = SimilarityVisualizer(
            figsize=tuple(cfg.visualization.figsize),
            cmap=cfg.visualization.cmap,
            dpi=cfg.output.dpi
        )

        # 통합 시각화
        if cfg.visualization.save_combined:
            combined_path = output_dir / f"similarity_analysis_combined.{cfg.output.format}"
            visualizer.visualize_all(result, save_path=str(combined_path))

        # 개별 시각화
        if cfg.visualization.save_individual:
            individual_dir = output_dir / "individual"
            visualizer.visualize_individual(result, output_dir=str(individual_dir))

        # 클러스터 시각화
        if cluster_result is not None:
            print("\nCreating cluster visualizations...")

            # 이미지 위에 클러스터 표시
            cluster_on_image_path = output_dir / f"clusters_on_image.{cfg.output.format}"
            visualizer.visualize_clusters_on_image(
                cluster_result,
                cfg.image.path,
                analyzer,  # analyzer 전달
                save_path=str(cluster_on_image_path),
                line_width=cfg.visualization.cluster_line_width
            )

            # 클러스터 맵
            cluster_map_path = output_dir / f"cluster_map.{cfg.output.format}"
            visualizer.visualize_cluster_map(cluster_result, save_path=str(cluster_map_path))

    print("\n" + "="*80)
    print("✓ Analysis Complete!")
    print(f"✓ All outputs saved to: {output_dir.absolute()}")
    if cluster_result is not None:
        print(f"✓ Found {cluster_result.num_clusters} clusters with threshold {cluster_result.threshold:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
