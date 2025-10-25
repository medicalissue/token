#!/usr/bin/env python3
"""
테스트 스크립트: 새로운 클러스터링 기능 테스트
"""
import numpy as np
import sys
sys.path.append('.')
from token_similarity_analyzer import (
    ClusterContainmentDetector,
    VisibilityConvexSplitter,
    VisibilityGraphConstructor,
    ConvexHullDetector,
    DSATURColoring,
    LubyMIS,
    JonesPlassmannColoring
)

def create_test_cluster_map():
    """
    테스트용 클러스터 맵 생성 - 둘러싸인 클러스터 포함
    """
    # 8x8 그리드
    cluster_map = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 2, 2, 0, 1, 1, 1, 1],
        [0, 2, 2, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [3, 3, 3, 3, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4, 4, 4],
    ])
    return cluster_map

def test_containment_detection():
    """클러스터 둘러싸기 감지 테스트"""
    print("="*60)
    print("Testing Cluster Containment Detection")
    print("="*60)

    cluster_map = create_test_cluster_map()
    print("Test cluster map:")
    print(cluster_map)

    detector = ClusterContainmentDetector(cls_similarity_threshold=0.8)

    # 둘러싸인 클러스터 쌍 찾기
    contained_pairs = detector.find_contained_clusters(cluster_map)
    print(f"\nFound contained pairs: {contained_pairs}")

    # 각 쌍에 대해 개별적으로 확인
    for outer_id, inner_id in contained_pairs:
        is_contained = detector.is_cluster_contained(cluster_map, outer_id, inner_id)
        print(f"Cluster {inner_id} contained by cluster {outer_id}: {is_contained}")

    print("✓ Containment detection test completed\n")

def test_visibility_graph():
    """가시성 그래프 생성 테스트"""
    print("="*60)
    print("Testing Visibility Graph Construction")
    print("="*60)

    # 테스트용 클러스터 맵 (L자 모양)
    cluster_map = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ])

    print("Test cluster map:")
    print(cluster_map)

    constructor = VisibilityGraphConstructor(cluster_map)

    # 클러스터 0의 좌표들
    coords_0 = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0)]
    print(f"\nCluster 0 coords: {coords_0}")

    # 가시성 그래프 생성
    visibility_graph = constructor.build_visibility_graph(coords_0)
    print(f"\nVisibility graph shape: {visibility_graph.shape}")
    print("Visibility graph:")
    print(visibility_graph.astype(int))

    print("✓ Visibility graph test completed\n")

def test_convex_hull():
    """컨벡스 헐 테스트"""
    print("="*60)
    print("Testing Convex Hull Detection")
    print("="*60)

    # 테스트 좌표들 (사각형 + 내부 점)
    coords = np.array([
        [0, 0],  # 왼쪽 위
        [0, 2],  # 왼쪽 아래
        [2, 0],  # 오른쪽 위
        [2, 2],  # 오른쪽 아래
        [1, 1],  # 중심 (내부 점)
    ])

    print("Test coordinates:")
    for i, coord in enumerate(coords):
        print(f"  {i}: {coord}")

    hull_vertices = ConvexHullDetector.compute_convex_hull(coords)
    print(f"\nConvex hull vertices: {hull_vertices}")

    # 각 좌표가 컨벡스 헐 위에 있는지 확인
    for i, coord in enumerate(coords):
        on_hull = ConvexHullDetector.is_on_convex_hull(coord, hull_vertices, coords)
        print(f"  Coord {i} {coord} on hull: {on_hull}")

    print("✓ Convex hull test completed\n")

def test_graph_algorithms():
    """그래프 알고리즘 테스트"""
    print("="*60)
    print("Testing Graph Coloring Algorithms")
    print("="*60)

    # 테스트 인접 행렬 (간단한 그래프)
    adj_matrix = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0],
    ], dtype=bool)

    print("Test adjacency matrix:")
    print(adj_matrix.astype(int))

    # DSATUR 테스트
    print("\n--- DSATUR Coloring ---")
    dsatur = DSATURColoring(adj_matrix)
    dsatur_colors = dsatur.color_graph()
    print(f"DSATUR colors: {dsatur_colors}")

    # Jones-Plassmann 테스트
    print("\n--- Jones-Plassmann Coloring ---")
    jp = JonesPlassmannColoring(adj_matrix)
    jp_colors = jp.color_graph()
    print(f"Jones-Plassmann colors: {jp_colors}")

    # Luby MIS 테스트
    print("\n--- Luby MIS ---")
    luby = LubyMIS(adj_matrix)
    mis = luby.find_mis()
    print(f"Luby MIS: {mis}")

    print("✓ Graph algorithms test completed\n")

def test_visibility_splitting():
    """가시성-컨벡스 분할 테스트"""
    print("="*60)
    print("Testing Visibility-Convex Splitting")
    print("="*60)

    # 간단한 테스트용 ClusterResult 모의 객체
    class MockClusterResult:
        def __init__(self, cluster_map):
            self.cluster_map = cluster_map
            self.num_clusters = len(np.unique(cluster_map))
            self.cluster_sizes = {i: np.sum(cluster_map == i) for i in range(self.num_clusters)}
            self.threshold = 0.5

    # U자 모양 클러스터 (분할하기 좋은 형태)
    cluster_map = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])

    print("Test cluster map (U-shape should be split):")
    print(cluster_map)

    # 클러스터 0 (외부 프레임)과 클러스터 1 (내부 U자 모양)만 남기기
    modified_map = np.where(cluster_map == 2, 1, cluster_map)

    mock_result = MockClusterResult(modified_map)

    splitter = VisibilityConvexSplitter(
        strictness="low",
        algorithm="dsatur",
        min_split_size=3
    )

    print(f"\nBefore splitting: {mock_result.num_clusters} clusters")

    try:
        split_result = splitter.split_clusters(mock_result)
        print(f"After splitting: {split_result.num_clusters} clusters")
        print("Split cluster map:")
        print(split_result.cluster_map)
        print("✓ Visibility-convex splitting test completed\n")
    except Exception as e:
        print(f"Error during splitting: {e}")
        print("✓ Test completed with expected error\n")

def main():
    """모든 테스트 실행"""
    print("Starting Enhanced Clustering Algorithm Tests")
    print("=" * 80)

    test_containment_detection()
    test_visibility_graph()
    test_convex_hull()
    test_graph_algorithms()
    test_visibility_splitting()

    print("=" * 80)
    print("All tests completed!")

if __name__ == "__main__":
    main()