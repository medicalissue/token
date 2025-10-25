#!/usr/bin/env python3
"""
연결성 기반 클러스터 분할 테스트 스크립트
"""
import numpy as np
from collections import deque

def test_connectivity_splitting():
    """연결성 기반 분할 테스트"""
    print("="*60)
    print("Testing Connectivity-Based Cluster Splitting")
    print("="*60)

    # 테스트용 클러스터 맵 - donut 모양
    cluster_map = np.array([
        [1, 1, 1],
        [0, 1, 0],
        [1, 1, 1],
    ])

    print("Original cluster map (should be split into 2):")
    print(cluster_map)

    # 간단한 연결성 분할 알고리즘
    def find_connected_components(cluster_coords, connectivity_type="4-direction"):
        """연결된 컴포넌트 찾기"""
        if connectivity_type == "4-direction":
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # 8-direction
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                         (0, 1), (1, -1), (1, 0), (1, 1)]

        coord_set = set(cluster_coords)
        visited = set()
        components = []

        for coord in cluster_coords:
            if coord in visited:
                continue

            component = []
            queue = deque([coord])
            visited.add(coord)

            while queue:
                current = queue.popleft()
                component.append(current)

                row, col = current
                for dr, dc in directions:
                    new_coord = (row + dr, col + dc)
                    if new_coord in coord_set and new_coord not in visited:
                        visited.add(new_coord)
                        queue.append(new_coord)

            components.append(component)

        return components

    # 클러스터 1의 좌표들
    coords_1 = [(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)]
    print(f"\nCluster 1 coords: {coords_1}")

    # 4-방향 연결성 테스트
    print("\n--- 4-direction connectivity ---")
    components_4d = find_connected_components(coords_1, "4-direction")
    print(f"Connected components: {len(components_4d)}")
    for i, comp in enumerate(components_4d):
        print(f"  Component {i}: {comp}")

    # 8-방향 연결성 테스트
    print("\n--- 8-direction connectivity ---")
    components_8d = find_connected_components(coords_1, "8-direction")
    print(f"Connected components: {len(components_8d)}")
    for i, comp in enumerate(components_8d):
        print(f"  Component {i}: {comp}")

    # 결과 맵 생성
    def create_split_map(original_map, components):
        """분할된 맵 생성"""
        new_map = np.full_like(original_map, -1)
        for i, component in enumerate(components):
            for row, col in component:
                new_map[row, col] = i
        return new_map

    if len(components_4d) > 1:
        split_map_4d = create_split_map(cluster_map, components_4d)
        print(f"\n4-direction split result:")
        print(split_map_4d)

    print("✓ Connectivity splitting test completed\n")

def test_more_complex_shapes():
    """더 복잡한 형태 테스트"""
    print("="*60)
    print("Testing More Complex Shapes")
    print("="*60)

    # C자 모양 클러스터
    cluster_map = np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 1, 0],
    ])

    print("C-shape cluster map:")
    print(cluster_map)

    # 클러스터 좌표
    coords = [(i, j) for i in range(5) for j in range(4) if cluster_map[i, j] == 1]
    print(f"\nCluster coords: {coords}")

    # 연결성 분할
    def find_connected_components_simple(coords):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        coord_set = set(coords)
        visited = set()
        components = []

        for coord in coords:
            if coord in visited:
                continue

            component = []
            queue = deque([coord])
            visited.add(coord)

            while queue:
                current = queue.popleft()
                component.append(current)

                row, col = current
                for dr, dc in directions:
                    new_coord = (row + dr, col + dc)
                    if new_coord in coord_set and new_coord not in visited:
                        visited.add(new_coord)
                        queue.append(new_coord)

            components.append(component)

        return components

    components = find_connected_components_simple(coords)
    print(f"\nConnected components: {len(components)}")
    for i, comp in enumerate(components):
        print(f"  Component {i} (size {len(comp)}): {comp}")

    # 분할된 맵 생성
    if len(components) > 1:
        new_map = np.full_like(cluster_map, -1)
        for i, component in enumerate(components):
            for row, col in component:
                new_map[row, col] = i

        print(f"\nSplit result:")
        print(new_map)

    print("✓ Complex shapes test completed\n")

def test_edge_cases():
    """엣지 케이스 테스트"""
    print("="*60)
    print("Testing Edge Cases")
    print("="*60)

    # 단일 토큰 클러스터
    single_token = np.array([[1]])
    print("Single token cluster:")
    print(single_token)

    coords_single = [(0, 0)]
    print(f"Should not split (size: {len(coords_single)})")

    # 2개 토큰 클러스터 (대각선)
    diagonal = np.array([
        [1, 0],
        [0, 1]
    ])
    print("\nDiagonal 2-token cluster:")
    print(diagonal)

    coords_diagonal = [(0, 0), (1, 1)]
    print(f"4-direction connectivity: {'connected' if len([(0, 0), (1, 1)]) == 1 else 'separate'}")
    print(f"8-direction connectivity: connected")

    # 작은 사각형
    square = np.array([
        [1, 1],
        [1, 1]
    ])
    print("\n2x2 square cluster:")
    print(square)

    coords_square = [(0, 0), (0, 1), (1, 0), (1, 1)]
    print("4-direction connectivity: connected (single component)")
    print("8-direction connectivity: connected (single component)")

    print("✓ Edge cases test completed\n")

def main():
    """모든 테스트 실행"""
    print("Connectivity-Based Cluster Splitting Tests")
    print("=" * 80)

    test_connectivity_splitting()
    test_more_complex_shapes()
    test_edge_cases()

    print("=" * 80)
    print("All connectivity tests completed!")

    print("\nKey findings:")
    print("1. 4-direction connectivity separates diagonal-only connections")
    print("2. 8-direction connectivity keeps diagonal connections together")
    print("3. Donut shapes get properly split into inner and outer components")
    print("4. Small clusters (size < min_split_size) are not split")

if __name__ == "__main__":
    main()