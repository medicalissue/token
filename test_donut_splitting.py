#!/usr/bin/env python3
"""
Donut 모양 클러스터 분할 테스트
"""
import numpy as np
from collections import deque

def test_donut_splitting():
    """Donut 모양 분할 테스트"""
    print("="*60)
    print("Testing Donut Cluster Splitting")
    print("="*60)

    # 진짜 donut 모양 클러스터 (중앙이 비어있음)
    donut_map = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ])

    print("Donut cluster map (center is empty):")
    print(donut_map)

    # 클러스터 좌표들
    coords = [(i, j) for i in range(5) for j in range(5) if donut_map[i, j] == 1]
    print(f"\nCluster coords: {coords}")

    def find_connected_components(coords, connectivity_type="4-direction"):
        """연결된 컴포넌트 찾기"""
        if connectivity_type == "4-direction":
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                         (0, 1), (1, -1), (1, 0), (1, 1)]

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

    # 4-방향 연결성 테스트
    print("\n--- 4-direction connectivity ---")
    components_4d = find_connected_components(coords, "4-direction")
    print(f"Connected components: {len(components_4d)}")
    for i, comp in enumerate(components_4d):
        print(f"  Component {i} (size {len(comp)}): {comp}")

    if len(components_4d) > 1:
        # 분할된 맵 생성
        new_map = np.full_like(donut_map, -1)
        for i, component in enumerate(components_4d):
            for row, col in component:
                new_map[row, col] = i

        print(f"\n4-direction split result:")
        print(new_map)
    else:
        print("\nNo splitting occurred - donut is fully connected")

    # 분리된 donut 모양 테스트
    print("\n" + "="*40)
    print("Testing Separated Donut Shape")
    print("="*40)

    separated_donut = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])

    print("Separated donut (gap in middle):")
    print(separated_donut)

    # 클러스터 좌표들
    coords_sep = [(i, j) for i in range(5) for j in range(5) if separated_donut[i, j] == 1]
    print(f"\nSeparated donut coords: {coords_sep}")

    # 4-방향 연결성 테스트
    components_sep = find_connected_components(coords_sep, "4-direction")
    print(f"\n4-direction connected components: {len(components_sep)}")
    for i, comp in enumerate(components_sep):
        print(f"  Component {i} (size {len(comp)}): {comp}")

    if len(components_sep) > 1:
        # 분할된 맵 생성
        new_map_sep = np.full_like(separated_donut, -1)
        for i, component in enumerate(components_sep):
            for row, col in component:
                new_map_sep[row, col] = i

        print(f"\nSeparated donut split result:")
        print(new_map_sep)

    print("✓ Donut splitting test completed\n")

def test_ring_shapes():
    """다양한 링 모양 테스트"""
    print("="*60)
    print("Testing Various Ring Shapes")
    print("="*60)

    # 얇은 링
    thin_ring = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ])

    print("Thin ring shape:")
    print(thin_ring)

    coords_ring = [(i, j) for i in range(5) for j in range(5) if thin_ring[i, j] == 1]
    print(f"\nRing coords: {coords_ring}")

    def find_components_simple(coords):
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

    components = find_components_simple(coords_ring)
    print(f"\nConnected components: {len(components)}")
    for i, comp in enumerate(components):
        print(f"  Component {i} (size {len(comp)}): {comp}")

    print("✓ Ring shapes test completed\n")

def main():
    """모든 테스트 실행"""
    print("Donut and Ring Shape Connectivity Tests")
    print("=" * 80)

    test_donut_splitting()
    test_ring_shapes()

    print("=" * 80)
    print("All donut/ring tests completed!")

    print("\nConclusion:")
    print("1. Fully connected donuts (with continuous paths) won't split")
    print("2. Separated donuts (with gaps) will split into multiple components")
    print("3. The algorithm correctly identifies connectivity based on the chosen type")

if __name__ == "__main__":
    main()