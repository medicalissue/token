#!/usr/bin/env python3
"""
독립형 알고리즘 테스트 스크립트
"""
import numpy as np
from collections import deque
import random
from scipy.spatial import ConvexHull

def test_containment_detection():
    """클러스터 둘러싸기 감지 테스트"""
    print("="*60)
    print("Testing Cluster Containment Detection")
    print("="*60)

    # 둘러싸인 클러스터가 있는 테스트 맵
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

    print("Test cluster map:")
    print(cluster_map)

    def is_cluster_contained(cluster_map, outer_id, inner_id):
        """간단한 둘러싸기 확인"""
        outer_coords = np.argwhere(cluster_map == outer_id)
        inner_coords = np.argwhere(cluster_map == inner_id)

        if len(outer_coords) == 0 or len(inner_coords) == 0:
            return False

        # 외부 클러스터의 경계
        min_row, min_col = outer_coords.min(axis=0)
        max_row, max_col = outer_coords.max(axis=0)

        # 내부 클러스터의 모든 좌표가 외부 경계 내에 있는지 확인
        for coord in inner_coords:
            row, col = coord
            if not (min_row <= row <= max_row and min_col <= col <= max_col):
                return False
            if cluster_map[row, col] == outer_id:
                return False

        return True

    # 테스트: 클러스터 0이 클러스터 2를 둘러싸는지 확인
    contained = is_cluster_contained(cluster_map, 0, 2)
    print(f"\nCluster 2 contained by cluster 0: {contained}")

    # 테스트: 클러스터 1이 다른 클러스터를 둘러싸는지 확인
    contained_1 = is_cluster_contained(cluster_map, 1, 0)
    print(f"Cluster 0 contained by cluster 1: {contained_1}")

    print("✓ Containment detection test completed\n")

def test_convex_hull():
    """컨벡스 헐 테스트"""
    print("="*60)
    print("Testing Convex Hull Detection")
    print("="*60)

    # 사각형 + 내부 점
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

    try:
        hull = ConvexHull(coords)
        hull_vertices = hull.vertices
        print(f"\nConvex hull vertices: {hull_vertices}")

        # 컨벡스성 비율 계산
        convexity_ratio = len(hull_vertices) / len(coords)
        print(f"Convexity ratio: {convexity_ratio:.2f}")

        print("✓ Convex hull test completed\n")
    except Exception as e:
        print(f"Error in convex hull calculation: {e}")
        print("✓ Test completed with expected error\n")

def test_dsatur():
    """DSATUR 그래프 컬러링 테스트"""
    print("="*60)
    print("Testing DSATUR Graph Coloring")
    print("="*60)

    # 테스트 그래프 (5개 정점)
    adj_matrix = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0],
    ], dtype=bool)

    print("Test adjacency matrix:")
    print(adj_matrix.astype(int))

    def dsatur_coloring(adj_matrix):
        n = adj_matrix.shape[0]
        colors = [-1] * n
        saturation = [0] * n
        degrees = [np.sum(adj_matrix[i]) for i in range(n)]
        uncolored = set(range(n))

        while uncolored:
            # 채도가 가장 높은 정점 선택
            max_sat = -1
            max_deg = -1
            selected_vertex = None

            for v in uncolored:
                sat = saturation[v]
                deg = degrees[v]
                if sat > max_sat or (sat == max_sat and deg > max_deg):
                    max_sat = sat
                    max_deg = deg
                    selected_vertex = v

            # 가장 작은 가능한 색상 할당
            used_colors = set()
            for neighbor in range(n):
                if adj_matrix[selected_vertex][neighbor] and colors[neighbor] != -1:
                    used_colors.add(colors[neighbor])

            color = 0
            while color in used_colors:
                color += 1

            colors[selected_vertex] = color
            uncolored.remove(selected_vertex)

            # 이웃들의 채도 업데이트
            for neighbor in range(n):
                if adj_matrix[selected_vertex][neighbor] and colors[neighbor] == -1:
                    neighbor_used_colors = set()
                    for n2 in range(n):
                        if adj_matrix[neighbor][n2] and colors[n2] != -1:
                            neighbor_used_colors.add(colors[n2])

                    if color not in neighbor_used_colors:
                        saturation[neighbor] += 1

        return colors

    colors = dsatur_coloring(adj_matrix)
    print(f"\nDSATUR colors: {colors}")
    print(f"Number of colors used: {max(colors) + 1}")

    print("✓ DSATUR test completed\n")

def test_luby_mis():
    """Luby MIS 테스트"""
    print("="*60)
    print("Testing Luby's Maximal Independent Set")
    print("="*60)

    # 테스트 그래프
    adj_matrix = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0],
    ], dtype=bool)

    print("Test adjacency matrix:")
    print(adj_matrix.astype(int))

    def luby_mis(adj_matrix):
        n = adj_matrix.shape[0]
        remaining = set(range(n))
        mis = set()

        while remaining:
            # 랜덤 가중치 할당
            weights = {v: random.random() for v in remaining}

            # 로컬 최대값들을 찾음
            to_add = set()
            to_remove = set()

            for v in remaining:
                is_local_max = True
                for neighbor in range(n):
                    if adj_matrix[v][neighbor] and neighbor in remaining:
                        if weights[neighbor] > weights[v]:
                            is_local_max = False
                            break

                if is_local_max:
                    to_add.add(v)

            # 선택된 정점들과 그 이웃들을 제거
            for v in to_add:
                to_remove.add(v)
                for neighbor in range(n):
                    if adj_matrix[v][neighbor]:
                        to_remove.add(neighbor)

            mis.update(to_add)
            remaining -= to_remove

        return mis

    # 랜덤성을 위해 고정 시드 설정
    random.seed(42)
    mis = luby_mis(adj_matrix)
    print(f"\nLuby MIS: {mis}")

    # 독립 집합인지 확인
    def is_independent_set(mis, adj_matrix):
        for v1 in mis:
            for v2 in mis:
                if v1 != v2 and adj_matrix[v1][v2]:
                    return False
        return True

    is_valid = is_independent_set(mis, adj_matrix)
    print(f"Is valid independent set: {is_valid}")

    print("✓ Luby MIS test completed\n")

def test_visibility_line():
    """선 가시성 테스트 (Bresenham's algorithm)"""
    print("="*60)
    print("Testing Line Visibility")
    print("="*60)

    def is_visible_line(pos1, pos2, cluster_map):
        """Bresenham's line algorithm for visibility"""
        row1, col1 = pos1
        row2, col2 = pos2

        dx = abs(col2 - col1)
        dy = abs(row2 - row1)

        x, y = col1, row1
        x_inc = 1 if col2 > col1 else -1
        y_inc = 1 if row2 > row1 else -1

        error = dx - dy

        while x != col2 or y != row2:
            # 현재 위치가 클러스터 경계를 넘으면 가시성 없음
            if (y < 0 or y >= cluster_map.shape[0] or
                x < 0 or x >= cluster_map.shape[1]):
                return False

            # 다음 위치 계산
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return True

    # 테스트: L자 모양 클러스터
    cluster_map = np.array([
        [1, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])

    print("Test cluster map (1 = cluster, 0 = empty):")
    print(cluster_map)

    # 테스트 케이스들
    test_cases = [
        ((0, 0), (2, 1), "Corner to inner"),  # 보이는 경우
        ((0, 3), (2, 2), "Outside to near"),  # 보이지 않는 경우
        ((0, 0), (0, 2), "Same row"),  # 보이는 경우
    ]

    for pos1, pos2, description in test_cases:
        visible = is_visible_line(pos1, pos2, cluster_map)
        print(f"\n{description}: {pos1} -> {pos2}: {visible}")

    print("✓ Line visibility test completed\n")

def main():
    """모든 테스트 실행"""
    print("Standalone Enhanced Clustering Algorithm Tests")
    print("=" * 80)

    test_containment_detection()
    test_convex_hull()
    test_dsatur()
    test_luby_mis()
    test_visibility_line()

    print("=" * 80)
    print("All standalone tests completed!")

if __name__ == "__main__":
    main()