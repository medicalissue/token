#!/usr/bin/env python3
"""
Convex Hull 기반 클러스터 분할 알고리즘 테스트
"""

import numpy as np

def test_convex_hull_splitting():
    """Convex Hull 분할 알고리즘 테스트"""

    print("="*60)
    print("Convex Hull 기반 클러스터 분할 테스트")
    print("="*60)

    # 테스트 케이스 1: 도넛 모양 (구멍 있음)
    print("\n테스트 1: 도넛 모양 클러스터")
    donut_coords = [
        (0,0), (0,1), (0,2), (0,3), (0,4),
        (1,0), (1,4),
        (2,0), (2,4),
        (3,0), (3,4),
        (4,0), (4,1), (4,2), (4,3), (4,4)
    ]

    # 구멍 부분 제외
    # (1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3) are missing

    print(f"도넛 모양 좌표 수: {len(donut_coords)}")

    # Bounding box 면적: 5x5 = 25
    # 실제 토큰 수: 16
    # 채우기 비율: 16/25 = 0.64

    # 간단한 Convex Hull 면적 계산 (bounding box로 근사)
    coords_array = np.array(donut_coords)
    min_coords = coords_array.min(axis=0)
    max_coords = coords_array.max(axis=0)
    hull_area = float(np.prod(max_coords - min_coords + 1))
    actual_size = len(donut_coords)
    fill_ratio = actual_size / hull_area

    print(f"  Bounding box 면적: {hull_area}")
    print(f"  실제 크기: {actual_size}")
    print(f"  채우기 비율: {fill_ratio:.3f}")

    # 다양한 threshold로 테스트
    thresholds = [0.5, 0.7, 0.9]
    for threshold in thresholds:
        should_split = fill_ratio < threshold
        print(f"  Threshold {threshold}: {'분할' if should_split else '분할 안함'}")

    print("\n테스트 2: 단단한 사각형 (구멍 없음)")
    solid_coords = [
        (0,0), (0,1), (0,2),
        (1,0), (1,1), (1,2),
        (2,0), (2,1), (2,2)
    ]

    coords_array = np.array(solid_coords)
    min_coords = coords_array.min(axis=0)
    max_coords = coords_array.max(axis=0)
    hull_area = float(np.prod(max_coords - min_coords + 1))
    actual_size = len(solid_coords)
    fill_ratio = actual_size / hull_area

    print(f"  사각형 좌표 수: {actual_size}")
    print(f"  Bounding box 면적: {hull_area}")
    print(f"  채우기 비율: {fill_ratio:.3f}")

    for threshold in thresholds:
        should_split = fill_ratio < threshold
        print(f"  Threshold {threshold}: {'분할' if should_split else '분할 안함'}")

    print("\n테스트 3: 오목한 L자 모양")
    l_shape_coords = [
        (0,0), (0,1), (0,2), (0,3), (0,4),
        (1,0),
        (2,0),
        (3,0),
        (4,0)
    ]

    coords_array = np.array(l_shape_coords)
    min_coords = coords_array.min(axis=0)
    max_coords = coords_array.max(axis=0)
    hull_area = float(np.prod(max_coords - min_coords + 1))
    actual_size = len(l_shape_coords)
    fill_ratio = actual_size / hull_area

    print(f"  L자 모양 좌표 수: {actual_size}")
    print(f"  Bounding box 면적: {hull_area}")
    print(f"  채우기 비율: {fill_ratio:.3f}")

    for threshold in thresholds:
        should_split = fill_ratio < threshold
        print(f"  Threshold {threshold}: {'분할' if should_split else '분할 안함'}")

    print("\n분할 방식 설명:")
    print("- 낮은 threshold (0.5): 매우 민감, 대부분의 비직사각형 모양 분할")
    print("- 중간 threshold (0.7): 도넛이나 심하게 오목한 모양 분할")
    print("- 높은 threshold (0.9): 거의 빈 클러스터만 분할")
    print("\n분할 전략:")
    print("1. 수직 분할 (좌우) 또는 수평 분할 (상하) 시도")
    print("2. 더 균등하게 나누는 방향 선택")
    print("3. 실패 시 대각선 또는 PCA 기반 분할 시도")

if __name__ == "__main__":
    test_convex_hull_splitting()