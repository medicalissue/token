#!/usr/bin/env python3
"""
수정된 Percentile 로직 방향성 테스트
"""

import numpy as np

def test_fixed_percentile_direction():
    """수정된 percentile 로직: 낮을수록 덜 병합되는지 테스트"""

    print("="*60)
    print("수정된 Percentile 방향성 테스트")
    print("="*60)

    # 테스트 데이터: CLS 유사도
    # 높은 유사도 그룹: [0.9, 0.89, 0.91] (서로 비슷해서 병합되어야 함)
    # 낮은 유사도 그룹: [0.4, 0.42, 0.38] (서로 비슷해서 병합되어야 함)
    # 중간 유사도: [0.65] (다른 그룹과는 병합되지 않아야 함)
    cls_similarities = {
        0: 0.90,  # 높은 그룹
        1: 0.89,  # 높은 그룹
        2: 0.91,  # 높은 그룹
        3: 0.65,  # 중간
        4: 0.40,  # 낮은 그룹
        5: 0.42,  # 낮은 그룹
        6: 0.38   # 낮은 그룹
    }

    print(f"CLS similarities: {cls_similarities}")
    print(f"Expected: High group (0,1,2) merge, Low group (4,5,6) merge, Middle (3) separate")
    print()

    # 모든 클러스터 쌍의 CLS 유사도 차이 계산
    sim_values = list(cls_similarities.values())
    diffs = []
    for i in range(len(sim_values)):
        for j in range(i + 1, len(sim_values)):
            diffs.append(abs(sim_values[i] - sim_values[j]))

    print(f"CLS similarity differences between all pairs:")
    print(f"  Differences: {[round(d, 3) for d in diffs]}")
    print(f"  Sorted diffs: {sorted([round(d, 3) for d in diffs])}")
    print()

    # 다양한 percentile 값 테스트
    percentiles = [10, 25, 50, 75, 90]

    for percentile in percentiles:
        print(f"--- Percentile {percentile} ---")
        threshold = np.percentile(diffs, percentile)
        print(f"  Computed threshold: {threshold:.4f}")

        # 모든 인접한 쌍에 대해 병합 결정 시뮬레이션
        merges = []
        cluster_ids = list(cls_similarities.keys())
        for i in range(len(cluster_ids) - 1):
            cluster1 = cluster_ids[i]
            cluster2 = cluster_ids[i + 1]

            cls_sim1 = cls_similarities[cluster1]
            cls_sim2 = cls_similarities[cluster2]
            cls_diff = abs(cls_sim1 - cls_sim2)

            should_merge = cls_diff <= threshold
            merges.append((cluster1, cluster2, should_merge, cls_diff))

            status = "MERGE" if should_merge else "SKIP"
            print(f"  {cluster1}+{cluster2}: {cls_sim1:.2f} vs {cls_sim2:.2f} "
                  f"(diff: {cls_diff:.3f}) -> {status}")

        total_merges = sum(1 for _, _, merge, _ in merges)
        print(f"  Total merges: {total_merges}/{len(merges)}")
        print()

    print("Expected behavior (FIXED):")
    print("- Low percentile (10-25) -> Low threshold -> Strict criteria -> Few merges")
    print("- High percentile (75-90) -> High threshold -> Lenient criteria -> More merges")

if __name__ == "__main__":
    test_fixed_percentile_direction()