#!/usr/bin/env python3
"""
간단한 Percentile 로직 방향성 테스트
"""

import numpy as np

def test_percentile_direction():
    """Percentile 값이 낮을수록 덜 병합되는지 테스트"""

    print("="*60)
    print("Percentile 방향성 테스트")
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

    sim_values = np.array(list(cls_similarities.values()))
    print(f"Sim values: {sim_values}")
    print(f"Sorted: {np.sort(sim_values)}")
    print()

    # 다양한 percentile 값 테스트
    percentiles = [10, 25, 50, 75, 90]

    for percentile in percentiles:
        print(f"--- Percentile {percentile} ---")
        threshold = np.percentile(sim_values, percentile)
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

    print("Expected behavior:")
    print("- Low percentile (10-25) -> Low threshold -> Strict criteria -> Few merges")
    print("- High percentile (75-90) -> High threshold -> Lenient criteria -> More merges")

if __name__ == "__main__":
    test_percentile_direction()