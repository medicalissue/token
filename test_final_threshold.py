#!/usr/bin/env python3
"""
최종 Threshold 테스트 스크립트
"""
import numpy as np

def test_final_threshold_logic():
    """최종 threshold 로직 테스트"""
    print("="*60)
    print("Testing Final Threshold Logic (Similarity-based Merging)")
    print("="*60)

    # 최종 threshold 계산 함수
    def compute_final_threshold(cls_sims, method):
        sim_values = np.array(list(cls_sims.values()))

        if method == "mean":
            threshold = np.mean(sim_values) * 0.1
        elif method == "median":
            threshold = np.median(sim_values) * 0.1
        elif method == "mean-std":
            threshold = max(np.mean(sim_values) - np.std(sim_values), 0.05)
            threshold = min(threshold, 0.3)
        elif method == "q1":
            threshold = np.percentile(sim_values, 25)
        else:
            threshold = 0.1  # 기본값

        return np.clip(threshold, 0.01, 0.5)

    # 테스트 시나리오
    realistic_sims = {
        0: 0.92,  # 주요 객체
        1: 0.89,  # 주요 객체 (0과 비슷)
        2: 0.91,  # 주요 객체 (0,1과 비슷)
        3: 0.65,  # 중요 객체 (0,1,2와 다름)
        4: 0.45,  # 배경 (5와 비슷)
        5: 0.48,  # 배경 (4와 비슷)
    }

    print("Realistic scenario:")
    print("  High similarity group: [0.92, 0.89, 0.91] (should merge)")
    print("  Medium similarity: 0.65 (should be separate)")
    print("  Low similarity group: [0.45, 0.48] (should merge)")

    # 인접 관계
    adjacent_pairs = [(0,1), (1,2), (2,3), (3,4), (4,5)]
    print(f"\nAdjacent pairs: {adjacent_pairs}")

    # 다양한 방법으로 테스트
    methods = ["mean", "median", "mean-std", "q1"]

    for method in methods:
        print(f"\n--- {method.upper()} method ---")
        threshold = compute_final_threshold(realistic_sims, method)
        print(f"  Computed threshold: {threshold:.3f}")

        print(f"  Merging decisions:")
        merge_count = 0
        for c1, c2 in adjacent_pairs:
            sim1, sim2 = realistic_sims[c1], realistic_sims[c2]
            sim_diff = abs(sim1 - sim2)
            should_merge = sim_diff <= threshold

            if should_merge:
                merge_count += 1
                print(f"    {c1}+{c2}: {sim1:.3f} vs {sim2:.3f} (diff: {sim_diff:.3f}) -> MERGE")
            else:
                print(f"    {c1}+{c2}: {sim1:.3f} vs {sim2:.3f} (diff: {sim_diff:.3f}) -> SKIP")

        print(f"  Total merges: {merge_count}/{len(adjacent_pairs)}")

def test_expected_behavior():
    """기대되는 동작 테스트"""
    print("\n" + "="*60)
    print("Testing Expected Behavior")
    print("="*60)

    # 기대: 비슷한 것들만 병합되어야 함
    test_cases = [
        {
            "name": "Very similar clusters",
            "cls_sims": {0: 0.85, 1: 0.86, 2: 0.84, 3: 0.87},
            "adjacent_pairs": [(0,1), (1,2), (2,3)],
            "expected_merges": 3  # 모두 병합되어야 함
        },
        {
            "name": "Very different clusters",
            "cls_sims": {0: 0.90, 1: 0.50, 2: 0.30, 3: 0.85},
            "adjacent_pairs": [(0,1), (1,2), (2,3)],
            "expected_merges": 0  # 아무것도 병합되지 않아야 함
        },
        {
            "name": "Mixed similarity",
            "cls_sims": {0: 0.90, 1: 0.88, 2: 0.65, 3: 0.45, 4: 0.47},
            "adjacent_pairs": [(0,1), (1,2), (2,3), (3,4)],
            "expected_merges": 2  # (0,1)과 (3,4)만 병합
        }
    ]

    def compute_threshold(cls_sims):
        return np.mean(list(cls_sims.values())) * 0.1

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        cls_sims = case["cls_sims"]
        adjacent_pairs = case["adjacent_pairs"]
        expected = case["expected_merges"]

        print(f"  CLS similarities: {cls_sims}")
        print(f"  Expected merges: {expected}")

        threshold = compute_threshold(cls_sims)
        print(f"  Computed threshold: {threshold:.3f}")

        actual_merges = 0
        print(f"  Actual merging:")
        for c1, c2 in adjacent_pairs:
            sim_diff = abs(cls_sims[c1] - cls_sims[c2])
            if sim_diff <= threshold:
                actual_merges += 1
                print(f"    {c1}+{c2}: {cls_sims[c1]:.3f} vs {cls_sims[c2]:.3f} (diff: {sim_diff:.3f}) -> MERGE")
            else:
                print(f"    {c1}+{c2}: {cls_sims[c1]:.3f} vs {cls_sims[c2]:.3f} (diff: {sim_diff:.3f}) -> SKIP")

        result = "✓" if actual_merges == expected else "✗"
        print(f"  Result: {actual_merges} merges (expected {expected}) {result}")

def main():
    """모든 테스트 실행"""
    print("Final Threshold Logic Tests")
    print("=" * 80)

    test_final_threshold_logic()
    test_expected_behavior()

    print("\n" + "=" * 80)
    print("All tests completed!")

    print("\nFinal logic summary:")
    print("1. Two clusters merge when their CLS similarities are similar")
    print("2. Similarity is measured by: abs(cls_sim1 - cls_sim2)")
    print("3. Merging condition: similarity_diff <= threshold")
    print("4. Threshold is computed dynamically from all CLS similarities")
    print("5. Recommended default: 'mean' method with 10% scaling")

if __name__ == "__main__":
    main()