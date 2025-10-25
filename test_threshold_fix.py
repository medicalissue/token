#!/usr/bin/env python3
"""
수정된 Threshold 로직 테스트 스크립트
"""
import numpy as np

def test_threshold_logic():
    """수정된 threshold 로직 테스트"""
    print("="*60)
    print("Testing Fixed Threshold Logic")
    print("="*60)

    # 테스트 시나리오
    test_cases = [
        {
            "name": "High similarity clusters",
            "cls_sims": {0: 0.85, 1: 0.87, 2: 0.86, 3: 0.84},
            "threshold": 0.1,
            "expected_merges": [(0,1), (0,2), (1,2), (0,3), (1,3), (2,3)]
        },
        {
            "name": "Mixed similarity clusters",
            "cls_sims": {0: 0.90, 1: 0.70, 2: 0.85, 3: 0.45},
            "threshold": 0.1,
            "expected_merges": [(0,2)]
        },
        {
            "name": "Low similarity threshold (lenient)",
            "cls_sims": {0: 0.60, 1: 0.80, 2: 0.70, 3: 0.90},
            "threshold": 0.3,
            "expected_merges": [(0,1), (0,2), (1,2), (1,3), (2,3)]
        },
        {
            "name": "High similarity threshold (strict)",
            "cls_sims": {0: 0.85, 1: 0.87, 2: 0.86, 3: 0.84},
            "threshold": 0.02,
            "expected_merges": []
        }
    ]

    # 인접 쌍 (모든 쌍이 인접한다고 가정)
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {case['name']} ---")
        cls_sims = case["cls_sims"]
        threshold = case["threshold"]
        expected_merges = set(case["expected_merges"])

        print(f"  CLS similarities: {cls_sims}")
        print(f"  Threshold: {threshold}")

        # 모든 가능한 쌍 테스트
        all_pairs = []
        cluster_ids = list(cls_sims.keys())
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                all_pairs.append((cluster_ids[i], cluster_ids[j]))

        actual_merges = set()
        print(f"\n  Pairwise analysis:")
        for pair in all_pairs:
            sim1, sim2 = cls_sims[pair[0]], cls_sims[pair[1]]
            sim_diff = abs(sim1 - sim2)

            # 수정된 로직: 차이가 threshold보다 작으면 병합
            should_merge = sim_diff <= threshold

            print(f"    {pair}: {sim1:.3f} vs {sim2:.3f} (diff: {sim_diff:.3f}) -> {'MERGE' if should_merge else 'SKIP'}")

            if should_merge:
                actual_merges.add(pair)

        print(f"\n  Expected merges: {sorted(expected_merges)}")
        print(f"  Actual merges:   {sorted(actual_merges)}")
        print(f"  Match: {'✓' if expected_merges == actual_merges else '✗'}")

def test_dynamic_threshold_behavior():
    """동적 threshold의 실제 동작 테스트"""
    print("\n" + "="*60)
    print("Testing Dynamic Threshold Behavior")
    print("="*60)

    # 다양한 분포의 CLS 유사도
    distributions = {
        "tight_cluster": {
            "cls_sims": {0: 0.88, 1: 0.89, 2: 0.87, 3: 0.90},
            "description": "모두 높고 비슷한 유사도"
        },
        "spread_out": {
            "cls_sims": {0: 0.95, 1: 0.60, 2: 0.85, 3: 0.40},
            "description": "넓게 퍼진 유사도"
        },
        "two_groups": {
            "cls_sims": {0: 0.90, 1: 0.88, 2: 0.45, 3: 0.47},
            "description": "두 개의 명확한 그룹"
        }
    }

    def compute_threshold(cls_sims, method):
        """다양한 방법으로 threshold 계산"""
        sim_values = list(cls_sims.values())

        if method == "mean":
            return np.mean(sim_values)
        elif method == "std":
            return np.std(sim_values)
        elif method == "mean-std":
            return np.mean(sim_values) - np.std(sim_values)
        elif method == "q1":
            return np.percentile(sim_values, 25)
        elif method == "median":
            return np.median(sim_values)
        else:
            return 0.1  # 기본값

    methods = ["mean", "std", "mean-std", "q1", "median"]

    for dist_name, dist_data in distributions.items():
        print(f"\n--- {dist_name.upper()}: {dist_data['description']} ---")
        cls_sims = dist_data["cls_sims"]
        sim_values = list(cls_sims.values())

        print(f"  Values: {[f'{v:.2f}' for v in sim_values]}")
        print(f"  Mean: {np.mean(sim_values):.3f}, Std: {np.std(sim_values):.3f}")

        print(f"\n  Thresholds by method:")
        for method in methods:
            threshold = compute_threshold(cls_sims, method)
            # threshold는 너무 크면 안되니 적절히 조정
            if method == "mean":
                threshold = min(threshold * 0.3, 0.5)  # mean의 30%로 조정
            elif method == "std":
                threshold = min(threshold * 0.5, 0.5)  # std의 50%로 조정

            threshold = max(threshold, 0.01)  # 최소 0.01 보장

            print(f"    {method:8}: {threshold:.3f}")

            # 이 threshold로 몇 개의 쌍이 병합될지 테스트
            cluster_ids = list(cls_sims.keys())
            merge_count = 0
            for i in range(len(cluster_ids)):
                for j in range(i+1, len(cluster_ids)):
                    sim_diff = abs(cls_sims[cluster_ids[i]] - cls_sims[cluster_ids[j]])
                    if sim_diff <= threshold:
                        merge_count += 1

            total_pairs = len(cluster_ids) * (len(cluster_ids) - 1) // 2
            print(f"              -> {merge_count}/{total_pairs} pairs would merge")

def main():
    """모든 테스트 실행"""
    print("Fixed Threshold Logic Tests")
    print("=" * 80)

    test_threshold_logic()
    test_dynamic_threshold_behavior()

    print("\n" + "=" * 80)
    print("All tests completed!")

    print("\nKey findings:")
    print("1. Fixed logic: cls_diff <= threshold (not 1 - threshold)")
    print("2. Lower threshold = more strict merging")
    print("3. Higher threshold = more lenient merging")
    print("4. Dynamic threshold should be adjusted based on data distribution")

if __name__ == "__main__":
    main()