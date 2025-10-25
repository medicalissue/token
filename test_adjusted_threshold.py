#!/usr/bin/env python3
"""
수정된 동적 Threshold 테스트 스크립트
"""
import numpy as np

def test_adjusted_dynamic_threshold():
    """수정된 동적 threshold 테스트"""
    print("="*60)
    print("Testing Adjusted Dynamic Threshold")
    print("="*60)

    # 수정된 threshold 계산 함수
    def compute_adjusted_threshold(cls_sims, method):
        sim_values = np.array(list(cls_sims.values()))

        if method == "mean":
            threshold = np.mean(sim_values) * 0.3
        elif method == "mean+std":
            threshold = min((np.mean(sim_values) + np.std(sim_values)) * 0.4, 0.5)
        elif method == "mean-std":
            threshold = max(np.mean(sim_values) - np.std(sim_values), 0.05)
            threshold = min(threshold, 0.8)
        elif method == "median":
            threshold = np.median(sim_values) * 0.4
        elif method == "q1":
            threshold = np.percentile(sim_values, 25)
        elif method == "q3":
            threshold = np.percentile(sim_values, 75) * 0.3
        else:
            threshold = 0.1  # 기본값

        return np.clip(threshold, 0.01, 0.9)

    # 테스트 시나리오
    test_scenarios = [
        {
            "name": "High similarity cluster (tight)",
            "cls_sims": {0: 0.88, 1: 0.89, 2: 0.87, 3: 0.90, 4: 0.86},
            "description": "모두 높고 비슷한 유사도"
        },
        {
            "name": "Wide spread cluster",
            "cls_sims": {0: 0.95, 1: 0.60, 2: 0.85, 3: 0.40, 4: 0.75},
            "description": "넓게 퍼진 유사도"
        },
        {
            "name": "Two distinct groups",
            "cls_sims": {0: 0.90, 1: 0.88, 2: 0.45, 3: 0.47, 4: 0.43},
            "description": "두 개의 명확한 그룹"
        },
        {
            "name": "Low similarity cluster",
            "cls_sims": {0: 0.35, 1: 0.40, 2: 0.38, 3: 0.42, 4: 0.36},
            "description": "전반적으로 낮은 유사도"
        }
    ]

    methods = ["mean", "mean+std", "mean-std", "median", "q1", "q3"]

    for scenario in test_scenarios:
        print(f"\n--- {scenario['name'].upper()}: {scenario['description']} ---")
        cls_sims = scenario["cls_sims"]
        sim_values = list(cls_sims.values())

        print(f"  CLS values: {[f'{v:.2f}' for v in sim_values]}")
        print(f"  Mean: {np.mean(sim_values):.3f}, Std: {np.std(sim_values):.3f}, Median: {np.median(sim_values):.3f}")

        print(f"\n  Thresholds by method:")
        for method in methods:
            threshold = compute_adjusted_threshold(cls_sims, method)

            # 이 threshold로 병합될 쌍 수 계산
            cluster_ids = list(cls_sims.keys())
            merge_count = 0
            total_pairs = 0

            for i in range(len(cluster_ids)):
                for j in range(i+1, len(cluster_ids)):
                    total_pairs += 1
                    sim_diff = abs(cls_sims[cluster_ids[i]] - cls_sims[cluster_ids[j]])
                    if sim_diff <= threshold:
                        merge_count += 1

            merge_ratio = merge_count / total_pairs if total_pairs > 0 else 0

            print(f"    {method:8}: {threshold:.3f} -> {merge_count}/{total_pairs} pairs ({merge_ratio:.1%})")

def test_realistic_merging():
    """현실적인 병합 시나리오 테스트"""
    print("\n" + "="*60)
    print("Testing Realistic Merging Scenarios")
    print("="*60)

    # 현실적인 시나리오: 6개 클러스터가 서로 다른 CLS 유사도를 가짐
    realistic_cls_sims = {
        0: 0.92,  # 주요 객체
        1: 0.89,  # 주요 객체 (0과 유사)
        2: 0.91,  # 주요 객체 (0,1과 유사)
        3: 0.65,  # 중요 객체
        4: 0.45,  # 배경/부수 객체
        5: 0.48,  # 배경/부수 객체 (4와 유사)
    }

    print("Realistic CLS similarities:")
    for cluster_id, sim in realistic_cls_sims.items():
        print(f"  Cluster {cluster_id}: {sim:.3f}")

    # 인접 관계 (실제 이미지에서 나올 수 있는 인접 관계)
    adjacent_pairs = [(0,1), (1,2), (2,3), (3,4), (4,5)]

    print(f"\nAdjacent pairs: {adjacent_pairs}")

    def test_threshold_method(cls_sims, pairs, method):
        # threshold 계산 (수정된 로직)
        sim_values = list(cls_sims.values())

        if method == "median":
            threshold = np.median(sim_values) * 0.4
        elif method == "mean":
            threshold = np.mean(sim_values) * 0.3
        elif method == "q1":
            threshold = np.percentile(sim_values, 25)
        elif method == "mean-std":
            threshold = max(np.mean(sim_values) - np.std(sim_values), 0.05)
            threshold = min(threshold, 0.8)
        else:
            threshold = 0.1

        threshold = np.clip(threshold, 0.01, 0.9)

        # 병합할 쌍 찾기
        merge_pairs = []
        for c1, c2 in pairs:
            sim_diff = abs(cls_sims[c1] - cls_sims[c2])
            if sim_diff <= threshold:
                merge_pairs.append((c1, c2))

        return threshold, merge_pairs

    methods = ["median", "mean", "q1", "mean-std"]

    print(f"\nMerging results by method:")
    for method in methods:
        threshold, merge_pairs = test_threshold_method(realistic_cls_sims, adjacent_pairs, method)

        print(f"\n  {method.upper()} (threshold: {threshold:.3f}):")
        print(f"    Merge pairs: {merge_pairs}")

        # 병합 결과 분석
        if merge_pairs:
            # Union-Find로 병합 그룹 계산
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                root_x, root_y = find(x), find(y)
                if root_x != root_y:
                    parent[root_y] = root_x

            for c1, c2 in merge_pairs:
                union(c1, c2)

            # 그룹별로 CLS 유사도 보기
            groups = {}
            for cluster_id in realistic_cls_sims.keys():
                root = find(cluster_id)
                if root not in groups:
                    groups[root] = []
                groups[root].append(cluster_id)

            print(f"    Resulting groups:")
            for group_id, clusters in groups.items():
                sims = [realistic_cls_sims[c] for c in clusters]
                avg_sim = np.mean(sims)
                print(f"      Group {clusters}: avg CLS = {avg_sim:.3f}")
        else:
            print("    No merging")

def main():
    """모든 테스트 실행"""
    print("Adjusted Dynamic Threshold Tests")
    print("=" * 80)

    test_adjusted_dynamic_threshold()
    test_realistic_merging()

    print("\n" + "=" * 80)
    print("All tests completed!")

    print("\nKey improvements:")
    print("1. Fixed threshold logic: cls_diff <= threshold")
    print("2. Adjusted dynamic thresholds to reasonable ranges [0.01, 0.9]")
    print("3. Applied scaling factors to prevent excessive merging")
    print("4. Recommended default: 'median' method with 40% scaling")
    print("5. Thresholds now produce more controlled merging behavior")

if __name__ == "__main__":
    main()