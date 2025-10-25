#!/usr/bin/env python3
"""
Percentile-only Threshold 테스트 스크립트
"""
import numpy as np

def test_percentile_threshold():
    """Percentile 기반 threshold 테스트"""
    print("="*60)
    print("Testing Percentile-only Threshold Calculation")
    print("="*60)

    # 테스트용 CLS 유사도 데이터
    test_cases = [
        {
            "name": "High similarity cluster",
            "cls_sims": {0: 0.92, 1: 0.91, 2: 0.93, 3: 0.90, 4: 0.94},
            "description": "모두 높고 비슷한 유사도"
        },
        {
            "name": "Wide spread cluster",
            "cls_sims": {0: 0.95, 1: 0.60, 2: 0.85, 3: 0.40, 4: 0.75},
            "description": "넓게 퍼진 유사도"
        },
        {
            "name": "Low similarity cluster",
            "cls_sims": {0: 0.45, 1: 0.48, 2: 0.42, 3: 0.50, 4: 0.44},
            "description": "전반적으로 낮은 유사도"
        }
    ]

    def compute_percentile_threshold(cls_sims, percentile):
        """Percentile 기반 threshold 계산 (수정된 로직)"""
        sim_values = list(cls_sims.values())
        # 백분위수의 10%를 threshold로 사용
        percentile_value = np.percentile(sim_values, percentile)
        threshold = percentile_value * 0.1
        return np.clip(threshold, 0.01, 0.5)

    # 다양한 percentile 값 테스트
    percentiles = [25, 50, 75, 90]

    for case in test_cases:
        print(f"\n--- {case['name'].upper()}: {case['description']} ---")
        cls_sims = case["cls_sims"]
        sim_values = list(cls_sims.values())

        print(f"  CLS values: {[f'{v:.2f}' for v in sim_values]}")
        print(f"  Mean: {np.mean(sim_values):.3f}, Std: {np.std(sim_values):.3f}")

        print(f"\n  Thresholds by percentile:")
        for percentile in percentiles:
            threshold = compute_percentile_threshold(cls_sims, percentile)
            print(f"    {percentile}th percentile: {threshold:.3f}")

def test_merging_behavior():
    """다양한 percentile 값에 따른 병합 동작 테스트"""
    print("\n" + "="*60)
    print("Testing Merging Behavior with Different Percentiles")
    print("="*60)

    # 현실적인 시나리오
    realistic_cls_sims = {
        0: 0.92,  # 주요 객체
        1: 0.89,  # 주요 객체 (0과 비슷)
        2: 0.91,  # 주요 객체 (0,1과 비슷)
        3: 0.65,  # 중간 객체
        4: 0.45,  # 배경 (5와 비슷)
        5: 0.48,  # 배경 (4와 비슷)
    }

    print("Realistic scenario:")
    print("  High similarity group: [0.92, 0.89, 0.91]")
    print("  Medium similarity: 0.65")
    print("  Low similarity group: [0.45, 0.48]")

    # 인접 관계
    adjacent_pairs = [(0,1), (1,2), (2,3), (3,4), (4,5)]
    print(f"\nAdjacent pairs: {adjacent_pairs}")

    # 다양한 percentile 값 테스트
    percentiles = [50, 75, 90]

    for percentile in percentiles:
        print(f"\n--- {percentile}th percentile ---")

        # threshold 계산 (수정된 로직)
        sim_values = list(realistic_cls_sims.values())
        percentile_value = np.percentile(sim_values, percentile)
        threshold = percentile_value * 0.1
        threshold = np.clip(threshold, 0.01, 0.5)

        print(f"  Threshold: {threshold:.3f}")

        # 병합할 쌍 찾기
        merge_pairs = []
        for c1, c2 in adjacent_pairs:
            sim_diff = abs(realistic_cls_sims[c1] - realistic_cls_sims[c2])
            if sim_diff <= threshold:
                merge_pairs.append((c1, c2))

        print(f"  Merge pairs: {merge_pairs}")
        print(f"  Total merges: {len(merge_pairs)}/{len(adjacent_pairs)}")

        # 병합 결과 분석
        if merge_pairs:
            # Union-Find로 그룹 계산
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                root_x, root_y = find(x), find(y)
                if root_x != root_x:
                    parent[root_y] = root_x

            for c1, c2 in merge_pairs:
                union(c1, c2)

            # 그룹별로 분석
            groups = {}
            for cluster_id in realistic_cls_sims.keys():
                root = find(cluster_id)
                if root not in groups:
                    groups[root] = []
                groups[root].append(cluster_id)

            print(f"  Resulting groups:")
            for group_id, clusters in groups.items():
                sims = [realistic_cls_sims[c] for c in clusters]
                avg_sim = np.mean(sims)
                print(f"    Group {clusters}: avg CLS = {avg_sim:.3f}")

def test_percentile_guidelines():
    """Percentile 값 선택 가이드라인 테스트"""
    print("\n" + "="*60)
    print("Percentile Selection Guidelines")
    print("="*60)

    # 다양한 분포에 따른 권장 percentile
    guidelines = [
        {
            "scenario": "Very similar clusters (low variance)",
            "cls_sims": {0: 0.88, 1: 0.89, 2: 0.87, 3: 0.90, 4: 0.86},
            "recommended": "80-90 (strict - only very similar merge)"
        },
        {
            "scenario": "Mixed similarity clusters",
            "cls_sims": {0: 0.90, 1: 0.70, 2: 0.85, 3: 0.45, 4: 0.60},
            "recommended": "50-75 (balanced - similar ones merge)"
        },
        {
            "scenario": "Wide spread clusters",
            "cls_sims": {0: 0.95, 1: 0.60, 2: 0.85, 3: 0.40, 4: 0.75},
            "recommended": "25-50 (lenient - more merging)"
        }
    ]

    for guide in guidelines:
        print(f"\n--- {guide['scenario']} ---")
        cls_sims = guide["cls_sims"]
        sim_values = list(cls_sims.values())
        recommended = guide["recommended"]

        print(f"  CLS values: {[f'{v:.2f}' for v in sim_values]}")
        print(f"  Recommended percentile: {recommended}")
        print(f"  Mean: {np.mean(sim_values):.3f}, Std: {np.std(sim_values):.3f}")

        # 권장 범위에서 테스트
        range_str = recommended.split('-')[0].strip()  # 첫 번째 숫자만 사용
        base_percentile = int(range_str)

        # 권장 주변에서 테스트
        test_percentiles = [base_percentile - 10, base_percentile, base_percentile + 10]
        for perc in test_percentiles:
            if 0 <= perc <= 100:
                threshold = np.percentile(sim_values, perc)
                threshold = np.clip(threshold, 0.01, 0.5)
                print(f"    {perc}th percentile: {threshold:.3f}")

def main():
    """모든 테스트 실행"""
    print("Percentile-only Threshold Tests")
    print("=" * 80)

    test_percentile_threshold()
    test_merging_behavior()
    test_percentile_guidelines()

    print("\n" + "=" * 80)
    print("All percentile tests completed!")

    print("\nKey insights:")
    print("1. Percentile-based threshold is simple and intuitive")
    print("2. Lower percentile = more merging (more lenient)")
    print("3. Higher percentile = less merging (more strict)")
    print("4. 75th percentile is a good default (only top 25% similar clusters merge)")
    print("5. 50th percentile is more balanced (average similarity clusters merge)")

if __name__ == "__main__":
    main()