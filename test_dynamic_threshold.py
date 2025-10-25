#!/usr/bin/env python3
"""
동적 Threshold 테스트 스크립트
"""
import numpy as np

def test_dynamic_threshold_calculation():
    """동적 threshold 계산 테스트"""
    print("="*60)
    print("Testing Dynamic Threshold Calculation")
    print("="*60)

    # 모의 CLS 유사도 데이터
    mock_cls_sims = {
        0: 0.45,  # 낮은 유사도 (배경)
        1: 0.78,  # 중간 유사도
        2: 0.82,  # 중간 유사도
        3: 0.91,  # 높은 유사도 (주요 객체)
        4: 0.89,  # 높은 유사도 (주요 객체)
        5: 0.67,  # 중간 유사도
        6: 0.93,  # 높은 유사도 (주요 객체)
        7: 0.52,  # 낮은 유사도 (배경)
    }

    print("Mock CLS similarities:")
    for cluster_id, sim in mock_cls_sims.items():
        print(f"  Cluster {cluster_id}: {sim:.3f}")

    # 통계 계산
    sim_values = np.array(list(mock_cls_sims.values()))
    mean_val = np.mean(sim_values)
    std_val = np.std(sim_values)
    median_val = np.median(sim_values)
    q1_val = np.percentile(sim_values, 25)
    q3_val = np.percentile(sim_values, 75)

    print(f"\nStatistics:")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std:  {std_val:.4f}")
    print(f"  Median: {median_val:.4f}")
    print(f"  Q1 (25%): {q1_val:.4f}")
    print(f"  Q3 (75%): {q3_val:.4f}")

    def compute_dynamic_threshold(cls_sims, method, threshold_value=None):
        """동적 threshold 계산"""
        sim_values = np.array(list(cls_sims.values()))

        if method == "mean":
            threshold = np.mean(sim_values)
        elif method == "mean+std":
            threshold = np.mean(sim_values) + np.std(sim_values)
        elif method == "mean-std":
            threshold = np.mean(sim_values) - np.std(sim_values)
        elif method == "mean+2std":
            threshold = np.mean(sim_values) + 2 * np.std(sim_values)
        elif method == "mean-2std":
            threshold = np.mean(sim_values) - 2 * np.std(sim_values)
        elif method == "median":
            threshold = np.median(sim_values)
        elif method == "q1":
            threshold = np.percentile(sim_values, 25)
        elif method == "q3":
            threshold = np.percentile(sim_values, 75)
        elif method == "percentile":
            if threshold_value is None:
                raise ValueError("threshold_value must be specified for percentile mode")
            threshold = np.percentile(sim_values, threshold_value)
        elif method == "fixed":
            threshold = 0.8  # 기본값
        else:
            raise ValueError(f"Unknown method: {method}")

        return np.clip(threshold, 0.0, 1.0)

    # 다양한 방법으로 threshold 계산
    methods = [
        "mean", "mean+std", "mean-std", "mean+2std", "mean-2std",
        "median", "q1", "q3"
    ]

    print(f"\nDynamic thresholds by method:")
    for method in methods:
        threshold = compute_dynamic_threshold(mock_cls_sims, method)
        print(f"  {method:12}: {threshold:.4f}")

    # 특정 백분위수 계산
    percentiles = [10, 25, 50, 75, 90]
    print(f"\nPercentile thresholds:")
    for perc in percentiles:
        threshold = compute_dynamic_threshold(mock_cls_sims, "percentile", perc)
        print(f"  {perc}th percentile: {threshold:.4f}")

    print("✓ Dynamic threshold calculation test completed\n")

def test_merging_behavior():
    """다양한 threshold에 따른 병합 동작 테스트"""
    print("="*60)
    print("Testing Merging Behavior with Different Thresholds")
    print("="*60)

    # 테스트용 인접 쌍과 CLS 유사도
    adjacent_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    cls_sims = {
        0: 0.45,
        1: 0.78,
        2: 0.82,
        3: 0.91,
        4: 0.89,
        5: 0.67,
        6: 0.93,
    }

    print("Test scenario:")
    print(f"  Adjacent pairs: {adjacent_pairs}")
    print(f"  CLS similarities: {cls_sims}")

    def get_merge_pairs(pairs, cls_sims, threshold):
        """주어진 threshold로 병합할 쌍 찾기"""
        merge_pairs = []
        for cluster1, cluster2 in pairs:
            sim1 = cls_sims.get(cluster1, 0)
            sim2 = cls_sims.get(cluster2, 0)
            sim_diff = abs(sim1 - sim2)
            if sim_diff <= (1 - threshold):
                merge_pairs.append((cluster1, cluster2))
        return merge_pairs

    # 다양한 threshold로 테스트
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

    print(f"\nMerging behavior by threshold:")
    for threshold in thresholds:
        merge_pairs = get_merge_pairs(adjacent_pairs, cls_sims, threshold)
        print(f"  Threshold {threshold:.2f}: {len(merge_pairs)} merge pairs")
        for pair in merge_pairs:
            sim1, sim2 = cls_sims[pair[0]], cls_sims[pair[1]]
            sim_diff = abs(sim1 - sim2)
            print(f"    {pair}: {sim1:.3f} vs {sim2:.3f} (diff: {sim_diff:.3f})")

    print("✓ Merging behavior test completed\n")

def test_adaptive_threshold():
    """적응적 threshold 테스트"""
    print("="*60)
    print("Testing Adaptive Threshold Selection")
    print("="*60)

    # 다양한 시나리오 테스트
    scenarios = {
        "low_variance": {
            "cls_sims": {0: 0.85, 1: 0.87, 2: 0.86, 3: 0.88, 4: 0.84},
            "description": "낮은 분산 (모든 클러스터가 유사함)"
        },
        "high_variance": {
            "cls_sims": {0: 0.30, 1: 0.95, 2: 0.25, 3: 0.92, 4: 0.35},
            "description": "높은 분산 (클러스터 간 차이가 큼)"
        },
        "bimodal": {
            "cls_sims": {0: 0.40, 1: 0.45, 2: 0.42, 3: 0.88, 4: 0.91, 5: 0.89},
            "description": "이봉 분포 (낮은 그룹과 높은 그룹)"
        },
        "skewed": {
            "cls_sims": {0: 0.60, 1: 0.65, 2: 0.70, 3: 0.75, 4: 0.85, 5: 0.95},
            "description": "치우친 분포"
        }
    }

    for scenario_name, scenario_data in scenarios.items():
        print(f"\n--- {scenario_name.upper()}: {scenario_data['description']} ---")
        cls_sims = scenario_data["cls_sims"]

        sim_values = list(cls_sims.values())
        mean_val = np.mean(sim_values)
        std_val = np.std(sim_values)

        print(f"  Values: {[f'{v:.2f}' for v in sim_values]}")
        print(f"  Mean: {mean_val:.3f}, Std: {std_val:.3f}")

        # 다양한 방법으로 threshold 계산
        methods = ["mean", "mean-std", "median", "q1", "q3"]
        for method in methods:
            if method == "mean":
                threshold = mean_val
            elif method == "mean-std":
                threshold = mean_val - std_val
            elif method == "median":
                threshold = np.median(sim_values)
            elif method == "q1":
                threshold = np.percentile(sim_values, 25)
            elif method == "q3":
                threshold = np.percentile(sim_values, 75)

            threshold = np.clip(threshold, 0.0, 1.0)
            print(f"    {method:8}: {threshold:.3f}")

    print("✓ Adaptive threshold test completed\n")

def main():
    """모든 테스트 실행"""
    print("Dynamic Threshold Adjacency Merging Tests")
    print("=" * 80)

    test_dynamic_threshold_calculation()
    test_merging_behavior()
    test_adaptive_threshold()

    print("=" * 80)
    print("All dynamic threshold tests completed!")

    print("\nKey findings:")
    print("1. Dynamic threshold adapts to the distribution of CLS similarities")
    print("2. Different methods work better for different data distributions:")
    print("   - 'mean': Good for normal distributions")
    print("   - 'mean-std': More lenient, merges more clusters")
    print("   - 'median': Robust to outliers")
    print("   - 'q1': Very lenient, merges most similar clusters")
    print("   - 'q3': More strict, merges only very similar clusters")
    print("3. The method should be chosen based on desired merging behavior")

if __name__ == "__main__":
    main()