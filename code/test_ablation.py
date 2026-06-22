"""
简单消融实验测试脚本
=====================
测试不同的特征融合方法
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, r"D:\work\mynetZXIAO")

from models.MyNet.net_desc import mynet, FUSION_METHODS


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_fusion_methods():
    """测试所有融合方法"""
    print("="*70)
    print("消融实验: GIT替代方法测试")
    print("="*70)

    methods = [
        'GIT',           # 原始方法
        'Concat',        # 简单拼接+卷积
        'SE',            # SE注意力
        'CBAM',          # CBAM注意力
        'SelfAttention', # 自注意力
        'WeightSum',     # 加权求和
        'NoFusion',      # 不融合
        'Identity',      # 恒等映射
        'Res2NetStyle', # Res2Net风格
    ]

    results = {}
    input_tensor = torch.randn(2, 3, 384, 384)

    print(f"\n输入形状: {input_tensor.shape}")
    print("\n")

    for method in methods:
        try:
            print(f"测试方法: {method:<15}", end=" ")

            # 创建模型
            model = mynet(
                channel=256,
                M=[32, 32, 32],
                N=[8, 16, 32],
                nr_types=3,
                fusion_method=method
            ).eval()

            # 前向传播
            with torch.no_grad():
                output = model(input_tensor)

            # 计算参数量
            params = count_parameters(model)

            # 验证输出形状
            out_shapes = {k: v.shape for k, v in output.items()}

            print(f"✓ 参数: {params:>10,} | ", end="")
            print(f"输出: hv={out_shapes['hv'].shape}, tp={out_shapes['tp'].shape}")

            results[method] = {
                'params': params,
                'output_shapes': out_shapes,
                'status': 'OK'
            }

        except Exception as e:
            print(f"✗ 错误: {e}")
            results[method] = {
                'params': 0,
                'error': str(e),
                'status': 'FAILED'
            }

    # 打印结果汇总
    print("\n" + "="*70)
    print("结果汇总")
    print("="*70)
    print(f"{'方法':<15} {'参数量':>12} {'状态':<10}")
    print("-"*40)

    for method, res in results.items():
        status = res.get('status', 'N/A')
        params = res.get('params', 0)
        print(f"{method:<15} {params:>12,} {status:<10}")

    print("="*70)

    # 对比分析
    print("\n分析:")
    if 'GIT' in results and results['GIT']['status'] == 'OK':
        git_params = results['GIT']['params']
        print(f"- GIT基线参数量: {git_params:,}")

        for method, res in results.items():
            if method == 'GIT' or res['status'] != 'OK':
                continue
            params_diff = res['params'] - git_params
            params_ratio = res['params'] / git_params * 100
            print(f"- {method}: {params_diff:+,} 参数 ({params_ratio:.1f}% of GIT)")

    return results


def compare_with_git():
    """与GIT进行详细对比"""
    print("\n" + "="*70)
    print("GIT vs 替代方法详细对比")
    print("="*70)

    # 创建输入
    xr3 = torch.randn(2, 256, 48, 48)
    xr4 = torch.randn(2, 256, 24, 24)
    xr5 = torch.randn(2, 256, 12, 12)
    xg = torch.randn(2, 32, 48, 48)

    print(f"\n输入:")
    print(f"  xr3: {xr3.shape}")
    print(f"  xr4: {xr4.shape}")
    print(f"  xr5: {xr5.shape}")
    print(f"  xg:  {xg.shape}")

    # 测试GIT
    from models.MyNet.net_utils import GradientInducedTransition
    git_module = GradientInducedTransition(channel=256, M=[32, 32, 32], N=[8, 16, 32])
    git_module.eval()

    with torch.no_grad():
        git_out = git_module(xr3, xr4, xr5, xg)

    print(f"\nGIT输出:")
    for i, out in enumerate(git_out):
        print(f"  zt{i+3}: {out.shape} | mean={out.mean():.4f}, std={out.std():.4f}")

    # 测试替代方法
    methods = ['Concat', 'SE', 'CBAM', 'NoFusion', 'Identity']

    for method_name in methods:
        try:
            model = mynet(fusion_method=method_name)
            model.eval()

            with torch.no_grad():
                out = model(torch.randn(2, 3, 384, 384))

            print(f"\n{method_name}输出:")
            for k, v in out.items():
                print(f"  {k}: {v.shape} | mean={v.mean():.4f}, std={v.std():.4f}")

        except Exception as e:
            print(f"\n{method_name}: 错误 - {e}")


if __name__ == "__main__":
    # 测试所有融合方法
    results = test_fusion_methods()

    # 与GIT对比
    compare_with_git()

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
