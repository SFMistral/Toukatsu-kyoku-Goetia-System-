#!/usr/bin/env python3
"""
测试运行脚本
"""
import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_type="all", verbose=False, coverage=False):
    """运行测试"""
    cmd = ["python", "-m", "pytest"]
    
    # 根据测试类型选择测试路径
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "e2e":
        cmd.append("tests/e2e/")
    elif test_type == "database":
        cmd.extend(["tests/unit/database/", "tests/integration/test_database_integration.py"])
    elif test_type == "all":
        cmd.append("tests/")
    else:
        print(f"未知的测试类型: {test_type}")
        return 1
    
    # 添加选项
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=database", "--cov-report=html", "--cov-report=term"])
    
    # 运行测试
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行测试")
    parser.add_argument(
        "test_type", 
        choices=["all", "unit", "integration", "e2e", "database"],
        default="all",
        nargs="?",
        help="测试类型"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("-c", "--coverage", action="store_true", help="生成覆盖率报告")
    
    args = parser.parse_args()
    
    # 检查是否在项目根目录
    if not Path("database").exists():
        print("错误: 请在项目根目录运行此脚本")
        return 1
    
    return run_tests(args.test_type, args.verbose, args.coverage)

if __name__ == "__main__":
    sys.exit(main())