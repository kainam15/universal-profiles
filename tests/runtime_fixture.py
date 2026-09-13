"""隔离依赖构建输入，避免身份测试修改工作区。"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def copy_dependency_tree(root):
    shutil.copytree(ROOT / 'dockerfiles', root / 'dockerfiles')
    (root / 'acprof').mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'acprof/dependency_locks.py', root / 'acprof/dependency_locks.py')
