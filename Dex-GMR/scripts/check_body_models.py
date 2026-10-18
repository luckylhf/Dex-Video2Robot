"""SMPL-X 人体模型自检。

SMPL-X 参数受 MPI 许可限制，不随源码包分发。缺少文件时给出明确指引，
避免下游看到 smplx 的底层异常而误以为包损坏。

用法：
    python scripts/check_body_models.py                # 检查 NEUTRAL/MALE/FEMALE
    python scripts/check_body_models.py --gender neutral
    python scripts/check_body_models.py --body-models /path/to/body_models --strict
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

SMPLX_URL = 'https://smpl-x.is.tue.mpg.de/'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / 'assets/body_models'

GENDER_FILES = {
    'neutral': 'SMPLX_NEUTRAL.npz',
    'male': 'SMPLX_MALE.npz',
    'female': 'SMPLX_FEMALE.npz',
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(smplx_dir: Path) -> dict[str, str]:
    manifest = smplx_dir / 'SHA256SUMS'
    if not manifest.is_file():
        return {}
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) == 2:
            expected[parts[1].lower()] = parts[0].lower()
    return expected


def check_body_models(
    body_models: Path = DEFAULT_DIR,
    genders: tuple[str, ...] = ('neutral',),
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """返回 (致命问题, 警告)。文件缺失一定是致命问题；哈希不一致默认只告警。"""
    smplx_dir = Path(body_models) / 'smplx'
    expected = read_manifest(smplx_dir)
    problems: list[str] = []
    warnings: list[str] = []

    for gender in genders:
        name = GENDER_FILES[gender]
        path = smplx_dir / name
        if not path.is_file():
            problems.append(
                f'缺少 {name}：{path}\n'
                f'    SMPL-X 参数不随源码包分发（MPI 许可限制），请自行下载：{SMPLX_URL}\n'
                f'    解压后把 smplx/{name} 放到 {smplx_dir}/\n'
                f'    下载后可用 `python scripts/check_body_models.py` 校验完整性'
            )
            continue

        want = expected.get(name.lower())
        if want:
            got = sha256_of(path)
            if got != want:
                message = (
                    f'{name} 的 sha256 与随包清单不一致：\n'
                    f'    期望 {want}\n'
                    f'    实际 {got}\n'
                    f'    可能是 SMPL-X 版本不同或下载损坏；确认来源可信后可继续使用'
                )
                (problems if strict else warnings).append(message)
        else:
            warnings.append(f'{smplx_dir / "SHA256SUMS"} 缺失或不含 {name}，跳过哈希校验')

    return problems, warnings


def require_body_models(
    body_models: Path = DEFAULT_DIR,
    genders: tuple[str, ...] = ('neutral',),
    strict: bool = False,
) -> None:
    """供 retarget_motion.py 调用：有问题时打印指引并以退出码 2 结束。"""
    problems, warnings = check_body_models(body_models, genders, strict)
    for warning in warnings:
        print(f'[body_models] 警告：{warning}')
    if problems:
        print('[body_models] 人体模型未就绪，无法继续：')
        for problem in problems:
            print(f'[body_models]   - {problem}')
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description='校验 SMPL-X 人体模型文件是否就绪')
    parser.add_argument('--body-models', type=Path, default=DEFAULT_DIR,
                        help='包含 smplx/ 的父目录（默认 assets/body_models）')
    parser.add_argument('--gender', choices=['neutral', 'male', 'female', 'all'], default='all')
    parser.add_argument('--strict', action='store_true', help='哈希不一致也视为失败')
    args = parser.parse_args()

    genders = tuple(GENDER_FILES) if args.gender == 'all' else (args.gender,)
    problems, warnings = check_body_models(args.body_models, genders, args.strict)
    for warning in warnings:
        print(f'[body_models] 警告：{warning}')
    if problems:
        print('[body_models] 校验失败：')
        for problem in problems:
            print(f'[body_models]   - {problem}')
        return 2
    print(f'[body_models] OK：{args.body_models}/smplx（{", ".join(genders)}）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
