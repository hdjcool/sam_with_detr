"""
모델 파일 자동 다운로드 스크립트
- DETR 설정 파일 및 체크포인트 다운로드
- SAM 모델 파일 다운로드 (향후 추가)
- 다운로드 진행상황 표시
"""

import os
import urllib.request
import hashlib
from pathlib import Path
from tqdm import tqdm


class ModelDownloader:
    """모델 파일 다운로드 클래스"""

    def __init__(self):
        self.models = {
            'detr_config': {
                'url': 'https://github.com/open-mmlab/mmdetection/raw/main/configs/detr/detr_r50_8xb2-150e_coco.py',
                'path': 'configs/detr_r50_8xb2-150e_coco.py',
                'description': 'DETR 설정 파일'
            },
            'detr_checkpoint': {
                'url': 'https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth',
                'path': 'weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth',
                'description': 'DETR 체크포인트 파일 (~166MB)',
                'size_mb': 166
            }
        }

    def create_directories(self):
        """필요한 디렉토리 생성"""
        dirs = ['weights', 'configs', 'results']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
            print(f"✓ {dir_name}/ 디렉토리 준비")

    def file_exists(self, filepath: str) -> bool:
        """파일 존재 여부 확인"""
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0

    def download_with_progress(self, url: str, filepath: str, description: str):
        """진행바와 함께 파일 다운로드"""
        print(f"\n📥 {description} 다운로드 중...")
        print(f"URL: {url}")
        print(f"저장 위치: {filepath}")

        try:
            # 파일 크기 확인
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))

            # 다운로드
            with urllib.request.urlopen(url) as response:
                with open(filepath, 'wb') as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc="다운로드") as pbar:
                            while True:
                                chunk = response.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))
                    else:
                        f.write(response.read())

            print(f"✅ {description} 다운로드 완료!")
            return True

        except Exception as e:
            print(f"❌ {description} 다운로드 실패: {e}")
            return False

    def download_model(self, model_key: str) -> bool:
        """특정 모델 다운로드"""
        if model_key not in self.models:
            print(f"❌ 알 수 없는 모델: {model_key}")
            return False

        model_info = self.models[model_key]
        filepath = model_info['path']

        # 이미 존재하는 경우 건너뛰기
        if self.file_exists(filepath):
            print(f"✓ {model_info['description']} 이미 존재: {filepath}")
            return True

        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 다운로드
        return self.download_with_progress(
            model_info['url'],
            filepath,
            model_info['description']
        )

    def download_all(self, skip_large_files: bool = False):
        """모든 모델 파일 다운로드"""
        print("🚀 모델 파일 다운로드 시작")
        print("=" * 50)

        self.create_directories()

        success_count = 0
        total_count = len(self.models)

        for model_key, model_info in self.models.items():
            # 큰 파일 건너뛰기 옵션
            if skip_large_files and model_info.get('size_mb', 0) > 50:
                print(f"⏭️  큰 파일 건너뛰기: {model_info['description']}")
                print(f"   수동 다운로드: wget -O {model_info['path']} {model_info['url']}")
                continue

            if self.download_model(model_key):
                success_count += 1

        print("\n" + "=" * 50)
        print(f"📊 다운로드 결과: {success_count}/{total_count} 성공")

        if success_count == total_count:
            print("🎉 모든 모델 파일 다운로드 완료!")
        else:
            print("⚠️  일부 파일 다운로드에 실패했습니다.")
            print("수동 다운로드 명령어를 참고하세요.")

    def verify_files(self):
        """다운로드된 파일 검증"""
        print("\n🔍 파일 검증 중...")

        for model_key, model_info in self.models.items():
            filepath = model_info['path']

            if self.file_exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"✅ {model_info['description']}: {file_size:.1f}MB")
            else:
                print(f"❌ {model_info['description']}: 파일 없음")

    def print_manual_commands(self):
        """수동 다운로드 명령어 출력"""
        print("\n📋 수동 다운로드 명령어:")
        print("=" * 50)

        for model_key, model_info in self.models.items():
            print(f"\n# {model_info['description']}")
            print(f"mkdir -p {os.path.dirname(model_info['path'])}")
            print(f"wget -O {model_info['path']} \\")
            print(f"  {model_info['url']}")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='모델 파일 다운로드')
    parser.add_argument('--skip-large', action='store_true',
                       help='50MB 이상의 큰 파일 건너뛰기')
    parser.add_argument('--verify-only', action='store_true',
                       help='다운로드 없이 파일 검증만 수행')
    parser.add_argument('--manual-commands', action='store_true',
                       help='수동 다운로드 명령어만 출력')

    args = parser.parse_args()

    downloader = ModelDownloader()

    if args.manual_commands:
        downloader.print_manual_commands()
    elif args.verify_only:
        downloader.verify_files()
    else:
        # 큰 파일 다운로드 여부 확인
        if not args.skip_large:
            print("⚠️  DETR 체크포인트 파일은 166MB입니다.")
            print("시간이 오래 걸릴 수 있습니다.")
            response = input("계속 다운로드하시겠습니까? (y/N): ")

            if response.lower() not in ['y', 'yes']:
                args.skip_large = True
                print("큰 파일을 건너뛰고 설정 파일만 다운로드합니다.")

        downloader.download_all(skip_large_files=args.skip_large)
        downloader.verify_files()

        if args.skip_large:
            downloader.print_manual_commands()


if __name__ == '__main__':
    main()
