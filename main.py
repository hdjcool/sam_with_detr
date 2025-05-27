"""
DETR 객체 탐지 메인 실행 파일
- 모델 초기화 및 객체 탐지
- 바운딩 박스 시각화 및 결과 저장
"""

import os
import sys
import torch
import argparse
from pathlib import Path

try:
    import detector
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    print("detector.py 파일이 현재 디렉터리에 있는지 확인하세요.")
    sys.exit(1)


def get_optimal_device():
    """최적의 디바이스를 선택하는 함수"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def validate_file_exists(file_path: str) -> bool:
    """파일 존재 여부 확인"""
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return False
    return True


def create_output_directory(output_dir: str = "results") -> str:
    """결과 저장용 디렉터리를 생성하는 함수"""
    Path(output_dir).mkdir(exist_ok=True)
    print(f"결과 디렉터리: {output_dir}")
    return output_dir


def download_model_files():
    """모델 파일 다운로드 안내"""
    print("=" * 60)
    print("모델 파일이 필요합니다. 다음 명령어로 다운로드하세요:")
    print()
    print("1. DETR 설정 파일:")
    print("   mkdir -p configs")
    print("   wget -O configs/detr_r50_8xb2-150e_coco.py \\")
    print("     https://github.com/open-mmlab/mmdetection/raw/main/configs/detr/detr_r50_8xb2-150e_coco.py")
    print()
    print("2. DETR 체크포인트 파일:")
    print("   mkdir -p weights")
    print("   wget -O weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth \\")
    print("     https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth")
    print()
    print("또는 mim을 사용하여 다운로드:")
    print("   pip install openmim")
    print("   mim download mmdet --config detr_r50_8xb2-150e_coco --dest ./weights")
    print("=" * 60)


def run_detection(input_path: str, score_threshold: float = 0.5):
    """
    객체 탐지를 수행하는 메인 함수

    Args:
        input_path: 입력 이미지 경로
        score_threshold: 탐지 신뢰도 임계값
    """
    # 출력 디렉터리 생성
    output_dir = create_output_directory("results")
    output_path = os.path.join(output_dir, f"detection_result_{Path(input_path).stem}.png")

    # 모델 경로 설정
    cfg_path = "configs/detr_r50_8xb2-150e_coco.py"
    ckpt_path = "weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"

    # 파일 존재 확인
    missing_files = []
    if not validate_file_exists(input_path):
        missing_files.append(input_path)
    if not validate_file_exists(cfg_path):
        missing_files.append(cfg_path)
    if not validate_file_exists(ckpt_path):
        missing_files.append(ckpt_path)

    if missing_files:
        if cfg_path in missing_files or ckpt_path in missing_files:
            download_model_files()
        return False

    try:
        # 디바이스 설정
        device = get_optimal_device()
        print(f"사용 디바이스: {device}")

        # 모델 초기화
        print("=" * 50)
        print("DETR 모델을 초기화하는 중...")
        model = detector.initialize(cfg_path, ckpt_path, device)
        print("모델 초기화 완료!")

        # 객체 탐지 수행
        print("=" * 50)
        print(f"객체 탐지 수행 중... (신뢰도 임계값: {score_threshold})")

        detections = detector.detect_objects(input_path, model, score_threshold)

        if not detections['bboxes']:
            print("탐지된 객체가 없습니다.")
            return False

        # 탐지 결과 출력
        print(f"\n탐지된 객체 정보:")
        for i, (bbox, score) in enumerate(zip(detections['bboxes'], detections['scores'])):
            x1, y1, x2, y2 = bbox
            print(f"  객체 {i+1}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] (신뢰도: {score:.3f})")

        # 결과 시각화 및 저장
        print("=" * 50)
        print("결과 시각화 및 저장 중...")

        success = detector.save_detection_result(
            image_path=input_path,
            output_path=output_path,
            model=model,
            score_threshold=score_threshold
        )

        if success:
            print("=" * 50)
            print("처리 완료!")
            print(f"  입력 이미지: {input_path}")
            print(f"  출력 결과: {output_path}")
            print(f"  탐지된 객체 수: {len(detections['bboxes'])}")
            print("=" * 50)
        else:
            print("결과 저장에 실패했습니다.")
            return False

        return True

    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='DETR 객체 탐지')
    parser.add_argument('--input', '-i', type=str, default='image_5.jpeg',
                       help='입력 이미지 경로')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='탐지 신뢰도 임계값 (0.0-1.0)')

    args = parser.parse_args()

    # 입력 검증
    if not (0.0 <= args.threshold <= 1.0):
        print(f"오류: threshold는 0.0과 1.0 사이여야 합니다. (입력값: {args.threshold})")
        return

    print("DETR 객체 탐지 시작")
    print(f"입력 이미지: {args.input}")
    print(f"신뢰도 임계값: {args.threshold}")

    # 탐지 실행
    success = run_detection(args.input, args.threshold)

    if success:
        print("\n탐지가 성공적으로 완료되었습니다!")
    else:
        print("\n탐지 실행 중 문제가 발생했습니다.")
        sys.exit(1)


if __name__ == '__main__':
    main()
