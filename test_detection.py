"""
DETR 탐지 기능 테스트 스크립트
- 모델 로딩 테스트
- 탐지 기능 검증
- 시각화 결과 확인
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

try:
    import detector
except ImportError:
    print("detector.py 모듈을 찾을 수 없습니다.")
    sys.exit(1)


def create_test_image(width=640, height=480, filename="test_image.jpg"):
    """테스트용 이미지 생성"""
    # 랜덤 배경 생성
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # 몇 개의 간단한 도형 추가 (객체처럼 보이도록)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # 파란 사각형
    cv2.circle(img, (300, 200), 60, (0, 255, 0), -1)  # 초록 원
    cv2.rectangle(img, (450, 300), (580, 420), (0, 0, 255), -1)  # 빨간 사각형

    # 텍스트 추가
    cv2.putText(img, "Test Image", (width//2-60, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(filename, img)
    print(f"✓ 테스트 이미지 생성: {filename}")
    return filename


def test_model_initialization():
    """모델 초기화 테스트"""
    print("=== 모델 초기화 테스트 ===")

    cfg_path = "configs/detr_r50_8xb2-150e_coco.py"
    ckpt_path = "weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"

    # 파일 존재 확인
    if not os.path.exists(cfg_path):
        print(f"✗ 설정 파일 없음: {cfg_path}")
        return None

    if not os.path.exists(ckpt_path):
        print(f"✗ 체크포인트 파일 없음: {ckpt_path}")
        return None

    try:
        # 디바이스 선택
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"사용 디바이스: {device}")

        # 모델 초기화
        model = detector.initialize(cfg_path, ckpt_path, device)
        print("✓ 모델 초기화 성공")
        return model

    except Exception as e:
        print(f"✗ 모델 초기화 실패: {e}")
        return None


def test_detection_function(model, test_image_path):
    """탐지 기능 테스트"""
    print("\n=== 탐지 기능 테스트 ===")

    if model is None:
        print("✗ 모델이 초기화되지 않음")
        return False

    try:
        # 다양한 임계값으로 테스트
        thresholds = [0.3, 0.5, 0.7]

        for threshold in thresholds:
            print(f"\n임계값 {threshold}로 테스트:")

            detections = detector.detect_objects(test_image_path, model, threshold)

            if detections['bboxes']:
                print(f"  ✓ {len(detections['bboxes'])}개 객체 탐지됨")

                # 첫 번째 객체 정보 출력
                bbox = detections['bboxes'][0]
                score = detections['scores'][0]
                print(f"  첫 번째 객체: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                print(f"  신뢰도: {score:.3f}")
            else:
                print(f"  - 탐지된 객체 없음")

        print("✓ 탐지 기능 테스트 완료")
        return True

    except Exception as e:
        print(f"✗ 탐지 테스트 실패: {e}")
        return False


def test_visualization(model, test_image_path):
    """시각화 기능 테스트"""
    print("\n=== 시각화 기능 테스트 ===")

    if model is None:
        print("✗ 모델이 초기화되지 않음")
        return False

    try:
        # 결과 디렉터리 생성
        Path("test_results").mkdir(exist_ok=True)

        # 시각화 테스트
        output_path = "test_results/visualization_test.png"

        success = detector.save_detection_result(
            image_path=test_image_path,
            output_path=output_path,
            model=model,
            score_threshold=0.5
        )

        if success and os.path.exists(output_path):
            print(f"✓ 시각화 결과 저장됨: {output_path}")

            # 결과 이미지 정보 확인
            result_img = cv2.imread(output_path)
            if result_img is not None:
                h, w, c = result_img.shape
                print(f"  결과 이미지 크기: {w}x{h}x{c}")
                print("✓ 시각화 기능 테스트 완료")
                return True
            else:
                print("✗ 결과 이미지 읽기 실패")
                return False
        else:
            print("✗ 시각화 결과 저장 실패")
            return False

    except Exception as e:
        print(f"✗ 시각화 테스트 실패: {e}")
        return False


def test_color_generation():
    """색상 생성 기능 테스트"""
    print("\n=== 색상 생성 테스트 ===")

    try:
        # 다양한 개수로 색상 생성 테스트
        for n in [1, 5, 10, 20]:
            colors = detector.generate_distinct_colors(n)

            if len(colors) == n:
                print(f"✓ {n}개 색상 생성 성공")

                # 첫 번째 색상 확인
                if n > 0:
                    color = colors[0]
                    if len(color) == 3 and all(0 <= c <= 255 for c in color):
                        print(f"  첫 번째 색상: {color}")
                    else:
                        print("✗ 잘못된 색상 형식")
                        return False
            else:
                print(f"✗ {n}개 색상 생성 실패 (실제: {len(colors)}개)")
                return False

        print("✓ 색상 생성 기능 테스트 완료")
        return True

    except Exception as e:
        print(f"✗ 색상 생성 테스트 실패: {e}")
        return False


def run_comprehensive_test():
    """종합 테스트 실행"""
    print("DETR 탐지 기능 종합 테스트 시작")
    print("=" * 60)

    # 테스트 이미지 생성
    test_image_path = create_test_image()

    # 색상 생성 테스트
    color_test = test_color_generation()

    # 모델 초기화 테스트
    model = test_model_initialization()
    model_test = model is not None

    # 탐지 기능 테스트
    detection_test = test_detection_function(model, test_image_path)

    # 시각화 기능 테스트
    visualization_test = test_visualization(model, test_image_path)

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약:")
    print(f"  색상 생성:    {'✓ 통과' if color_test else '✗ 실패'}")
    print(f"  모델 초기화:  {'✓ 통과' if model_test else '✗ 실패'}")
    print(f"  객체 탐지:    {'✓ 통과' if detection_test else '✗ 실패'}")
    print(f"  결과 시각화:  {'✓ 통과' if visualization_test else '✗ 실패'}")

    all_passed = all([color_test, model_test, detection_test, visualization_test])

    if all_passed:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("DETR 탐지 시스템이 정상적으로 작동합니다.")
    else:
        print("\n⚠️  일부 테스트가 실패했습니다.")
        print("문제를 해결한 후 다시 테스트해주세요.")

    print("=" * 60)

    # 정리
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        print("테스트 이미지 정리 완료")

    return all_passed


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='DETR 탐지 기능 테스트')
    parser.add_argument('--image', type=str, default=None,
                       help='테스트할 이미지 경로 (기본값: 자동 생성)')

    args = parser.parse_args()

    if args.image and os.path.exists(args.image):
        print(f"사용자 지정 이미지로 테스트: {args.image}")
        # 사용자 지정 이미지를 사용한 개별 테스트
        model = test_model_initialization()
        if model:
            test_detection_function(model, args.image)
            test_visualization(model, args.image)
    else:
        # 종합 테스트 실행
        run_comprehensive_test()


if __name__ == '__main__':
    main()
