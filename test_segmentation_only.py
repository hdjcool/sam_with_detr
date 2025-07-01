"""
세그멘테이션 전용 기능 테스트 스크립트
- 새로운 visualize_segmentation_only 함수 테스트
- 다양한 투명도 및 옵션 테스트
- 품질 검증
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

try:
    import sam
    import detector
except ImportError as e:
    print(f"모듈 import 실패: {e}")
    sys.exit(1)


def create_test_image_with_clear_objects():
    """명확한 객체가 있는 테스트 이미지 생성"""
    print("🎨 테스트 이미지 생성 중...")

    # 고해상도 테스트 이미지 생성
    height, width = 720, 960

    # 그라데이션 배경 생성 (벡터화)
    y = np.arange(height).reshape(-1, 1)
    x = np.arange(width).reshape(1, -1)
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[..., 0] = np.clip(120 + 60 * np.sin(y / 100), 0, 255).astype(np.uint8)
    background[..., 1] = np.clip(140 + 40 * np.cos(x / 150), 0, 255).astype(np.uint8)
    background[..., 2] = np.clip(160 + 30 * np.sin((y + x) / 200), 0, 255).astype(np.uint8)

    # 명확한 객체들 추가

    # 1. 큰 원형 객체 (person 비슷하게)
    cv2.circle(background, (200, 300), 80, (100, 150, 255), -1)  # 주황색 원
    cv2.circle(background, (200, 200), 40, (100, 150, 255), -1)  # 머리 부분

    # 2. 사각형 객체 (car 비슷하게)
    cv2.rectangle(background, (400, 350), (650, 450), (255, 150, 100), -1)  # 파란색 사각형
    cv2.circle(background, (450, 450), 25, (50, 50, 50), -1)  # 바퀴
    cv2.circle(background, (600, 450), 25, (50, 50, 50), -1)  # 바퀴

    # 3. 타원형 객체 (animal 비슷하게)
    center = (750, 200)
    axes = (60, 40)
    cv2.ellipse(background, center, axes, 0, 0, 360, (150, 255, 150), -1)  # 초록색 타원

    # 4. 복잡한 형태 객체 (chair 비슷하게)
    pts = np.array([
        [100, 500], [150, 500], [150, 600], [140, 600], [140, 650],
        [110, 650], [110, 600], [100, 600]
    ], np.int32)
    cv2.fillPoly(background, [pts], (200, 100, 255))  # 보라색 의자 형태

    # 5. 원형 객체 (bottle 비슷하게)
    cv2.rectangle(background, (500, 100), (530, 80), (255, 255, 100), -1)  # 뚜껑
    cv2.rectangle(background, (490, 80), (540, 180), (255, 255, 100), -1)  # 몸체

    # 텍스처 추가 (더 현실적으로)
    noise = np.random.randint(-20, 20, background.shape, dtype=np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 저장
    test_image_path = "test_segmentation_image.jpg"
    cv2.imwrite(test_image_path, background)

    print(f"✅ 테스트 이미지 생성 완료: {test_image_path}")

    # 테스트용 bbox들 (각 객체를 포함하도록)
    test_bboxes = [
        [120, 120, 280, 380],  # 사람 형태
        [390, 340, 660, 470],  # 자동차 형태
        [690, 160, 810, 240],  # 동물 형태
        [90, 490, 160, 660],   # 의자 형태
        [480, 70, 550, 190]    # 병 형태
    ]

    return test_image_path, test_bboxes


def test_sam_initialization():
    """SAM 모델 초기화 테스트"""
    print("\n🤖 SAM 모델 초기화 테스트")
    print("=" * 30)

    # SAM 사용 가능 여부 확인
    if not sam.check_sam_availability():
        print("❌ segment-anything 패키지가 설치되지 않았습니다.")
        return None

    # 모델 초기화
    predictor = sam.initialize('vit_b', 'auto')

    if predictor:
        print("✅ SAM 모델 초기화 성공")
        return predictor
    else:
        print("❌ SAM 모델 초기화 실패")
        print("모델 파일을 확인하세요: weights/sam_vit_b_01ec64.pth")
        return None


def test_segmentation_inference(predictor, test_image_path, test_bboxes):
    """세그멘테이션 추론 테스트"""
    print("\n🎯 세그멘테이션 추론 테스트")
    print("=" * 25)

    if predictor is None:
        print("❌ SAM predictor를 사용할 수 없습니다.")
        return []

    # SAM inference 수행
    masks = sam.inference_with_boxes(test_image_path, test_bboxes, predictor)

    if masks:
        print(f"✅ {len(masks)}개 마스크 생성 성공")

        # 마스크 품질 확인
        valid_masks = 0
        for i, mask in enumerate(masks):
            mask_area = np.sum(mask)
            if mask_area > 100:  # 최소 크기 확인
                valid_masks += 1
                print(f"  마스크 {i+1}: {mask.shape}, {mask_area} 픽셀")
            else:
                print(f"  마스크 {i+1}: 빈 마스크 또는 너무 작음")

        print(f"✅ 유효한 마스크: {valid_masks}/{len(masks)}개")
        return masks
    else:
        print("❌ 세그멘테이션 추론 실패")
        return []


def test_segmentation_only_visualization(test_image_path, masks):
    """세그멘테이션 전용 시각화 테스트"""
    print("\n🎨 세그멘테이션 전용 시각화 테스트")
    print("=" * 32)

    if not masks:
        print("❌ 테스트할 마스크가 없습니다.")
        return False

    try:
        # 원본 이미지 로딩
        image = cv2.imread(test_image_path)
        if image is None:
            print("❌ 테스트 이미지를 읽을 수 없습니다.")
            return False

        # 결과 디렉토리 생성
        result_dir = "segmentation_test_results"
        Path(result_dir).mkdir(exist_ok=True)

        test_results = []

        # 1. 기본 세그멘테이션 전용 시각화 (alpha=0.6)
        print("1️⃣ 기본 세그멘테이션 시각화 (alpha=0.6)")
        result1 = sam.visualize_segmentation_only(image, masks, alpha=0.6)
        output1 = os.path.join(result_dir, "test_segmentation_default.png")
        cv2.imwrite(output1, result1)
        test_results.append(("기본 세그멘테이션", output1))
        print(f"   ✅ 저장: {output1}")

        # 2. 투명한 세그멘테이션 (alpha=0.3)
        print("2️⃣ 투명한 세그멘테이션 시각화 (alpha=0.3)")
        result2 = sam.visualize_segmentation_only(image, masks, alpha=0.3)
        output2 = os.path.join(result_dir, "test_segmentation_light.png")
        cv2.imwrite(output2, result2)
        test_results.append(("투명한 세그멘테이션", output2))
        print(f"   ✅ 저장: {output2}")

        # 3. 진한 세그멘테이션 (alpha=0.8)
        print("3️⃣ 진한 세그멘테이션 시각화 (alpha=0.8)")
        result3 = sam.visualize_segmentation_only(image, masks, alpha=0.8)
        output3 = os.path.join(result_dir, "test_segmentation_bold.png")
        cv2.imwrite(output3, result3)
        test_results.append(("진한 세그멘테이션", output3))
        print(f"   ✅ 저장: {output3}")

        # 4. 부드러운 가장자리 없는 버전 (use_smooth_edges=False)
        print("4️⃣ 날카로운 가장자리 세그멘테이션")
        result4 = sam.visualize_segmentation_only(image, masks, alpha=0.6, use_smooth_edges=False)
        output4 = os.path.join(result_dir, "test_segmentation_sharp.png")
        cv2.imwrite(output4, result4)
        test_results.append(("날카로운 가장자리", output4))
        print(f"   ✅ 저장: {output4}")

        # 5. save_segmentation_only_result 함수 테스트
        print("5️⃣ 세그멘테이션 전용 저장 함수 테스트")
        output5 = os.path.join(result_dir, "test_save_function.png")
        success = sam.save_segmentation_only_result(
            test_image_path, masks, output5, alpha=0.6, use_smooth_edges=True
        )
        if success:
            test_results.append(("저장 함수 테스트", output5))
            print(f"   ✅ 저장 함수 테스트 성공: {output5}")
        else:
            print("   ❌ 저장 함수 테스트 실패")

        # 결과 요약
        print(f"\n✅ 세그멘테이션 전용 시각화 테스트 완료!")
        print(f"📂 결과 위치: {result_dir}")
        for desc, path in test_results:
            print(f"  - {desc}: {path}")

        return True

    except Exception as e:
        print(f"❌ 세그멘테이션 시각화 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_color_generation():
    """고품질 색상 생성 테스트"""
    print("\n🌈 고품질 색상 생성 테스트")
    print("=" * 25)

    try:
        # 다양한 개수로 색상 생성 테스트
        test_counts = [1, 3, 5, 10, 15, 20]

        for count in test_counts:
            colors = sam.generate_high_quality_colors(count)

            if len(colors) == count:
                print(f"✅ {count}개 색상 생성 성공")
                if count <= 5:  # 적은 개수일 때만 색상 값 출력
                    for i, color in enumerate(colors):
                        print(f"   색상 {i+1}: {color}")
            else:
                print(f"❌ {count}개 색상 생성 실패")
                return False

        print("✅ 고품질 색상 생성 테스트 완료")
        return True

    except Exception as e:
        print(f"❌ 색상 생성 테스트 실패: {e}")
        return False


def compare_with_existing_method(test_image_path, masks):
    """기존 방법과 새로운 방법 비교"""
    print("\n📊 기존 방법과 새로운 방법 비교")
    print("=" * 30)

    if not masks:
        print("❌ 비교할 마스크가 없습니다.")
        return

    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print("❌ 테스트 이미지를 읽을 수 없습니다.")
            return

        result_dir = "comparison_results"
        Path(result_dir).mkdir(exist_ok=True)

        # 1. 기존 방법 (visualize_masks)
        print("1️⃣ 기존 방법으로 시각화")
        old_result = sam.visualize_masks(image, masks, alpha=0.6)
        old_path = os.path.join(result_dir, "comparison_old_method.png")
        cv2.imwrite(old_path, old_result)
        print(f"   ✅ 기존 방법 저장: {old_path}")

        # 2. 새로운 방법 (visualize_segmentation_only)
        print("2️⃣ 새로운 방법으로 시각화")
        new_result = sam.visualize_segmentation_only(image, masks, alpha=0.6)
        new_path = os.path.join(result_dir, "comparison_new_method.png")
        cv2.imwrite(new_path, new_result)
        print(f"   ✅ 새로운 방법 저장: {new_path}")

        # 3. 차이점 분석
        print("3️⃣ 차이점 분석")
        diff = cv2.absdiff(old_result, new_result)
        diff_path = os.path.join(result_dir, "comparison_difference.png")
        cv2.imwrite(diff_path, diff)

        # 평균 차이 계산
        mean_diff = np.mean(diff)
        print(f"   평균 픽셀 차이: {mean_diff:.2f}")
        print(f"   차이 이미지 저장: {diff_path}")

        if mean_diff < 10:
            print("   ✅ 두 방법의 결과가 매우 유사합니다.")
        elif mean_diff < 30:
            print("   ⚠️ 두 방법 간에 약간의 차이가 있습니다.")
        else:
            print("   📊 두 방법 간에 상당한 차이가 있습니다 (새로운 기능이 적용됨).")

        print(f"\n📂 비교 결과 위치: {result_dir}")

    except Exception as e:
        print(f"❌ 비교 테스트 실패: {e}")


def run_comprehensive_test():
    """종합 테스트 실행"""
    print("🧪 세그멘테이션 전용 기능 종합 테스트")
    print("=" * 45)

    # 테스트 결과 추적
    test_results = {
        'image_creation': False,
        'sam_initialization': False,
        'segmentation_inference': False,
        'visualization_test': False,
        'color_generation': False,
        'comparison_test': False
    }

    try:
        # 1. 테스트 이미지 생성
        print("\n🎯 1단계: 테스트 이미지 생성")
        test_image_path, test_bboxes = create_test_image_with_clear_objects()
        test_results['image_creation'] = True

        # 2. SAM 모델 초기화
        print("\n🎯 2단계: SAM 모델 초기화")
        predictor = test_sam_initialization()
        test_results['sam_initialization'] = predictor is not None

        # 3. 색상 생성 테스트 (모델 없이도 가능)
        print("\n🎯 3단계: 색상 생성 테스트")
        test_results['color_generation'] = test_color_generation()

        if predictor is not None:
            # 4. 세그멘테이션 추론
            print("\n🎯 4단계: 세그멘테이션 추론")
            masks = test_segmentation_inference(predictor, test_image_path, test_bboxes)
            test_results['segmentation_inference'] = len(masks) > 0

            if masks:
                # 5. 세그멘테이션 전용 시각화 테스트
                print("\n🎯 5단계: 세그멘테이션 전용 시각화")
                test_results['visualization_test'] = test_segmentation_only_visualization(
                    test_image_path, masks
                )

                # 6. 기존 방법과 비교
                print("\n🎯 6단계: 기존 방법과 비교")
                compare_with_existing_method(test_image_path, masks)
                test_results['comparison_test'] = True

        # 결과 요약
        print("\n" + "=" * 45)
        print("🎯 종합 테스트 결과 요약")
        print("=" * 45)

        passed_tests = sum(test_results.values())
        total_tests = len(test_results)

        for test_name, result in test_results.items():
            status = "✅ 통과" if result else "❌ 실패"
            test_display_name = {
                'image_creation': '테스트 이미지 생성',
                'sam_initialization': 'SAM 모델 초기화',
                'segmentation_inference': '세그멘테이션 추론',
                'visualization_test': '시각화 테스트',
                'color_generation': '색상 생성 테스트',
                'comparison_test': '비교 테스트'
            }
            print(f"  {test_display_name[test_name]}: {status}")

        print(f"\n📊 전체 결과: {passed_tests}/{total_tests} 테스트 통과")

        if passed_tests == total_tests:
            print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
            print("✨ 새로운 세그멘테이션 전용 기능이 정상적으로 작동합니다!")
        elif passed_tests >= total_tests - 1:
            print("✅ 대부분의 테스트가 성공했습니다!")
            print("⚠️ 일부 기능에 문제가 있을 수 있습니다.")
        else:
            print("⚠️ 여러 테스트에서 문제가 발생했습니다.")
            print("🔍 로그를 확인하고 문제를 해결해주세요.")

        # 사용법 안내
        if test_results['sam_initialization'] and test_results['segmentation_inference']:
            print("\n🚀 사용법 안내:")
            print("1. 세그멘테이션 전용 모드로 실행:")
            print("   python main.py --input your_image.jpg --mode segmentation-only")
            print("2. 투명도 조절:")
            print("   python main.py --input your_image.jpg --mode segmentation-only --alpha 0.4")
            print("3. 전체 모드 (모든 결과 포함):")
            print("   python main.py --input your_image.jpg --mode full")

        return passed_tests >= total_tests - 1

    except Exception as e:
        print(f"\n❌ 종합 테스트 중 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files():
    """테스트 파일 정리"""
    print("\n🧹 테스트 파일 정리")
    print("=" * 18)

    # 정리할 파일들
    test_files = [
        "test_segmentation_image.jpg"
    ]

    # 정리할 디렉토리들
    test_dirs = [
        "segmentation_test_results",
        "comparison_results"
    ]

    removed_count = 0

    # 파일 정리
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            removed_count += 1
            print(f"✅ 파일 삭제: {file_path}")

    # 디렉토리 정리
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            import shutil
            try:
                shutil.rmtree(dir_path)
                removed_count += 1
                print(f"✅ 디렉토리 삭제: {dir_path}")
            except Exception as e:
                print(f"⚠️ 디렉토리 삭제 실패 {dir_path}: {e}")

    if removed_count > 0:
        print(f"✅ {removed_count}개 항목 정리 완료")
    else:
        print("ℹ️ 정리할 파일이 없습니다.")


def test_individual_function(function_name):
    """개별 기능 테스트"""
    print(f"🔍 개별 기능 테스트: {function_name}")
    print("=" * 30)

    if function_name == "color":
        return test_color_generation()
    elif function_name == "sam":
        predictor = test_sam_initialization()
        return predictor is not None
    elif function_name == "visualization":
        # 간단한 더미 데이터로 시각화 테스트
        test_image_path, test_bboxes = create_test_image_with_clear_objects()
        predictor = test_sam_initialization()
        if predictor:
            masks = test_segmentation_inference(predictor, test_image_path, test_bboxes)
            return test_segmentation_only_visualization(test_image_path, masks)
        return False
    else:
        print(f"❌ 알 수 없는 테스트 기능: {function_name}")
        return False


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='세그멘테이션 전용 기능 테스트')
    parser.add_argument('--test', choices=['all', 'color', 'sam', 'visualization'],
                       default='all',
                       help='실행할 테스트 종류')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='테스트 파일을 정리하지 않음')

    args = parser.parse_args()

    print("🧪 세그멘테이션 전용 기능 테스트 도구")
    print("=" * 40)

    if args.test == 'all':
        # 종합 테스트 실행
        success = run_comprehensive_test()
    else:
        # 개별 테스트 실행
        success = test_individual_function(args.test)

    # 테스트 파일 정리
    if not args.no_cleanup:
        response = input("\n테스트 파일을 정리하시겠습니까? (y/N): ")
        if response.lower() in ['y', 'yes']:
            cleanup_test_files()
    else:
        print("ℹ️ --no-cleanup 옵션으로 인해 테스트 파일을 유지합니다.")

    # 최종 결과
    print("\n" + "=" * 40)
    if success:
        print("🎉 테스트가 성공적으로 완료되었습니다!")
        print("✨ 세그멘테이션 전용 기능을 사용할 준비가 되었습니다!")
    else:
        print("❌ 테스트에서 문제가 발견되었습니다.")
        print("🔍 오류 메시지를 확인하고 문제를 해결해주세요.")
    print("=" * 40)


if __name__ == "__main__":
    main()
