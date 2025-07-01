"""
업데이트된 SAM 모듈 테스트 스크립트
- 새로운 SAM 기능들 검증
- DETR과의 통합 준비 테스트
"""

import os
import cv2
import numpy as np
from pathlib import Path

try:
    import sam
    print("✅ sam.py 모듈 import 성공")
except ImportError as e:
    print(f"❌ sam.py 모듈 import 실패: {e}")
    exit(1)


def test_sam_availability():
    """SAM 사용 가능 여부 테스트"""
    print("\n🔍 SAM 사용 가능 여부 테스트")
    print("=" * 30)

    available = sam.check_sam_availability()
    if available:
        print("✅ segment-anything 패키지 사용 가능")
    else:
        print("❌ segment-anything 패키지 설치 필요")
        print("설치 방법: pip install segment-anything")

    return available


def test_model_listing():
    """SAM 모델 목록 테스트"""
    print("\n📋 SAM 모델 목록 테스트")
    print("=" * 25)

    sam.list_available_models()


def test_model_initialization():
    """SAM 모델 초기화 테스트"""
    print("\n🤖 SAM 모델 초기화 테스트")
    print("=" * 27)

    # ViT-B 모델로 테스트
    predictor = sam.initialize('vit_b', 'auto')

    if predictor is not None:
        print("✅ SAM 모델 초기화 성공!")
        return predictor
    else:
        print("❌ SAM 모델 초기화 실패")
        print("모델 파일이 다운로드되었는지 확인하세요:")
        print("weights/sam_vit_b_01ec64.pth")
        return None


def create_test_image_and_bbox():
    """테스트용 이미지와 bbox 생성"""
    print("\n🎨 테스트 이미지 및 bbox 생성")
    print("=" * 28)

    # 테스트 이미지 생성
    height, width = 480, 640
    test_image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)

    # 간단한 도형들 추가
    cv2.rectangle(test_image, (100, 100), (250, 200), (255, 0, 0), -1)  # 파란 사각형
    cv2.circle(test_image, (400, 300), 80, (0, 255, 0), -1)  # 초록 원
    cv2.rectangle(test_image, (50, 350), (200, 450), (0, 0, 255), -1)  # 빨간 사각형

    # 테스트 이미지 저장
    test_image_path = "test_sam_image.jpg"
    cv2.imwrite(test_image_path, test_image)
    print(f"✅ 테스트 이미지 생성: {test_image_path}")

    # 테스트용 bounding box들 (각 도형을 포함)
    test_bboxes = [
        [90, 90, 260, 210],    # 파란 사각형
        [320, 220, 480, 380],  # 초록 원
        [40, 340, 210, 460]    # 빨간 사각형
    ]

    print(f"✅ 테스트 bbox 생성: {len(test_bboxes)}개")

    return test_image_path, test_bboxes


def test_sam_inference(predictor, test_image_path, test_bboxes):
    """SAM inference 기능 테스트"""
    print("\n🎯 SAM inference 테스트")
    print("=" * 22)

    if predictor is None:
        print("❌ SAM predictor가 없어서 inference 테스트를 건너뜁니다.")
        return []

    # 다중 bbox inference 테스트
    masks = sam.inference_with_boxes(test_image_path, test_bboxes, predictor)

    if masks:
        print(f"✅ SAM inference 성공: {len(masks)}개 마스크 생성")

        # 마스크 정보 출력
        for i, mask in enumerate(masks):
            mask_area = np.sum(mask)
            print(f"  마스크 {i+1}: {mask.shape} 크기, {mask_area} 픽셀")

        return masks
    else:
        print("❌ SAM inference 실패")
        return []


def test_visualization(test_image_path, masks):
    """시각화 기능 테스트"""
    print("\n🎨 시각화 기능 테스트")
    print("=" * 20)

    if not masks:
        print("❌ 시각화할 마스크가 없습니다.")
        return

    try:
        # 원본 이미지 로딩
        image = cv2.imread(test_image_path)
        if image is None:
            print("❌ 테스트 이미지를 읽을 수 없습니다.")
            return

        # 마스크 시각화
        result_image = sam.visualize_masks(image, masks, alpha=0.6)

        # 결과 저장
        output_path = "test_sam_result.jpg"
        success = cv2.imwrite(output_path, result_image)

        if success:
            print(f"✅ 시각화 결과 저장: {output_path}")
        else:
            print("❌ 시각화 결과 저장 실패")

        # 개별 마스크 저장 테스트
        mask_dir = "test_masks"
        saved_files = sam.save_masks(masks, mask_dir, "test_mask")
        print(f"✅ 개별 마스크 저장: {len(saved_files)}개 파일")

    except Exception as e:
        print(f"❌ 시각화 테스트 실패: {e}")


def test_model_info():
    """모델 정보 기능 테스트"""
    print("\n📊 모델 정보 테스트")
    print("=" * 18)

    for model_type in ['vit_b', 'vit_l', 'vit_h']:
        info = sam.get_model_info(model_type)
        if 'error' not in info:
            status = "사용 가능" if info['available'] else "다운로드 필요"
            print(f"{model_type.upper()}: {info['description']} ({status})")
        else:
            print(f"{model_type.upper()}: {info['error']}")


def cleanup_test_files():
    """테스트 파일 정리"""
    test_files = [
        "test_sam_image.jpg",
        "test_sam_result.jpg"
    ]

    removed_count = 0
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            removed_count += 1

    # 테스트 마스크 디렉토리 정리
    mask_dir = Path("test_masks")
    if mask_dir.exists():
        for mask_file in mask_dir.glob("*.png"):
            mask_file.unlink()
        mask_dir.rmdir()
        removed_count += 1

    if removed_count > 0:
        print(f"\n🧹 테스트 파일 정리 완료: {removed_count}개 항목")


def main():
    """메인 테스트 함수"""
    print("🧪 SAM 모듈 업데이트 테스트 시작")
    print("=" * 35)

    # 1. SAM 사용 가능 여부 확인
    if not test_sam_availability():
        print("\n❌ SAM 패키지가 설치되지 않아 테스트를 중단합니다.")
        return

    # 2. 모델 목록 확인
    test_model_listing()

    # 3. 모델 정보 확인
    test_model_info()

    # 4. 모델 초기화 테스트
    predictor = test_model_initialization()

    # 5. 테스트 데이터 생성
    test_image_path, test_bboxes = create_test_image_and_bbox()

    # 6. SAM inference 테스트
    masks = test_sam_inference(predictor, test_image_path, test_bboxes)

    # 7. 시각화 테스트
    test_visualization(test_image_path, masks)

    # 결과 요약
    print("\n" + "=" * 35)
    print("🎯 테스트 결과 요약")
    print("=" * 35)

    if predictor is not None:
        print("✅ SAM 모델 초기화: 성공")
    else:
        print("❌ SAM 모델 초기화: 실패")

    if masks:
        print(f"✅ SAM inference: 성공 ({len(masks)}개 마스크)")
        print("✅ 시각화: 성공")
        print("\n🎉 SAM 모듈 업데이트가 성공적으로 완료되었습니다!")
        print("이제 DETR과 통합할 준비가 되었습니다.")
    else:
        print("❌ SAM inference: 실패")
        print("\n⚠️ SAM 모델 파일을 다운로드해주세요:")
        print("wget -O weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")

    # 테스트 파일 정리 여부 확인
    response = input("\n테스트 파일을 정리하시겠습니까? (y/N): ")
    if response.lower() in ['y', 'yes']:
        cleanup_test_files()


if __name__ == "__main__":
    main()
