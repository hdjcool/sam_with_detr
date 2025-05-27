"""
DETR + SAM 통합 파이프라인 테스트 스크립트
- 전체 파이프라인 동작 검증
- 모델 파일 확인 및 설치 가이드
- 단계별 결과 확인
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

def check_imports():
    """필요한 모듈 import 확인"""
    print("📦 모듈 import 테스트")
    print("=" * 25)

    modules = []

    try:
        import detector
        print("✅ detector.py import 성공")
        modules.append(('detector', detector))
    except ImportError as e:
        print(f"❌ detector.py import 실패: {e}")
        return False

    try:
        import sam
        print("✅ sam.py import 성공")
        modules.append(('sam', sam))
    except ImportError as e:
        print(f"❌ sam.py import 실패: {e}")
        return False

    return modules


def check_dependencies():
    """필요한 패키지 확인"""
    print("\n🔍 패키지 의존성 확인")
    print("=" * 22)

    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV (opencv-python)'),
        ('numpy', 'NumPy'),
        ('mmdet', 'MMDetection'),
        ('segment_anything', 'Segment Anything')
    ]

    missing_packages = []

    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} 설치됨")
        except ImportError:
            print(f"❌ {name} 설치 필요")
            missing_packages.append(name)

    if missing_packages:
        print(f"\n설치 필요한 패키지: {', '.join(missing_packages)}")
        return False

    return True


def check_model_files():
    """모델 파일 존재 확인"""
    print("\n📁 모델 파일 확인")
    print("=" * 18)

    model_files = [
        ("DETR Config", "configs/detr_r50_8xb2-150e_coco.py"),
        ("DETR Checkpoint", "weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"),
        ("SAM Checkpoint", "weights/sam_vit_b_01ec64.pth")
    ]

    missing_files = []

    for name, path in model_files:
        if os.path.exists(path):
            if name.endswith("Checkpoint"):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"✅ {name}: {path} ({size_mb:.1f}MB)")
            else:
                print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (파일 없음)")
            missing_files.append((name, path))

    if missing_files:
        print("\n📥 누락된 파일 다운로드 방법:")
        for name, path in missing_files:
            if "DETR" in name:
                print(f"  {name}: python download_models.py")
            elif "SAM" in name:
                print(f"  {name}: python simple_sam_download.py")
        return False

    return True


def create_test_image():
    """테스트용 이미지 생성"""
    print("\n🎨 테스트 이미지 생성")
    print("=" * 20)

    # 고품질 테스트 이미지 생성
    height, width = 640, 800
    image = np.ones((height, width, 3), dtype=np.uint8) * 240  # 밝은 회색 배경

    # 다양한 객체 추가 (COCO 클래스에 해당)

    # 1. 사람 모양 (person) - 파란색
    cv2.rectangle(image, (50, 200), (150, 500), (255, 100, 100), -1)  # 몸체
    cv2.circle(image, (100, 150), 40, (255, 100, 100), -1)  # 머리

    # 2. 자동차 모양 (car) - 초록색
    cv2.rectangle(image, (300, 400), (500, 500), (100, 255, 100), -1)  # 차체
    cv2.circle(image, (330, 500), 25, (50, 50, 50), -1)  # 바퀴1
    cv2.circle(image, (470, 500), 25, (50, 50, 50), -1)  # 바퀴2

    # 3. 고양이 모양 (cat) - 주황색
    cv2.ellipse(image, (650, 300), (80, 50), 0, 0, 360, (100, 150, 255), -1)  # 몸체
    cv2.circle(image, (650, 250), 35, (100, 150, 255), -1)  # 머리
    # 귀
    pts = np.array([[630, 230], [640, 200], [650, 230]], np.int32)
    cv2.fillPoly(image, [pts], (100, 150, 255))
    pts = np.array([[650, 230], [660, 200], [670, 230]], np.int32)
    cv2.fillPoly(image, [pts], (100, 150, 255))

    # 4. 의자 모양 (chair) - 보라색
    cv2.rectangle(image, (200, 100), (280, 300), (255, 100, 255), -1)  # 등받이
    cv2.rectangle(image, (200, 250), (350, 300), (255, 100, 255), -1)  # 좌석
    cv2.rectangle(image, (200, 300), (220, 400), (255, 100, 255), -1)  # 다리1
    cv2.rectangle(image, (260, 300), (280, 400), (255, 100, 255), -1)  # 다리2
    cv2.rectangle(image, (330, 300), (350, 400), (255, 100, 255), -1)  # 다리3

    # 5. 병 모양 (bottle) - 청록색
    cv2.rectangle(image, (600, 100), (630, 80), (255, 255, 100), -1)  # 뚜껑
    cv2.rectangle(image, (590, 80), (640, 200), (255, 255, 100), -1)  # 병 몸체

    # 테스트 이미지 저장
    test_image_path = "pipeline_test_image.jpg"
    cv2.imwrite(test_image_path, image)

    print(f"✅ 테스트 이미지 생성 완료: {test_image_path}")
    print("  포함 객체: person, car, cat, chair, bottle")

    return test_image_path


def run_pipeline_test(test_image_path, modules):
    """전체 파이프라인 테스트 실행"""
    print(f"\n🚀 DETR + SAM 파이프라인 테스트")
    print("=" * 32)

    detector_module, sam_module = modules[0][1], modules[1][1]

    try:
        # 디바이스 선택
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"🖥️ 사용 디바이스: {device}")

        # 1단계: DETR 모델 초기화
        print("\n1️⃣ DETR 모델 초기화 중...")
        detr_config = "configs/detr_r50_8xb2-150e_coco.py"
        detr_checkpoint = "weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"

        detr_model = detector_module.initialize(detr_config, detr_checkpoint, device)
        if detr_model is None:
            print("❌ DETR 모델 초기화 실패")
            return False
        print("✅ DETR 모델 초기화 완료")

        # 2단계: SAM 모델 초기화
        print("\n2️⃣ SAM 모델 초기화 중...")
        sam_predictor = sam_module.initialize('vit_b', device)
        if sam_predictor is None:
            print("❌ SAM 모델 초기화 실패")
            return False
        print("✅ SAM 모델 초기화 완료")

        # 3단계: DETR 객체 탐지
        print("\n3️⃣ DETR 객체 탐지 수행 중...")
        detections = detector_module.detect_objects(test_image_path, detr_model, 0.3)

        if not detections['bboxes']:
            print("❌ 탐지된 객체가 없습니다.")
            return False

        bboxes = detections['bboxes']
        scores = detections['scores']
        labels = detections.get('labels', [])

        print(f"✅ DETR 탐지 완료: {len(bboxes)}개 객체")
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            if i < len(labels):
                class_name = detector_module.get_class_name(labels[i])
                print(f"  {i+1}. {class_name}: {score:.3f}")
            else:
                print(f"  {i+1}. Object_{i+1}: {score:.3f}")

        # 4단계: SAM 세그멘테이션
        print("\n4️⃣ SAM 세그멘테이션 수행 중...")
        masks = sam_module.inference_with_boxes(test_image_path, bboxes, sam_predictor)

        if not masks:
            print("❌ SAM 세그멘테이션 실패")
            return False

        successful_masks = [mask for mask in masks if mask.any()]
        print(f"✅ SAM 세그멘테이션 완료: {len(successful_masks)}/{len(masks)} 성공")

        # 5단계: 결과 시각화
        print("\n5️⃣ 결과 시각화 중...")

        # 결과 디렉토리 생성
        output_dir = "pipeline_test_results"
        Path(output_dir).mkdir(exist_ok=True)

        # 원본 이미지 로딩
        image = cv2.imread(test_image_path)

        # 다양한 시각화 결과 생성
        results = []

        # DETR 결과만
        detr_vis = detector_module.visualize_detections(image, detections)
        detr_path = os.path.join(output_dir, "test_1_detr_only.png")
        cv2.imwrite(detr_path, detr_vis)
        results.append(("DETR 탐지", detr_path))

        # SAM 마스크만
        mask_vis = sam_module.visualize_masks(image, masks, alpha=0.6)
        mask_path = os.path.join(output_dir, "test_2_sam_only.png")
        cv2.imwrite(mask_path, mask_vis)
        results.append(("SAM 마스크", mask_path))

        # 통합 결과
        combined_vis = create_combined_result(image, bboxes, masks, scores, labels, detector_module)
        combined_path = os.path.join(output_dir, "test_3_combined.png")
        cv2.imwrite(combined_path, combined_vis)
        results.append(("통합 결과", combined_path))

        print("✅ 모든 시각화 완료!")
        for desc, path in results:
            print(f"  - {desc}: {path}")

        return True

    except Exception as e:
        print(f"❌ 파이프라인 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_combined_result(image, bboxes, masks, scores, labels, detector_module):
    """통합 결과 시각화 생성"""
    # 마스크 오버레이
    result = image.copy()

    # 마스크 색상 생성
    colors = detector_module.generate_distinct_colors(len(masks))

    # 마스크 적용
    for mask, color in zip(masks, colors):
        if mask.any():
            colored_mask = np.zeros_like(result)
            colored_mask[mask] = color
            result = cv2.addWeighted(result, 0.7, colored_mask, 0.3, 0)

    # bbox와 라벨 추가
    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        color = colors[i % len(colors)]

        # bbox
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)

        # 라벨
        if i < len(labels):
            class_name = detector_module.get_class_name(labels[i])
            label_text = f"{class_name}: {score:.2f}"
        else:
            label_text = f"Object_{i+1}: {score:.2f}"

        # 텍스트 배경 및 텍스트
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )

        text_x, text_y = x1, y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10

        cv2.rectangle(result, (text_x, text_y - text_height - baseline),
                     (text_x + text_width, text_y + baseline), color, -1)
        cv2.putText(result, label_text, (text_x, text_y - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result


def cleanup_test_files():
    """테스트 파일 정리"""
    print("\n🧹 테스트 파일 정리")
    print("=" * 18)

    # 테스트 이미지 삭제
    test_files = ["pipeline_test_image.jpg"]
    removed_files = 0

    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            removed_files += 1
            print(f"✅ {file} 삭제")

    # 결과 디렉토리는 유지 (사용자가 결과를 확인할 수 있도록)
    results_dir = Path("pipeline_test_results")
    if results_dir.exists():
        result_count = len(list(results_dir.glob("*.png")))
        print(f"📂 결과 파일 유지: pipeline_test_results/ ({result_count}개 파일)")

    if removed_files > 0:
        print(f"✅ {removed_files}개 임시 파일 정리 완료")


def main():
    """메인 테스트 함수"""
    print("🧪 DETR + SAM 통합 파이프라인 테스트")
    print("=" * 40)

    # 1. 모듈 import 확인
    modules = check_imports()
    if not modules:
        print("\n❌ 필수 모듈을 import할 수 없습니다.")
        return False

    # 2. 패키지 의존성 확인
    if not check_dependencies():
        print("\n❌ 필요한 패키지가 설치되지 않았습니다.")
        return False

    # 3. 모델 파일 확인
    if not check_model_files():
        print("\n❌ 필요한 모델 파일이 없습니다.")
        return False

    # 4. 테스트 이미지 생성
    test_image_path = create_test_image()

    # 5. 파이프라인 테스트 실행
    success = run_pipeline_test(test_image_path, modules)

    # 결과 요약
    print("\n" + "=" * 40)
    print("🎯 파이프라인 테스트 결과")
    print("=" * 40)

    if success:
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("✅ DETR 모델 로딩 및 객체 탐지")
        print("✅ SAM 모델 로딩 및 세그멘테이션")
        print("✅ 통합 시각화 및 결과 저장")
        print("\n📂 결과 확인: pipeline_test_results/")
        print("이제 실제 이미지로 파이프라인을 사용할 수 있습니다!")

        print("\n🚀 사용 방법:")
        print("python main.py --input your_image.jpg --threshold 0.5 --mode full")
    else:
        print("❌ 파이프라인 테스트에 실패했습니다.")
        print("위의 오류 메시지를 확인하고 문제를 해결해주세요.")

    # 파일 정리 여부 확인
    response = input("\n임시 테스트 파일을 정리하시겠습니까? (y/N): ")
    if response.lower() in ['y', 'yes']:
        cleanup_test_files()

    return success


if __name__ == "__main__":
    main()
