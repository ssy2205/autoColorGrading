import argparse
import cv2
import os
from vectorInfer import infer_parameters, apply_correction_with_strength

def main():
    """
    로컬에서 이미지 색상 보정을 실행하는 메인 함수.
    """
    parser = argparse.ArgumentParser(description="입력 이미지에 대상 이미지의 색상 스타일을 적용합니다.")
    parser.add_argument("input_image", type=str, help="색상 보정을 적용할 원본 이미지 경로")
    parser.add_argument("target_image", type=str, help="색상 스타일의 기준이 될 대상 이미지 경로")
    parser.add_argument("output_image", type=str, help="결과 이미지를 저장할 경로")
    parser.add_argument("--strength", type=float, default=0.5, help="보정 강도 (기본값: 0.5)")

    args = parser.parse_args()

    # 입력 파일 존재 여부 확인
    if not os.path.exists(args.input_image):
        print(f"오류: 입력 이미지 파일을 찾을 수 없습니다: {args.input_image}")
        return
    if not os.path.exists(args.target_image):
        print(f"오류: 대상 이미지 파일을 찾을 수 없습니다: {args.target_image}")
        return

    print("이미지 처리을 시작합니다...")

    # 1. 파라미터 추론
    predicted_params = infer_parameters(args.input_image, args.target_image)

    # 2. 추론된 파라미터와 강도를 사용하여 색상 보정 적용
    corrected_img = apply_correction_with_strength(
        args.input_image,
        predicted_params,
        args.strength
    )

    # 3. 결과 이미지 저장
    cv2.imwrite(args.output_image, corrected_img)

    print(f"처리가 완료되었습니다. 결과 이미지가 다음 경로에 저장되었습니다: {args.output_image}")

if __name__ == "__main__":
    main()
