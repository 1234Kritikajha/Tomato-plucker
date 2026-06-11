import argparse

from tomato_detector import TomatoDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether a tomato is ripe or unripe.")
    parser.add_argument("image", help="Path to the tomato image.")
    args = parser.parse_args()

    detector = TomatoDetector()
    result = detector.predict(args.image)

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.2f}%")
    print("Scores     :")
    for label, score in result["scores"].items():
        print(f"  {label}: {score:.2f}%")


if __name__ == "__main__":
    main()
