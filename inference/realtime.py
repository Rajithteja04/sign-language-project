import argparse
import cv2
from features.mediapipe_extractor import MediaPipeExtractor
from models.transformer import correct_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="how2sign")
    _ = parser.parse_args()

    cap = cv2.VideoCapture(0)
    extractor = MediaPipeExtractor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _feats = extractor.extract(frame)
        # TODO: send feats to LSTM and get predicted gloss
        predicted_gloss = "HELLO"  # placeholder
        corrected = correct_text(predicted_gloss)

        cv2.putText(frame, corrected, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Sign Language Translation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
