import cv2 as cv
import mediapipe as mp
import time
from handDetector import HandDetector 


def main():
    # time
    pTime = 0
    cTime = 0

    # video cam
    wc = cv.VideoCapture(0)

    # hand detector
    detector = HandDetector()

    while True:
        # Read
        isSuccess, frame = wc.read()

        # work
        detector.processHandImg(img=frame)
        detector.showLandMarks(img=frame)
        lm_list = detector.getLandmarksPosByIndex(img=frame, index=[4, 8])
        
        print(lm_list)
        
        # lm_list = detector.getAllHandLandmarkPos(img=frame)
        # if len(lm_list) != 0:
        #     print(lm_list[8])

        # FPS
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv.putText(img=frame, text=str(int(fps)), org=(
            10, 80), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=3, color=(255, 0, 255), thickness=2)

        # Display
        cv.imshow("webcam", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # cleanup
    wc.release()
    cv.destroyAllWindows()


# dummy code
if __name__ == "__main__":
    main()