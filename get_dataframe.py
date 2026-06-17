import cv2
import pandas as pd
from paddleocr import PaddleOCR


def get_dataframe(match_path, output_csv="file1.csv"):
    dataframe = {
        "Timestamp": [],
        "Team1": [],
        "Team2": [],
        "Score1": [],
        "Score2": []
    }

    cap = cv2.VideoCapture(match_path)
    i = 0
    frame_skip = 200

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    while cap.isOpened():
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = timestamp // 1000
        ret, frame = cap.read()
        if not ret:
            break
        if i > frame_skip - 1:
            cropped_frame = frame[40:65, 110:313]
            try:
                result = ocr.ocr(cropped_frame, cls=True, det=False)
            except Exception as e:
                print("Error during OCR:", e)
                i = 0
                continue
            for line in result:
                for text, conf in line:
                    if conf > 0.90:
                        main_text = text.strip()
                        break
                else:
                    continue
                if main_text[:3] != ' ':
                    dataframe['Team1'].append(main_text[:3])
                if main_text[4:5] != ' ':
                    dataframe['Score2'].append(main_text[4:5])
                if main_text[5:] != ' ':
                    dataframe['Team2'].append(main_text[5:])
                if main_text[3] != ' ':
                    dataframe['Score1'].append(main_text[3:4])
                dataframe['Timestamp'].append(timestamp)
            i = 0
            print('------GENERATING DATA FRAME------')
            continue
        i += 1

    df_final = pd.DataFrame.from_dict(dataframe)
    cap.release()
    cv2.destroyAllWindows()

    df_final.to_csv(output_csv, index=False)
    return output_csv
