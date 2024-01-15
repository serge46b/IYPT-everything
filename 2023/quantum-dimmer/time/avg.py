import cv2
import numpy as np
import easygui
COLOR=(255, 100, 100)
WINDOW_FLAG = cv2.WINDOW_NORMAL

def ask_for_poi(frame: np.ndarray, msg: str = 'Select the polygon of interest'):
    def onClick(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            poi.append([x, y])

    poi = []
    frame = frame.copy()
    cv2.namedWindow("frame", WINDOW_FLAG)

    while len(poi) < 4:
        # Draw all points on the frame
        if poi:
            cv2.circle(frame, tuple(poi[-1]), 5, COLOR, -1)
        cv2.putText(frame, msg+': click on 4 points', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onClick)
        cv2.waitKey(1)
    cv2.circle(frame, tuple(poi[-1]), 5, COLOR, -1)
    cv2.putText(frame, 'Please wait...', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    ctr_poi = np.array(poi).reshape((-1, 1, 2)).astype(np.int32)
    poi_mask = np.zeros(frame.shape[0:2]).astype(bool)
    
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if cv2.pointPolygonTest(ctr_poi, (i, j), False) >= 0:
                poi_mask[j, i] = True
    
    frame[poi_mask] = 0.6 * frame[poi_mask] + 0.4 * np.array([255,255,255])
    cv2.putText(frame, 'Press any key to continue', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyWindow('frame')
    return poi_mask
    
def get_bg_lightness(frame: np.ndarray):
    # frame copy to grayscale
    poi_mask = ask_for_poi(frame.copy(), 'Select the background sample')

    gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    return np.mean(gray[poi_mask])

def get_base_frame(cap: cv2.VideoCapture):
    # User presses arrowkeys to select the frame and confirms with enter
    success, frame = cap.read()
    cv2.namedWindow("frame", WINDOW_FLAG)
    while True:
        if not success:
            raise ValueError("Try again. You reached the end of the video.")
        displ = frame.copy()
        cv2.putText(displ, 'Select the base frame. -> and <- to nav, Enter to select', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)
        cv2.imshow('frame', displ)
        key = cv2.waitKey(0)
        if key == 13: # Enter
            break
        elif key == 81: # Left arrow
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2)
            success, frame = cap.read()
        elif key == 83: # Right arrow
            success, frame = cap.read()
    cv2.destroyWindow('frame')
    return frame

def main():
    # Load video and try reading
    video_path = easygui.fileopenbox("Select video")
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success:
        raise ValueError("Can't open video file")

    # Select the frame for poi selection
    base_frame = get_base_frame(cap)

    poi_mask = ask_for_poi(base_frame, 'Select the polygon containing the flame shadow')
    bg_lightness = get_bg_lightness(base_frame)

    cv2.namedWindow("shadow", WINDOW_FLAG)
    #cv2.namedWindow("grey", WINDOW_FLAG)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    results = []
    while success:
        success, frame = cap.read()
        if not success:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        grey = grey - bg_lightness
        #cv2.imshow('grey', -grey)
        
        shadow_mask = np.bitwise_and(grey < 0, poi_mask)    # Where the image is darker than bg and inside the polygon 
        shadow = grey[shadow_mask]

        frame[shadow_mask] = 0.7 * frame[shadow_mask] + 0.3 * np.array([0,0,255])
        cv2.imshow('shadow', frame)
        cv2.waitKey(1)

        results.append(np.mean(shadow))

    print(video_path.split("/")[-1].split(".")[0], - np.mean(np.array(results)) / bg_lightness)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()