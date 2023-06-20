import cv2

# 動画を読み込む
cap = cv2.VideoCapture('path/to/your/video/file.mp4')

# ビデオファイルからフレームのサイズを取得
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# フレームサイズに合わせてウィンドウを作成
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('video', frame_width, frame_height)

# 動画の初期位置
x_offset = 0
y_offset = 0

# 動画の回転角度（度数法）
angle = 0

# 回転行列を取得
center = (frame_width / 2, frame_height / 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

# 動画が再生中かどうかを示すフラグ
playing = True

# フレームを繰り返し処理
while True:
    # webcamからフレームを読み込む
    ret, frame = cap.read()

    # 動画が終了したら、再生を停止する
    if not ret:
        playing = False

    # 動画の回転と移動を適用
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_width, frame_height))
    moved_frame = rotated_frame[y_offset:y_offset + frame_height, x_offset:x_offset + frame_width]

    # webcamのフレームを読み込み
    _, webcam_frame = cv2.VideoCapture(0).read()

    # webcamのフレームに動画のフレームを上書き
    webcam_frame[y_offset:y_offset + frame_height, x_offset:x_offset + frame_width] = moved_frame

    # 画面に表示
    cv2.imshow('video', webcam_frame)

    # キー操作
    key = cv2.waitKey(1)

    # ESCキーが押されたら終了
    if key == 27:
        break

    # Wキーが押されたら、再生を再開
    if key == ord('w') and not playing:
        cap = cv2.VideoCapture('path/to/your/video/file.mp4')
        playing = True

    # Eキーが押されたら、再生を停止
    if key == ord('e'):
        playing = False

    # 動画の移動と回転を更新
    if playing:
        x_offset += 10
        y_offset += 10
        angle += 1
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

# ウィンドウを破棄
cv2.destroyAllWindows()
