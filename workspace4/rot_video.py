import cv2
import numpy as np

# フォントとテキストの設定
font = cv2.FONT_HERSHEY_SIMPLEX
video_dir ="video_concat_dir"
text = "kanji1"
font_scale = 3
font_thickness = 2
back_color = (128, 0, 0) 

# 動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_dir +"/"+ text + '.mp4', fourcc, 20.0, (640, 480))

# 背景を持つ画像を生成
text_img = np.zeros((480, 640, 3), dtype=np.uint8)
back = np.zeros_like(text_img)
back[:, :, :] = back_color  # 背景に設定

# テキストを描画
(text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
text_x = int((640 - text_width) / 2)
text_y = int((480 + text_height) / 2)
cv2.putText(text_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

# 動画のフレーム数分ループ
for i in range(360):
    # 回転角度を計算
    angle = i * np.pi / 180

    # 回転行列を生成
    rotation_matrix = cv2.getRotationMatrix2D((text_x + text_width / 2, text_y - text_height / 2), i, 1.0)

    # テキストを回転
    rotated_text = cv2.warpAffine(text_img, rotation_matrix, (640, 480))
    rotated_text += back
    
    # 動画にフレームを追加
    out.write(rotated_text)

# 動画をリリース
out.release()
