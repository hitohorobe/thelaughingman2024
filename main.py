import logging
import cv2
import numpy
from PIL import Image, ImageChops, ImageSequence
import requests

logger = logging.getLogger(__name__)


def download_files(url: str, filepath: str) -> None:
    """
    使用するファイルをダウンロードして保存
    """
    try:
        data = requests.get(url).content
    except Exception as e:
        logger.error(e)
        logger.error(f"{url}からのダウンロードに失敗しました")

    with open(filepath, mode="wb") as f:
        f.write(data)


def convert_gif_anime_alpha_to_black(
    filepath: str, savepath: str = "waraiotoko_converted.gif"
) -> None:
    """
    公式に配布されている笑い男のgifアニメは透過情報を持っているが
    opencvではgifアニメの透過情報を読み込めない
    透過情報の代わりに、透過部分を黒塗りに変換する
    """
    try:
        img = Image.open(filepath)
    except Exception as e:
        logger.error(e)
        logger.error(f"ファイル{filepath}の読み込みに失敗しました")

    converted_image_list = []

    index = 1
    for frame in ImageSequence.Iterator(img):
        rgba = frame.split()
        if len(rgba) == 4:
            rgb = frame.convert("RGB")
            # alpha channel を mask に
            mask = rgba[3].convert(mode="RGB")

            converted_image = ImageChops.darker(rgb, mask)
            converted_image_list.append(converted_image)

            index += 1

    converted_image_list[0].save(
        savepath,
        save_all=True,
        append_images=converted_image_list[1:],
        loop=0,
    )


def main() -> None:
    # 分類器をダウンロード
    face_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    face_cascade_path = "haarcascade_frontalface_default.xml"
    download_files(face_cascade_url, face_cascade_path)

    # 笑い男の素材をダウンロード
    laughing_man_url = "https://thelaughingman2024.jp/assets/img/img_mark_04.gif"
    laughing_man_path = "laughing_man.gif"
    download_files(laughing_man_url, laughing_man_path)

    # 笑い男の素材を合成用に変換する
    converted_laughing_man_path = "converted_laughing_man.gif"
    convert_gif_anime_alpha_to_black(laughing_man_path, converted_laughing_man_path)

    # カメラキャプチャ
    try:
        cap = cv2.VideoCapture(0)
    except Exception as e:
        logger.error(e)
        logger.error("カメラからの映像取得に失敗しました")

    # 笑い男の読み取り
    try:
        gif = cv2.VideoCapture(converted_laughing_man_path)
    except Exception as e:
        logger.error(e)
        logger.error("笑い男の素材の読み込みに失敗しました")

    # 顔検出器
    try:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
    except Exception as e:
        logger.error(e)
        logger.error("検出器ファイルの読み込みに失敗しました")

    while True:
        # カメラから1フレームずつ取得
        ret, frame = cap.read()
        # フレームの反転
        frame = cv2.flip(frame, 1)

        # 笑い男アニメから1フレームずつ取得
        g, icon = gif.read()
        # ループ再生
        if not g:
            gif.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 笑い男アイコンのもともとの縦横比を計算
        orig_height, orig_width = icon.shape[:2]
        aspect_ratio = orig_width / orig_height

        # 顔検出
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facerect = face_cascade.detectMultiScale(
            gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100)
        )

        if len(facerect) > 0:
            # 検出した顔の数だけ処理を行う
            for rect in facerect:
                # 顔サイズに合わせて笑い男アイコンをリサイズ
                icon = cv2.resize(
                    icon, tuple([int(rect[2] * aspect_ratio), int(rect[3])])
                )

                # 透過処理準備
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                icon = cv2.cvtColor(icon, cv2.COLOR_RGB2RGBA)

                # マスクの作成
                icon_gray = cv2.cvtColor(icon, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(icon_gray, 10, 255, cv2.THRESH_BINARY)

                # カメラフレームとリサイズ済み笑い男アイコンのサイズを取得
                height, width = icon.shape[:2]
                frame_height, frame_width = frame.shape[:2]

                # 合成時にはみ出さない場合だけ合成を行う
                if frame_height > rect[1] + height and frame_width > rect[0] + width:
                    # 合成する座標を指定
                    roi = frame[rect[1] : height + rect[1], rect[0] : width + rect[0]]

                    # カメラフレームのうち、顔座標に相当する部分を笑い男アイコンに置き換える
                    # マスクを使い、笑い男アイコン背景の黒い部分を透過させる
                    frame[
                        rect[1] : height + rect[1], rect[0] : width + rect[0]
                    ] = numpy.where(numpy.expand_dims(binary == 255, -1), icon, roi)

        cv2.imshow("result", frame)

        # 何らかのキーが入力されると終了
        k = cv2.waitKey(1)
        if k != -1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
