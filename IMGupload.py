import os
import io
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from keras.models import load_model
from werkzeug import secure_filename
app = Flask(__name__)

label_names = ['リンゴ', 'トマト']
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']

        # 許容ファイル確認
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
        else:
            return ''' <p>許可されていない拡張子です</p> '''

        # 画像読み込み
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # サイズ変形
        raw_img = cv2.resize(img, (IMAGE_WIDTH, int(IMAGE_WIDTH * img.shape[0] / img.shape[1])))
        raw_img_min = cv2.resize(img, dsize=(100, 100))
        # 分類モデル読み込みと予測
        model = load_model('C:/Users/itaku/PycharmProjects/GenerationClass/AppleTomato_model.h5')
        raw_img_std = np.array(raw_img_min)/255
        raw_img_std = raw_img_std.reshape(-1, 100, 100, 3)
        prediction = model.predict(raw_img_std)
        predicted_label = np.argmax(prediction)
        predict_result = label_names[predicted_label]
        predict_ratio = 100*np.max(prediction)

        # 画像を保存する
        raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_'+filename)
        cv2.imwrite(raw_img_url, raw_img)

        return render_template('index.html',
                               raw_img_url=raw_img_url,
                               predict_ratio=predict_ratio,
                               predict_result=predict_result)

    else:
        return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.debug = True
    app.run()
