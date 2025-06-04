from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2
from imageio import imread, imsave
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

img_size = 256

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

# TensorFlow 2.x compatibility for loading the .meta model
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()

# Restore the model
saver = tf.compat.v1.train.import_meta_graph(os.path.join('model', 'model.meta'))
saver.restore(sess, tf.compat.v1.train.latest_checkpoint('model'))

graph = tf.compat.v1.get_default_graph()

# Get tensor placeholders
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Upload files
        non_makeup_file = request.files['non_makeup']
        makeup_file = request.files['makeup']

        non_makeup_filename = secure_filename(non_makeup_file.filename)
        makeup_filename = secure_filename(makeup_file.filename)

        non_makeup_path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + '_' + non_makeup_filename)
        makeup_path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + '_' + makeup_filename)

        non_makeup_file.save(non_makeup_path)
        makeup_file.save(makeup_path)

        # Read and preprocess images
        non_makeup_img = cv2.resize(imread(non_makeup_path), (img_size, img_size))
        makeup_img = cv2.resize(imread(makeup_path), (img_size, img_size))

        X_img = np.expand_dims(preprocess(non_makeup_img), 0)
        Y_img = np.expand_dims(preprocess(makeup_img), 0)

        # Run model (use TensorFlow 2.x session for prediction)
        Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        Xs_ = deprocess(Xs_)
        result_img = np.uint8(Xs_[0] * 255)

        # Combine all images: non-makeup, makeup, and result
        combined = np.hstack([non_makeup_img, makeup_img, result_img])

        # Save the combined image
        output_path = os.path.join(RESULT_FOLDER, 'result_' + str(uuid.uuid4()) + '.jpg')
        imsave(output_path, combined)

        # Return the result to the front end
        return render_template('index.html', result_image=output_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

