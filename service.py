import os
from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory
from wct_nst import wct_nst

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))

    for upload in request.files.getlist("content"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        content_destination = "/".join([target, 'content.jpg'])
        print ("Accept incoming file:", filename)
        print ("Save it to:", content_destination)
        upload.save(content_destination)

    for upload in request.files.getlist("style"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        style_destination = "/".join([target, 'style.jpg'])
        print ("Accept incoming file:", filename)
        print ("Save it to:", style_destination)
        upload.save(style_destination)

    save_destination = "/".join([target, 'final.jpg'])

    wct_nst(content_destination, style_destination, save_destination)

    return render_template("complete.html", image_name='final.jpg')

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)