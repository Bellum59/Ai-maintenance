from flask import Flask, request
# import a utility function for loading Roboflow models
from inference import get_model
# import supervision to visualize our results
import supervision as sv
# import cv2 to helo load our image
import cv2
import json
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
   return "Hello World"

@app.route('/AI',methods=['POST'])
def convertImage():
#    # define the image url to use for inference
   data = request.files['imageUpload']
   data.save(data.filename)
   print(data)
   image = cv2.imread(data.filename)


#    # load a pre-trained yolov8n model
   model = get_model(model_id="f1-car-2023/5")

#    # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
   results = model.infer(image)

#    # load the results into the supervision Detections api
   detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

#    # create supervision annotators
   bounding_box_annotator = sv.BoundingBoxAnnotator()
   label_annotator = sv.LabelAnnotator()

#    # annotate the image with our inference results
   annotated_image = bounding_box_annotator.annotate(
      scene=image, detections=detections)
   annotated_image = label_annotator.annotate(
      scene=annotated_image, detections=detections)

#    # display the image
#    sv.plot_image(annotated_image)
   cv2.imwrite(data.filename, annotated_image)
   return json.dumps(data.filename)
#    return "Page AI"


if __name__ == '__main__':
   app.run("htmaze.devbel.xyz",debug=True,ssl_context=('cert.pem', 'privkey.pem'))