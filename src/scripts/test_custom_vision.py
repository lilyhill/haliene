from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com/"
PREDICTION_KEY = "0ca6cebdbcb14aefabde4b16dd87e646" # same as training. not sure what to do 260a7de5f4ee4b90ad296ed68cc4505a
PREDICTION_RESOURCE_ID = "/subscriptions/78dec634-b052-4b54-b638-b7b0ea5f9a62/resourceGroups/testCVResourceGroup/providers/Microsoft.CognitiveServices/accounts/testCVResource"
publish_iteration_name = "testingAPICall"
project_name = "testCV"

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)



project_id = 'd5bcad34-5ace-4df6-a80b-8f93857471f7'
base_image_location = '.' #inside labelled_sorted

all_image_paths = []
curr_path = os.getcwd()
true_count = 0
total_count = 0
for dir_ in os.listdir(base_image_location):
    label_dir = os.path.join(curr_path, dir_)
    if not os.path.isdir(label_dir):
        continue
    for image in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image)
        label = dir_
        total_count += 1
        with open(image_path, "rb") as image_contents:
            results = predictor.classify_image(
                project_id, publish_iteration_name, image_contents.read())

            # Display the results.
            # print("predictions", results.prediction.tag_name, prediction.probability)
            tag_lis = []

            for prediction in results.predictions:
                print("\t" + prediction.tag_name +
                    ": {0:.2f}%".format(prediction.probability * 100))
                tag_lis.append((prediction.tag_name, prediction.probability))

            tag_lis.sort(key=lambda x: x[1])
            correct_label = tag_lis[-1][0]
            if correct_label.lower() != label.lower():
                print("image", image_path, correct_label, label)
            true_count += (correct_label.lower() == label.lower())


print("precision", true_count/total_count)
