from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os, json
import numpy as np
import matplotlib.pyplot as plt

left = 0
right = 1
plt.xlim(left, right) # Put x axis ticks every 10 units.
plt.ylim(left, right)
plt.autoscale(False)


ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com/"
PREDICTION_KEY = "0ca6cebdbcb14aefabde4b16dd87e646" # same as training. not sure what to do 260a7de5f4ee4b90ad296ed68cc4505a
PREDICTION_RESOURCE_ID = "/subscriptions/78dec634-b052-4b54-b638-b7b0ea5f9a62/resourceGroups/testCVResourceGroup/providers/Microsoft.CognitiveServices/accounts/testCVResource"
CONFIDENCE = 'confidence'
CORRECT_LABEL = 'correct_label'


model_configs = {
    'testCV': {
        'publish_iteration_name': 'testingAPICall',
        'project_id': 'd5bcad34-5ace-4df6-a80b-8f93857471f7',
        'result_file': 'results_testCV'
    },
    'DFI': {
        'publish_iteration_name': 'testDFI',
        'project_id': 'f9a005cd-2a0a-4875-85f4-a1c61a207ddf',
        'result_file': 'results_DFI',
        'test_dir': '/Users/nilinswap/Documents/ml_data/test/labelled_sorted_new'
    },
    'Merged_Augmented': {
        'publish_iteration_name': 'testDFImergedAug',
        'project_id': 'f99e6ad2-c631-4777-ad02-1ad88a20b773',
        'result_file': 'results_Merged_Augmented',
        'test_dir': '/Users/nilinswap/Documents/ml_data/test/labelled_sorted_new_merged'
    },
    'resized_dfi': {
        'publish_iteration_name': 'dfi_resized',
        'project_id': '6bbd3ecb-5929-47a5-a5e0-d12118e42512',
        'result_file': 'results_dfi_resized',
        'test_dir': '/Users/nilinswap/Documents/ml_data/test/labelled_sorted_new'
    }
}

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

def clean_results(raw_results):
    clean_results = {}
    for key, val in raw_results.items():
        clean_results[key.lower()] = val

class TestInstance:
    def __init__(self, project_id, publish_iteration_name):
        self.project_id = project_id
        self.publish_iteration_name = publish_iteration_name
        self.result_map = {}


    def test_image(self, image, label_dir, correct_label):
        image_path = os.path.join(label_dir, image)
        if os.path.splitext(image_path)[1] != '.jpg':
            return 
        self.result_map[image_path] = dict({})
        self.result_map[image_path][CORRECT_LABEL] = correct_label
        self.result_map[image_path][CONFIDENCE] = {}
        with open(image_path, "rb") as image_contents:
            results = predictor.classify_image(
                self.project_id, self.publish_iteration_name, image_contents.read())
            
            
            for prediction in results.predictions:
               self.result_map[image_path][CONFIDENCE][prediction.tag_name.lower()] = prediction.probability

            



def test_model(model: str, curr_path: str):
    project_id = model_configs[model]['project_id']
    publish_iteration_name = model_configs[model]['publish_iteration_name']
    results_file = os.path.join(curr_path, model_configs[model]['result_file'] +'.json')
    count = 0
    with open(results_file, 'w') as result_file_obj:
        test_instance = TestInstance(project_id = project_id, publish_iteration_name= publish_iteration_name)
        for dir_ in os.listdir(curr_path):
            label_dir = os.path.join(curr_path, dir_)
            if not os.path.isdir(label_dir):
                continue
            correct_label = dir_
            for image in os.listdir(label_dir):
                count += 1
                test_instance.test_image(image, label_dir, correct_label)
                print("count", count)
        json.dump(test_instance.result_map, result_file_obj)
    print("test_instance.json", count)

PROBABILITY_THRESHOLDS = []
for x in range(0, 100, 5):
    PROBABILITY_THRESHOLDS.append(x)
PROBABILITY_THRESHOLDS.append(99)

def get_just_a_list_of_classes(results):
    classes = []
    for key in results:
        for clas in results[key]['confidence']:
            classes.append(clas)
        break
    return classes

def instantiate_final_result(results):
    final_result = {}
    for prob_thresh in PROBABILITY_THRESHOLDS:
        final_result[prob_thresh] = {}
        classes = get_just_a_list_of_classes(results)
        for clas in classes:
            final_result[prob_thresh][clas] = {
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'tn': 0,
            }
    return final_result

def calc_precision(result):
    tp = result['tp']
    fp = result['fp']
    if tp == 0:
        return 0
    return tp/(tp+fp)

def calc_recall(result):
    tp = result['tp']
    fn = result['fn']
    if tp == 0:
        return 0
    return tp/(tp+fn)

def calc_f1(precision, recall):
    if (precision*recall) == 0:
        return 0
    return 2*(precision*recall)/(precision + recall)

def calc_conf_mat(ytrue, ypred, clas_list):
    n = len(clas_list)
    mat_lis = [
        [0 for _ in range(n)]
        for _ in range(n)
    ]
    print("ytrue", ytrue)
    print("ypred", ypred)
    for i in range(len(ytrue)):
        mat_lis[ytrue[i]][ypred[i]] += 1
    
    return np.array(mat_lis), clas_list

def add_macro_precision_recall_f1(pt_results, classes):
    total_precision = sum([pt_results[clas]['metrics']['precision'] for clas in classes])
    total_recall = sum([pt_results[clas]['metrics']['recall'] for clas in classes])

    n = len(classes)    
    macro_precision = total_precision/n
    macro_recall = total_recall/n
    macro_f1 = calc_f1(macro_precision, macro_recall)
    return {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1
    }
    
def get_conf_mat_sum(pt_results, classes, type):
    return sum([pt_results[clas][type] for clas in classes])

def add_micro_precision_recall_f1(pt_results, classes):
    total_tp = get_conf_mat_sum(pt_results=pt_results, classes=classes, type='tp')
    total_fp = get_conf_mat_sum(pt_results=pt_results, classes=classes, type='fp')
    total_fn = get_conf_mat_sum(pt_results=pt_results, classes=classes, type='fn')
    if total_tp == 0:
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
        }
    micro_precision = total_tp/(total_fp + total_tp)
    micro_recall = total_tp/(total_fn + total_tp)
    micro_f1 = calc_f1(micro_precision, micro_recall)

    return {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1': micro_f1
    }


def just_draw_graphs(final_test_json_path: str, test_json_path: str):
    with open(final_test_json_path) as tj:
        final_result = json.load(tj)
        classes = []
        with open(test_json_path) as tj:
            results = json.load(tj)
            classes = get_just_a_list_of_classes(results) 
        macro_precisions = []
        macro_recalls = []

        micro_precisions = []
        micro_recalls = []

        clas_metrics = {
            
        }

        for clas in classes:
            clas_metrics[clas] = {
                'precisions': [],
                'recalls': []
            }

        for thresh in final_result:
            if thresh == 'accuracy':
                continue
            print("final_result1", final_result[thresh])
            print("final_result2", final_result[thresh]['metrics'])
            print("final_result3", final_result[thresh]['metrics']['macro'])
            macro_precisions.append(final_result[thresh]['metrics']['macro']['precision'])
            macro_recalls.append(final_result[thresh]['metrics']['macro']['recall'])

            micro_precisions.append(final_result[thresh]['metrics']['micro']['precision'])
            micro_recalls.append(final_result[thresh]['metrics']['micro']['recall'])

            for clas in classes:
                clas_metrics[clas]['precisions'].append(final_result[thresh][clas]['metrics']['precision'])
                clas_metrics[clas]['recalls'].append(final_result[thresh][clas]['metrics']['recall'])
        #create precision recall curve
        draw_graph(macro_recalls, macro_precisions, color='purple', name='Macro')
        draw_graph(micro_recalls, micro_precisions, color='red', name='Micro')

        all_precisions = []
        all_recalls = []
        colors = ['blue', 'green', 'red', 'aqua', 'purple']
        for color, clas in zip(colors, classes):
            all_precisions.append(clas_metrics[clas]['precisions'])
            all_recalls.append(clas_metrics[clas]['recalls'])
            draw_graph(clas_metrics[clas]['recalls'], clas_metrics[clas]['precisions'], color=color, name=clas)
        draw_graphs_in_one(all_precisions=all_precisions, all_recalls=all_recalls, names = classes)



        #display plo
        plt.show()

def calculate_result(test_json_path: str):
    with open(test_json_path) as tj:
        results = json.load(tj)
        final_result = instantiate_final_result(results)
        print("final_result_structure", final_result)
        # one vs all
        classes = get_just_a_list_of_classes(results)
        for prob_thresh in PROBABILITY_THRESHOLDS:
            for clas in classes:
                for image in results:
                    image_result = results[image]
                    correct_label = image_result['correct_label'].lower()   
                    if image_result['confidence'][clas]*100 >= prob_thresh:
                        # I labelled the image as of clas
                        predicted_label = clas.lower()
                        if correct_label == predicted_label:
                            final_result[prob_thresh][predicted_label]['tp'] += 1
                        else:
                            final_result[prob_thresh][predicted_label]['fp'] += 1
                    else:
                        # I labelled the image as of negative clas
                        relevant_class = clas.lower()
                        if correct_label == relevant_class:
                            final_result[prob_thresh][relevant_class]['fn'] += 1
                        else:
                            final_result[prob_thresh][relevant_class]['tn'] += 1
                print("class", clas, final_result[prob_thresh][clas])
                clas_precision = calc_precision(final_result[prob_thresh][clas])
                clas_recall = calc_recall(final_result[prob_thresh][clas])
                clas_f1 = calc_f1(clas_precision, clas_recall)
                final_result[prob_thresh][clas]['metrics'] = {
                    'precision': clas_precision,
                    'recall': clas_recall,
                    'f1': clas_f1,
                }
            final_result[prob_thresh]['metrics'] = {}
            final_result[prob_thresh]['metrics']['macro'] = add_macro_precision_recall_f1(final_result[prob_thresh], classes) 
            final_result[prob_thresh]['metrics']['micro'] = add_micro_precision_recall_f1(final_result[prob_thresh], classes) 
            
        ## create (ypred ytrue) based on max prob and calc macro and micro precision
        ypred = []
        ytrue = []

        ## to numeralize the classes
        clas_list = classes.copy()
        clas_list.sort()

        for image_res in results:
            ytrue.append(clas_list.index(results[image_res][CORRECT_LABEL]))

            tag_lis = list(results[image_res][CONFIDENCE].items())
            tag_lis.sort(key=lambda x: x[1])
            ypred.append(clas_list.index(tag_lis[-1][0]))

        conf_mat = calc_conf_mat(ytrue=ytrue, ypred=ypred, clas_list=clas_list)
        print("conf_mat", conf_mat)
        
        accuracy = sum(np.diag(conf_mat[0]))/sum(sum(conf_mat[0]))
        print("accuracy", accuracy)
        final_result['accuracy'] = accuracy
        # print(metrics.confusion_matrix(ytrue, yprod))

        # # Print the precision and recall, among other metrics
        # print(metrics.classification_report(ytrue, yprod, digits=3))


    ## global confusion matrix 
    name, ext = os.path.splitext(test_json_path)
    new_name = name + '_final' + ext
    with open(new_name, 'w') as f:
        json.dump(final_result, f)
            
                
def draw_graph(precisions, recalls, name, color = 'blue'):
    #create precision recall curve
    _, ax = plt.subplots()
    ax.plot(recalls, precisions, color=color)

    #add axis labels to plot
    ax.set_title(name + ' Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

def draw_graphs_in_one(all_precisions, all_recalls, names, colors = ['blue', 'green', 'red', 'aqua', 'purple']):
    #create precision recall curve
    _, ax = plt.subplots()
    for precisions, recalls, color, name in zip(all_precisions, all_recalls, colors, names):
        ax.plot(recalls, precisions, color=color)

    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

if __name__ == '__main__':
    model_name = 'Merged_Augmented'
    test_dir = model_configs[model_name]['test_dir']
    test_json_path = test_dir + '/' + model_configs[model_name]['result_file'] + '.json' 
    # test_model(model_name, test_dir)
    # calculate_result(test_json_path=test_json_path)

    name, ext = os.path.splitext(test_json_path)
    final_test_json_path = name + '_final' + ext
    just_draw_graphs(final_test_json_path=final_test_json_path, test_json_path=test_json_path)
    

## Parameters should be clubbed in a dictionary
## do something with confusion matrix
