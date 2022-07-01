import os, json
import numpy as np
from sklearn import metrics

CONFIDENCE = "confidence"
CORRECT_LABEL = "correct_label"

PROBABILITY_THRESHOLDS = []
for x in range(0, 100, 5):
    PROBABILITY_THRESHOLDS.append(x)


def get_just_a_list_of_classes(results):
    classes = []
    for key in results:
        for clas in results[key]["confidence"]:
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
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }
    return final_result


def calc_precision(result):
    tp = result["tp"]
    fp = result["fp"]
    if tp == 0:
        return 0
    return tp / (tp + fp)


def calc_recall(result):
    tp = result["tp"]
    fn = result["fn"]
    if tp == 0:
        return 0
    return tp / (tp + fn)


def calc_f1(precision, recall):
    if (precision * recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def calc_conf_mat(ytrue, ypred, clas_list):
    n = len(clas_list)
    mat_lis = [[0 for _ in range(n)] for _ in range(n)]
    print("ytrue", ytrue)
    print("ypred", ypred)
    for i in range(len(ytrue)):
        mat_lis[ytrue[i]][ypred[i]] += 1

    return np.array(mat_lis), clas_list


def add_macro_precision_recall_f1(pt_results, classes):
    total_precision = sum(
        [pt_results[clas]["metrics"]["precision"] for clas in classes]
    )
    total_recall = sum([pt_results[clas]["metrics"]["recall"] for clas in classes])

    n = len(classes)
    macro_precision = total_precision / n
    macro_recall = total_recall / n
    macro_f1 = calc_f1(macro_precision, macro_recall)
    return {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1}


def get_conf_mat_sum(pt_results, classes, type):
    return sum([pt_results[clas][type] for clas in classes])


def add_micro_precision_recall_f1(pt_results, classes):
    total_tp = get_conf_mat_sum(pt_results=pt_results, classes=classes, type="tp")
    total_fp = get_conf_mat_sum(pt_results=pt_results, classes=classes, type="fp")
    total_fn = get_conf_mat_sum(pt_results=pt_results, classes=classes, type="fn")

    micro_precision = total_tp / (total_fp + total_tp)
    micro_recall = total_tp / (total_fn + total_tp)
    micro_f1 = calc_f1(micro_precision, micro_recall)

    return {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1}


def calculate_result(results):
    final_result = instantiate_final_result(results)
    print("final_result_structure", final_result)
    # one vs all
    classes = get_just_a_list_of_classes(results)
    for prob_thresh in PROBABILITY_THRESHOLDS:
        for clas in classes:
            c_ypred = []
            c_ytrue = []
            for image in results:
                image_result = results[image]
                correct_label = image_result["correct_label"].lower()
                predicted_label = None 
                if image_result["confidence"][clas] * 100 >= prob_thresh:
                    predicted_label = '+ve'
                else:
                    predicted_label = '-ve'
                c_ypred.append(predicted_label)                    
                c_ytrue.append('+ve' if correct_label == clas else '-ve')

            cconf_mat = metrics.confusion_matrix(y_true = c_ytrue, y_pred = c_ypred, labels = ['+ve', '-ve'])
            final_result[prob_thresh][clas]['tp'] = cconf_mat[0][0]   
            final_result[prob_thresh][clas]['fp'] = cconf_mat[1][0]   
            final_result[prob_thresh][clas]['tn'] = cconf_mat[1][1]   
            final_result[prob_thresh][clas]['fn'] = cconf_mat[0][1]   
            # final_result[prob_thresh][clas]['tp']                 
            cmcp = metrics.classification_report(y_true = c_ytrue, y_pred = c_ypred)
            clas_precision = cmcp['+ve']['precision']
            clas_recall = cmcp['+ve']['recall']
            clas_f1 = cmcp['+ve']['f1-score']
            final_result[prob_thresh][clas]["metrics"] = {
                "precision": clas_precision,
                "recall": clas_recall,
                "f1": clas_f1,
            }
        
        final_result[prob_thresh]["metrics"] = {}
        final_result[prob_thresh]["metrics"][
            "macro"
        ] = add_macro_precision_recall_f1(final_result[prob_thresh], classes)
        final_result[prob_thresh]["metrics"][
            "micro"
        ] = add_micro_precision_recall_f1(final_result[prob_thresh], classes)

    ## create (ypred ytrue) based on max prob and calc macro and micro precision
    ypred = []
    ytrue = []

    ## to numeralize the classes
    clas_list = classes.copy()
    clas_list.sort()

    for image_res in results:
        ytrue.append(results[image_res][CORRECT_LABEL])
        tag_lis = list(results[image_res][CONFIDENCE].items())
        tag_lis.sort(key=lambda x: x[1])
        ypred.append(tag_lis[-1][0])

    conf_mat = metrics.confusion_matrix(ytrue, ypred)
    mcp = metrics.classification_report(y_true = ytrue, y_pred = ypred)
    final_result["accuracy"] = mcp["accuracy"]
    print("mcp macro avg", mcp["macro avg"])
    print("mcp weighted avg", mcp["weighted avg"])

        # print()

        ## plot graphs

    ## global confusion matrix
    print("conf_mat", conf_mat)
    print("final_result", final_result)


if __name__ == "__main__":
    # test_model('testCV', '/Users/nilinswap/forgit/others/haliene/src/scripts/test')
    calculate_result(
        test_json_path={
  "/Users/nilinswap/forgit/others/haliene/src/scripts/test/dust/dust.jpg": {
    "correct_label": "dust",
    "confidence": {
      "healthy": 0.4534231,
      "dust": 0.44819477,
      "degraded": 0.09838209
    }
  },
  "/Users/nilinswap/forgit/others/haliene/src/scripts/test/healthy/big.jpg": {
    "correct_label": "healthy",
    "confidence": {
      "healthy": 0.986611,
      "dust": 0.011537336,
      "degraded": 0.0018516191
    }
  },
  "/Users/nilinswap/forgit/others/haliene/src/scripts/test/healthy/medium.jpg": {
    "correct_label": "healthy",
    "confidence": {
      "healthy": 0.99531484,
      "dust": 0.003514858,
      "degraded": 0.0011702776
    }
  },
  "/Users/nilinswap/forgit/others/haliene/src/scripts/test/degraded/degraded.jpg": {
    "correct_label": "degraded",
    "confidence": {
      "dust": 0.733405,
      "degraded": 0.23180486,
      "healthy": 0.034790188
    }
  },
  "/Users/nilinswap/forgit/others/haliene/src/scripts/test/degraded/small.jpg": {
    "correct_label": "degraded",
    "confidence": {
      "degraded": 0.8034252,
      "dust": 0.15128472,
      "healthy": 0.04529012
    }
  }
}

    )


## Parameters should be clubbed in a dictionary
## do something with confusion matrix
