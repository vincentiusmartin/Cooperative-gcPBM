from coopgcpbm.util import bio as bio
from coopgcpbm.sitespredict import basepred, basemodel
import matplotlib.patches as patches

class KompasPWM(basemodel.BaseModel):
    def __init__(self, kompas, pwm):
        self.pwm = pwm
        self.kompas = kompas

    def predict_sequence(self, seq):
        pred = self.kompas.predict_sequence(seq)
        for k in pred:
            flanklen = (self.pwm.length - k["core_width"])//2
            start, end = k["core_start"]-flanklen, k["core_start"]+k["core_width"]+flanklen
            pwmpred =  self.pwm.predict_sequence(seq[start:end],zero_thres=False)
            if len(pwmpred) == 0:
                return []
            pwmpred = pwmpred[0]
            k["score"] = pwmpred["score"]
            k["pos"] = k["core_mid"]
            k["ori"] = pwmpred["orientation"]
        return pred

    def predict_sequences(self, sequences):
        seqdict = bio.get_seqdict(sequences)
        predictions = {}
        for key in seqdict:
            prediction = self.predict_sequence(seqdict[key])
            predictions[key] = basepred.BasePrediction(seqdict[key], prediction)
        return predictions

    def make_plot_data(self, predictions_dict, color = "cyan"):
        func_dict = {}
        for key in predictions_dict:
            sequence = predictions_dict[key].sequence
            sites_prediction = predictions_dict[key].predictions
            func_pred = []
            # max_y = 1
            for pred in sites_prediction:
                core_rect = patches.Rectangle((pred["core_start"],0),pred["core_width"] - 1,pred["score"],
                                 facecolor=color,alpha=0.9,edgecolor='black')
                func_pred.append({"func": "add_patch",
                                  "args": [core_rect],
                                  "kwargs": {}})
                # if pred["score"] > max_y:
                #     max_y = pred["score"]
            # func_pred.append({"func": "set_ylim",
            #                   "args": [],
            #                   "kwargs": {"top":max_y+1}})
            func_dict[key] = {"sequence": sequence,
                              "plt": func_pred}
        return func_dict
