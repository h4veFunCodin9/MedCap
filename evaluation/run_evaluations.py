# encoding: utf-8
# Copyright 2017 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation utility for image Chinese captioning task."""
# __author__ = 'ZhengHe'
# python2.7
# python run_evaluations.py --submit=your_result_json_file --ref=reference_json_file

import sys
import argparse
import os
import pickle

reload(sys)
sys.setdefaultencoding('utf8')
from coco_caption.pycxtools.coco import COCO
from coco_caption.pycxevalcap.eval import COCOEvalCap


def compute_m1(json_predictions_file, reference_file):
    """Compute m1_score"""
    m1_score = {}
    m1_score['error'] = 0
    #try:
    if True:
        coco = COCO(reference_file)
        print 'loaded reference'
        coco_res = coco.loadRes(json_predictions_file)
        print('loaded results')
        # create coco_eval object.
        coco_eval = COCOEvalCap(coco, coco_res)
        print('Eval finished.')
        # evaluate results
        coco_eval.evaluate()
        print('Finished')
    #except Exception:
    #    m1_score['error'] = 1
    if True:#else:
        # print output evaluation scores
        for metric, score in coco_eval.eval.items():
            print '%s: %.3f'%(metric, score)
            m1_score[metric] = score
    return m1_score


def main():
    """The evaluator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-submit", "--submit", type=str, required=True,
                        help=' JSON containing submit sentences.')
    parser.add_argument("-ref", "--ref", type=str,
                        help=' JSON references.')
    parser.add_argument("-store", '--store', type=str,
                        help='metrics score pickle file')
    args = parser.parse_args()

    json_predictions_root = args.submit
    reference_file = args.ref
    pickle_file = args.store

    if os.path.isfile(pickle_file):
        scores = pickle.load(open(pickle_file, 'rb'))
    else:
        scores = {}

    for json_file in os.listdir(json_predictions_root):
        if 'json' not in json_file:
            continue

        iter_num = int(json_file[:-5])
        if iter_num in scores:
            continue

        json_file = os.path.join(json_predictions_root, json_file)
        res = compute_m1(json_file, reference_file)
        scores[iter_num] = res

    print res # print the last metrics scores
    pickle.dump(scores, open(pickle_file, 'wb'))
    print 'evaluation done!'

if __name__ == "__main__":
    main()
