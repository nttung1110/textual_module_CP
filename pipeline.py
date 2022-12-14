import os 
import numpy as np 
import cv2
import pdb
import bbox_visualizer as bbv
import torch
import json
import os.path as osp
import pdb

from dotmap import DotMap
from datetime import datetime
from googletrans import Translator

# temporary import
import sys
sys.path.append('./ES_extractor')

from ES_extractor.text_feat import DARPA_text_data_read, TextualESCoMPM
from UCP.inference_ucp import detect_CP_tracks
from CP_aggregator import aggregator_core

# from CP_aggregator.segment_core import UniformSegmentator
# from CP_aggregator.aggregator_core import SimpleAggregator

def run_pipeline_single_document(args, single_path_document, textES, file_name):
    # read and pre-process document
    translator = Translator()
    utterance_by_speakers, conversation_list, start_offset_utt_by_speakers = DARPA_text_data_read(single_path_document, translator)

    start = datetime.now()
    
    # extract features by speakers
    softmax = torch.nn.Softmax(dim=1)
    
    text_es_signals = textES.extract_ES_emotion_per_track(conversation_list, utterance_by_speakers)

    no_cp_confirm = False

    if text_es_signals[0].shape[0] < args.len_utt_tracks or text_es_signals[1].shape[0] < args.len_utt_tracks:
        # short utterance => no change point 
        no_cp_confirm = True
        final_cp = []
        res_score = []

    if no_cp_confirm is False:
        # UCP Detector
        all_peaks_track, all_scores_track = detect_CP_tracks(text_es_signals)

        # refined peak track and convert it back to character level index
        all_peaks_track_refined = []
        all_scores_pick_softmax_track = []

        ma = start_offset_utt_by_speakers

        tmp = {'s1': all_peaks_track[0], 's2': all_peaks_track[1]}
        tmp_score = {'s1': all_scores_track[0], 's2': all_scores_track[1]}

        for each_peak_track_key in tmp:
            each_peak_track = tmp[each_peak_track_key]

            refined_peak_track = []
            score_pick_track = []
            for idx, each_cp in enumerate(each_peak_track):
                refined_peak_track.append(ma[each_peak_track_key][each_cp-1]) # convert utterance level index to character level index 
                score_pick_track.append(tmp_score[each_peak_track_key][each_cp])

            sm = softmax(torch.Tensor(np.array([score_pick_track])))

            all_peaks_track_refined.append(refined_peak_track)
            all_scores_pick_softmax_track.append(sm[0].tolist())
        # Aggregate to find final change point
        final_cp, res_score = aggregator_core.simple_aggregator(all_peaks_track_refined, all_scores_pick_softmax_track, args.max_cp)

    time_processing = datetime.now() - start
    res = {'final_cp_result': list(final_cp),
            'final_cp_llr': res_score,
            'type': 'text',
            'time_processing': int(time_processing.total_seconds())}

    write_fname = file_name.split('.')[0]+'.json'
    path_write = osp.join(args.output_cp_path, write_fname)
    with open(path_write, 'w') as fp:
        json.dump(res, fp, indent=4)


if __name__ == "__main__":
    # init argument
    args = DotMap()
    args.pretrained = 'roberta-large'
    args.dyadic = False
    args.cls = 'emotion'
    args.initial = 'pretrained'
    args.freeze = False
    args.sample = 1.0
    args.pretrained_model_ckpt = './pretrained_ckpt/model.bin'
    args.len_utt_tracks = 25
    args.max_cp = 3
    
    args.output_cp_path = './output_cp'
    if os.path.isdir(args.output_cp_path) is False:
        os.mkdir(args.output_cp_path)

    path_inference_document = "/home/nttung/research/Monash_CCU/mini_eval/text_data/en_train"
    
    # init model
    textES = TextualESCoMPM(args)

    for file_name in os.listdir(path_inference_document):
        # only read chinese file
        if 'en' in file_name:
            continue

        # debug mode only
        # if file_name != 'M01000GK4_zh.txt':
        #     continue

        print(file_name)

        full_path_document = osp.join(path_inference_document, file_name)
        

        run_pipeline_single_document(args, full_path_document, textES, file_name)  
