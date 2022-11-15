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
from googletrans import Translator

# temporary import
import sys
sys.path.append('./ES_extractor')

from ES_extractor.text_feat import DARPA_text_data_read, TextualESCoMPM
from UCP.inference_ucp import detect_CP_tracks
from CP_aggregator import aggregator_core

# from CP_aggregator.segment_core import UniformSegmentator
# from CP_aggregator.aggregator_core import SimpleAggregator

def run_pipeline_single_document(args, single_path_document, file_name):
    # read and pre-process document
    translator = Translator()
    utterance_by_speakers, conversation_list, start_offset_utt_by_speakers = DARPA_text_data_read(single_path_document, translator)

    # init textual feature extractor
    textES = TextualESCoMPM(args)
    
    # extract features by speakers
    
    text_es_signals = textES.extract_ES_emotion_per_track(conversation_list, utterance_by_speakers)
    
    pdb.set_trace()

    # UCP Detector
    all_peaks_track, all_scores_track = detect_CP_tracks(text_es_signals)

    # refined peak track and convert it back to character level index
    all_peaks_track_refined = []

    ma = start_offset_utt_by_speakers

    tmp = {'s1': all_peaks_track[0], 's2': all_peaks_track[1]}

    for each_peak_track_key in tmp:
        each_peak_track = tmp[each_peak_track_key]
        refined_peak_track = []
        for each_cp in each_peak_track:
            if each_cp - 1 >= len(ma[each_peak_track_key]):
                pdb.set_trace()
            refined_peak_track.append(ma[each_peak_track_key][each_cp-1]) # convert utterance level index to character level index 

        all_peaks_track_refined.append(refined_peak_track)

    # Aggregate to find final change point
    final_cp = aggregator_core.simple_aggregator(all_peaks_track_refined)

    res = {'final_cp': list(final_cp)}

    write_fname = file_name.split('.')[0]+'.json'
    path_write = osp.join(args.output_cp_path, write_fname)
    with open(path_write, 'w') as fp:
        json.dump(res, fp, indent=4)

    pdb.set_trace()

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
    
    args.output_cp_path = './output_cp'


    path_inference_document = "/home/nttung/research/Monash_CCU/mini_eval/text_data/en_train"
    
    for file_name in os.listdir(path_inference_document):
        # only read chinese file
        if 'en' in file_name:
            continue

        # debug mode only
        # if file_name != 'M01000GIN_zh.txt':
        #     continue

        print(file_name)

        full_path_document = osp.join(path_inference_document, file_name)
        

        run_pipeline_single_document(args, full_path_document, file_name)  
