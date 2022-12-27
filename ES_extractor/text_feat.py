import os
import numpy as np
import pdb
import copy
import torch
import xml.etree.ElementTree as ET


from dotmap import DotMap
from tqdm import tqdm

from CoMPM.utils import make_batch_roberta as make_batch
from CoMPM.DARPA_infer_dataset import *
from CoMPM.model import ERC_model
from googletrans import Translator
from transformers import RobertaTokenizer

def DARPA_text_data_read_from_xml(path_xml_ltf, path_xml_psm, translator):
    '''
        Processing DARPA textual dataset to separate utterances by speakers
            + Input: Accepting xml file
            + Output: Return dictionary indexing to specific speaker along with that speaker list of utterances
    '''


    doc_ltf = ET.parse(path_xml_ltf)
    doc_psm = ET.parse(path_xml_psm)

    root_ltf = doc_ltf.getroot()[0][0]
    root_psm = doc_psm.getroot()

    list_cur_utterance = []
    list_start_char_offset = []
    list_total_char = []

    # construct from ltf
    for each_ltf in root_ltf:

        zn_text = each_ltf[0].text
        print(zn_text)
        en_text = translator.translate(zn_text).text 

        list_cur_utterance.append(en_text)
        list_start_char_offset.append(int(each_ltf.get('start_char')))

        total_char = int(each_ltf.get('end_char'))- int(each_ltf.get('start_char')) + 1
        list_total_char.append(total_char)

    # construct from psm
    list_corresponding_sid = []
    list_corresponding_id_seg = []

    run_idx = 0
    for each_psm in root_psm:

        if each_psm.get('type') == "message":
            run_idx += 1

            # find sid 

            found_match_psm = False
            for ii in range(3):
                aa = each_psm[ii]

                if aa.get('name') == 'participant':
                    exact_psm = aa
                    found_match_psm = True
                    break
            
            if found_match_psm:
                s_id = int(exact_psm.get('value'))
                id_seg = each_psm[0].get('value')

                list_corresponding_sid.append(s_id)
                list_corresponding_id_seg.append(id_seg)
    
    # There are some cases missing utterance sid => filter list utterance
    len_s_id = len(list_corresponding_sid)
    list_cur_utterance = list_cur_utterance[:len_s_id]
    list_start_char_offset = list_start_char_offset[:len_s_id]
    list_total_char = list_total_char[:len_s_id]


    # construct text data similarly as DARPA_text_data_read
    text_data = []
    dict_utterance = {'s1': [], 's2': []}
    dict_start_offset = {'s1': [], 's2': []}

    first_speaker = list_corresponding_sid[0]
    for idx in range(len(list_cur_utterance)):
        cur_sid = list_corresponding_sid[idx]
        cur_utterance = list_cur_utterance[idx]
        cur_start_offset_char = list_start_char_offset[idx]

        sample_text = str(cur_sid) + ':' + cur_utterance
        text_data.append(sample_text)

        if cur_sid == first_speaker:
            dict_utterance['s1'].append(cur_utterance)
            dict_start_offset['s1'].append(cur_start_offset_char)
        else:
            dict_utterance['s2'].append(cur_utterance)
            dict_start_offset['s2'].append(cur_start_offset_char)

    return dict_utterance, text_data, dict_start_offset


def DARPA_text_data_read(path, translator):
    '''
        Processing DARPA textual dataset to separate utterances by speakers
            + Input: Only accept txt file at the moment
            + Output: Return dictionary indexing to specific speaker along with that speaker list of utterances
    '''
    
    with open(path, "r") as fp:
        text_data = fp.read()

    text_data = text_data.split("\n")
    if text_data[-1] == "":
        text_data = text_data[:-1] # ignore last empty segment    

    # iterate and separate each segment
    s1 = [] # track utterance of speaker1 through conversation
    s2 = [] # track utterance of speaker2 through conversation

    # translate all text
    for idx, data in enumerate(text_data):
        # translate
        text_data[idx] = translator.translate(text_data[idx]).text

    # preprocess by eliminating auto-reply utterance

    # whoever is the first speaker will be assigned to s1
    first_speaker = None
    conversation_list = []

    running_offset = 0
    start_offset_s1 = []
    start_offset_s2 = []


    for segment in text_data:
        seg_id = int(segment.split(':')[0])
        seg_text = segment.split(':')[1]
        conversation_list.append(seg_text)

        if first_speaker == None:
            first_speaker = seg_id
            start_offset_s1.append(0)
            s1.append(seg_text)

            running_offset += len(seg_text)
            continue

        if seg_id == first_speaker:
            s1.append(seg_text)
            start_offset_s1.append(running_offset)
        else:
            s2.append(seg_text)
            start_offset_s2.append(running_offset)

        running_offset += len(seg_text)

    # also return start offset in character of every utterance
    
    return {'s1': s1, 's2': s2}, text_data, {'s1': start_offset_s1, 's2': start_offset_s2}


class TextualESCoMPM():
    def __init__(self, args):
        self.args = args
        self.init_model()

    
    def init_model(self):
        initial = self.args.initial
        model_type = self.args.pretrained
        freeze = self.args.freeze
        if freeze:
            freeze_type = 'freeze'
        else:
            freeze_type = 'no_freeze' 

        sample = self.args.sample
        if 'gpt2' in model_type:
            last = True
        else:
            last = False


        model_path = self.args.pretrained_model_ckpt
        clsNum = 7 # 7 emotion categories in daily dialog
        self.model = ERC_model(model_type, clsNum, last, freeze, initial)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.cuda()    
        self.model.eval() 
        

    def extract_ES_emotion_per_track(self, conversation_list, dict_speaker_utt):
        
        print("==========Extracting ES from utterance===========")
        # init dataloader for that conversation_list
        dataset = Darpa_raw_loader(conversation_list, 'emotion')


        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)

        list_feat_s1 = []
        list_feat_s2 = []

        count = 0
        list_cur_utterance = {'s1': [], 's2': []}
        dict_feat = {}

        with torch.no_grad():
            for i_batch, data in enumerate(tqdm(test_dataloader)):
                """Prediction"""
                batch_input_tokens, _, batch_speaker_tokens, batch_utterance = data
                batch_input_tokens = batch_input_tokens.cuda()

                pred_logits = self.model(batch_input_tokens, batch_speaker_tokens) # (1, clsNum)

                feat_emotion = torch.nn.Softmax(dim=1)(pred_logits)

                # check whether this feat belongs to which speaker
                cur_utterance = batch_utterance[0]
                if cur_utterance not in dict_feat:
                    dict_feat[cur_utterance] = feat_emotion[0].tolist()

                # if cur_utterance in dict_speaker_utt['s1']:
                #     list_feat_s1.append(feat_emotion[0].tolist())
                #     list_cur_utterance['s1'].append(cur_utterance)
                # else:
                #     list_feat_s2.append(feat_emotion[0].tolist())
                #     list_cur_utterance['s2'].append(cur_utterance)
                
                count += 1

        for each_key_speaker in dict_speaker_utt:
            list_speaker_utterance = dict_speaker_utt[each_key_speaker]

            for each_utt in list_speaker_utterance:
                if each_key_speaker == 's1':
                    list_feat_s1.append(dict_feat[each_utt])
                else:
                    list_feat_s2.append(dict_feat[each_utt])

        feat_s1 = np.array(list_feat_s1)
        feat_s2 = np.array(list_feat_s2)

        all_es_text_feat_tracks = [feat_s1, feat_s2]
        return all_es_text_feat_tracks

if __name__ == "__main__":
    # test args 
    args = DotMap()
    args.pretrained = 'roberta-large'
    args.dyadic = False
    args.cls = 'emotion'
    args.initial = 'pretrained'
    args.freeze = False
    args.sample = 1.0
    args.pretrained_model_ckpt = '/home/nttung/research/Monash_CCU/mini_eval/text_module/src_multistage_approach/ES_extractor/CoMPM/ckpt/models/dailydialog_models/roberta-large/pretrained/no_freeze/emotion/1.0/model.bin'

    path_csv_test = '/home/nttung/research/Monash_CCU/mini_eval/text_data/en_train/M01000G9C_en.txt'
    # utterance_by_speakers, conversation_list = DARPA_text_data_read(path_csv_test)
    # textES = TextualES(args)


    # torch.cuda.empty_cache()
    # textES = TextualESCoMPM(args)

    # textES.extract_ES_emotion_per_track(conversation_list, utterance_by_speakers)

    translator = Translator()
    # test with xml file
    path_xml_ltf = '/home/nttung/research/Monash_CCU/mini_eval/sub_data/text/ltf/M01000EY0.ltf.xml'
    path_xml_psm = '/home/nttung/research/Monash_CCU/mini_eval/sub_data/text/psm/M01000EY0.psm.xml'
    DARPA_text_data_read_from_xml(path_xml_ltf, path_xml_psm, translator)
