import pdb

def simple_aggregator(all_peaks_cp_track, all_scores_pick_softmax_track, max_cp):
    all_cp = []

    final_cp = []
    final_score = []

    for cp_list, score_list in zip(all_peaks_cp_track, all_scores_pick_softmax_track):
        for cp, score in zip(cp_list, score_list):
            if cp not in final_cp:
                final_cp.append(cp)
                final_score.append(score)

    # find $max_cp most significant change point
    if len(final_cp) >= max_cp:
        final_cp = final_cp[:max_cp]
        final_score = final_score[:max_cp]

    return final_cp, final_score