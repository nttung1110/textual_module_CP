def simple_aggregator(all_peaks_cp_track):
    all_cp = []
    for each_cp_track in all_peaks_cp_track:
        all_cp.extend(each_cp_track)

    return set(all_cp)