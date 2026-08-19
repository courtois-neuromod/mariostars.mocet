import os
import sys
import mocet

# IMPORT FOR THE FUNCTIONS THAT WILL BE PLACED IN UTILS
import cv2
import eyerec
import numpy as np
import pandas as pd

# FUTURE IMPORT
# from analysis import select_run_from_qc, find_time_between_rec_and_start, creat_data_file, get_pupils_data_from_mp4


def select_run_from_qc(qc_fname):
    # import QC report as pd
    df_qc = pd.read_csv(qc_fname)
    
    # filter row where DO_NOT_USE!=1 & empty_log == False 
    filter_qc = ((df_qc['DO_NOT_USE']!=1)&(df_qc['empty_log']==False)) 
    df_filter = df_qc[filter_qc]
    
    # list the ['file_number'] not to use
    df_without_duplicates = (
        df_filter
        .groupby(['subject', 'session', 'run'])
        .first()
        .reset_index())

    # tuple of sub-ses-run to use
    df_grouped = df_without_duplicates.groupby(['subject', 'session', 'run', 'file_number'])
    run_list = []
    for keys, _ in df_grouped:
        run_list.append(keys)

    return run_list

def find_time_between_rec_and_start(log_fname, run):

    delays = []
    ttl_time = None
    eyetracking_time = None

    with open(log_fname, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 3:
                continue

            timestamp = float(parts[0])
            message = parts[2]

            if "fMRI TTL 0" in message:
                ttl_time = timestamp

            elif "starting eyetracking recording" in message and ttl_time is not None:
                eyetracking_time = timestamp

            if eyetracking_time is not None:
                delay = ttl_time - eyetracking_time
                delays.append(delay)
                ttl_time = None
                eyetracking_time = None
    print (delays)
    return 1, 2
    # Prendre le temps dans la ligne sélectionnée
    # Look for 3549331.1833 	INFO 	stopping eyetracking recording
#3549331.1837 	EXP 	window1: waitBlanking = True
#3549331.2342 	EXP 	task - <class 'src.tasks.videogame.VideoGameMultiLevel'> : task-mariostars_run-01: complete
# To compute the video duration in sec.
    return # strat, duration

def creat_data_file(pldata_fname):
    # Open the pupil.pldata avec load_pladata_file de eyetrackprep
    # Extract the timestamps
    # Compute deltas between timestamps
    # Format a text file under tsv normes, with the right structure to match the code
    # Creat the file 
    return

def get_pupils_data_from_mp4(mp4_fname):
    capture = cv2.VideoCapture(mp4_fname)
    read, frame = capture.read()

    tracker = eyerec.PupilTracker(name='purest')

    df = pd.DataFrame(columns=['diameter_px','width_px','height_px','axisRatio',
                                'center_x','center_y', 'angle_deg', 'confidence'])
    count = 0
    while True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pupil = tracker.detect(capture.get(cv2.CAP_PROP_POS_MSEC), frame)
           
        df.loc[count] = {'diameter_px':np.max([pupil['size']]),
                            'width_px':pupil['size'][0],
                            'height_px':pupil['size'][1],
                            'axisRatio':np.min([pupil['size']])/np.max([pupil['size']]),
                            'center_x':pupil['center'][0],
                            'center_y':pupil['center'][1],
                            'angle_deg': pupil['angle'],
                            'confidence':pupil['confidence']}
            
        read, frame = capture.read()
        if not read:
            break
        count += 1
    data_fname = f'recording-eyetracking_physio_log.csv'
    df.to_csv(data_fname, index=False)
    return df, data_fname

def main(source_data):
    
    qc_fname = os.path.join(source_data, 'neuromod_eyetrack_mariostars_QC.csv')

    if not os.path.isfile(qc_fname):
        qc_fname = os.path.join('source_data', 'neuromod_eyetrack_mariostars_QC.csv')

    run_list = select_run_from_qc(qc_fname)
    
    for sub, ses, run, file_nb in run_list:

        log_fname = os.path.join(source_data, sub, ses, f'{sub}_{ses}_{file_nb}.log')
        pldata_fname = os.path.join(source_data, sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'task-mariostars_{run}', '000', 'pupil.pldata')
        mp4_fname = os.path.join(source_data, sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'task-mariostars_{run}', '000', 'eye0.mp4')

        if not all(os.path.isfile(f) for f in [log_fname, pldata_fname, mp4_fname]):
            print(f'ERROR with not existing files: subject:{sub}, session:{ses}, file_nbfile number:{file_nb} and run:{run}')
            print('Please complet the QC file')
            continue

        start, duration = find_time_between_rec_and_start(log_fname, run)
        #timestamps_fname = creat_data_file(pldata_fname)
        #_, data_fname = get_pupils_data_from_mp4(mp4_fname)

        #pupil_data, pupil_timestamps, pupil_confidence, _ = mocet.utils.clean_viewpoint_data(data_fname,
                                                                             #timestamps_fname,
                                                                             #start=start,
                                                                             #duration=duration)
        
        #confounds_fname = f'{sub}_{ses}_task_{run}_desc-confounds_timeseries.tsv'
        #pupil_data = mocet.apply_mocet(pupil_data, 
                               #motion_params_fname=confounds_fname, 
                               #pupil_confidence=pupil_confidence, 
                               #motion_source='fmriprep',
                               #polynomial_order=3)

        # save output

if __name__ == "__main__":
    main(sys.argv[1])
