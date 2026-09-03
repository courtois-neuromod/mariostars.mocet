import cv2
import eyerec
import numpy as np
import pandas as pd
import glob

def select_run_from_qc(filenames):
    df_qc = pd.read_csv('source_data/neuromod_eyetrack_mariostars_QC.csv')
    df_to_use = df_qc[df_qc['DO_NOT_USE']!=1]
    df_without_duplicates = delete_duplicates(df_to_use) 
    mp4_list = [] # filter
    return mp4_list

def delete_duplicates(df):
    df_without_duplicates = (
        df
        .groupby(['sub', 'session', 'run'])
        .first()
        .reset_index())
    return df_without_duplicates

def pupil_extraction():
    path_pattern = '/u*/e*/n*/mariostars/sourcedata/sub-0*/ses-0*/sub-0*_ses-0*_202*****-******.pupil/task-mariostars_run-0*/000/eye0.mp4'
    filenames = set(glob.glob(path_pattern))
    for filename in filenames:
        capture = cv2.VideoCapture(filename)
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
        
        target_filename = f'recording-eyetracking_physio_log.csv'
        df.to_csv(target_filename, index=False) 
        return df
