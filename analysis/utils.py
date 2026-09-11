import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import msgpack
import eyerec

def resolve_paths(sub, ses, run, file_nb, source_dir_eyetracking, source_dir_fmriprep):

    if source_dir_eyetracking == None and source_dir_fmriprep == None:
            source_dir_eyetracking = os.path.join('source_data', 'eyetracking')
            source_dir_fmriprep = os.path.join('source_data','mariostarts.fmriprep')
    
    pldata_fname = os.path.join(source_dir_eyetracking, sub, ses, f'{sub}_{ses}_{file_nb}.pupil', f'task-mariostars_{run}', '000', 'pupil.pldata')
    confounds_fname = os.path.join(source_dir_fmriprep,sub, ses, 'func', f'{sub}_{ses}_task-mariostars_run-{run[-1]}_part-mag_desc-confounds_timeseries.tsv')
    calibration_data_fname = os.path.join(source_dir_eyetracking, sub, ses, f'{sub}_{ses}_{file_nb}_eyeTrackercalibration-{run[-1]}_calib-data.npz')

    name_files = [pldata_fname, confounds_fname, calibration_data_fname]
    if not all(os.path.isfile(f) for f in name_files):
            #print(f'ERROR with not existing files: subject:{sub}, session:{ses}, file_nbfile number:{file_nb} and run:{run}')
            #print('Please complet the QC file')
            #print(pldata_fname)
            #print(confounds_fname)
            #print(calibration_data_fname)
            return [None]*3
    
    return name_files

def select_run_from_qc(qc_fname):
    # import QC report as pd
    df_qc = pd.read_csv(qc_fname)
    
    # filter row where DO_NOT_USE!=1 & empty_log == False 
    filter_qc = ((df_qc['DO_NOT_USE']!=1)&
                 (df_qc['has_pupil']==True)&
                 (df_qc['has_gaze']==True)&
                  (df_qc['has_eyemovie']==True)) 
    df_filter = df_qc[filter_qc]
    
    # list the ['file_number'] not to use
    # df_without_duplicates = (
        # df_filter
        # .groupby(['subject', 'session', 'run'])
        # .first()
        # .reset_index())

    # tuple of sub-ses-run to use
    df_grouped = df_filter.groupby(['subject', 'session', 'run', 'file_number'])
    run_list = []
    for keys, _ in df_grouped:
        run_list.append(keys)

    return run_list

def extract_data(pldata_fname):
    norm_pos = []
    timestamps = []
    confidences = []
    diameters = []
    with open(pldata_fname, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False, use_list=False)
        for topic, payload in unpacker:
            packet = msgpack.unpackb(payload, raw=False)
            norm_pos.append(packet['norm_pos'])
            timestamps.append(packet['timestamp'])
            confidences.append(packet['confidence'])
            diameters.append(packet['diameter'])
    return np.asarray(norm_pos), np.asarray(timestamps), np.asarray(confidences), np.asarray(diameters)

def extract_calibration_data(fname):
    data = np.load(fname, allow_pickle=True)
    pupils = data['pupils']
    markers = data['markers']

    df_pupil = pd.DataFrame(list(pupils))
    first_timestamp = df_pupil.loc[0, 'timestamp']
    df_pupil['norm_pos_x'] = df_pupil['norm_pos'].apply(lambda x : x[0])
    df_pupil['norm_pos_y'] = df_pupil['norm_pos'].apply(lambda x : x[1])
    df_pupil['timestamp'] = df_pupil['timestamp'] - first_timestamp
    df_pupil = df_pupil.loc[:,['norm_pos_x', 'norm_pos_y', 'timestamp']]

    df_marker = pd.DataFrame(list(markers))
    df_marker['norm_pos_x'] = df_marker['norm_pos'].apply(lambda x : x[0])
    df_marker['norm_pos_y'] = df_marker['norm_pos'].apply(lambda x : x[1])
    df_marker['timestamp'] = df_marker['timestamp'] - first_timestamp
    df_marker = df_marker.drop(columns=['norm_pos', 'screen_pos'])

    df_grp_marker = df_marker.groupby(['norm_pos_x', 'norm_pos_y'], as_index = False).agg(
        start=pd.NamedAgg(column="timestamp", aggfunc="min"),
        end=pd.NamedAgg(column="timestamp", aggfunc="max")
    )

    marker_pos = []
    pupil_mean_pos = []
    order = df_grp_marker['start'].rank(ascending=False)-1
    order = order.astype(int).to_numpy()
    for idx, df_grp in df_grp_marker.groupby(['norm_pos_x', 'norm_pos_y']):
        marker_pos.append(idx)
        marker_filter = ((df_pupil['timestamp']>=df_grp['start'].iloc[0])&
                            (df_pupil['timestamp']<=df_grp['end'].iloc[0]))
        x_mean = df_pupil.loc[marker_filter, 'norm_pos_x'].mean()
        y_mean = df_pupil.loc[marker_filter, 'norm_pos_y'].mean()

        if np.isnan(x_mean) or np.isnan(y_mean):
            print("WARNING: no pupil data for marker:", idx)
            print("Marker interval:", df_grp['start'].iloc[0], df_grp['end'].iloc[0])
            return None, None, None
        else:
            pupil_mean_pos.append([x_mean, y_mean])
    #print(np.asarray(marker_pos), '\n',order, '\n',np.asarray(pupil_mean_pos))
    return np.asarray(marker_pos), order, np.asarray(pupil_mean_pos)