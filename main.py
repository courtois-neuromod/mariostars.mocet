import cv2
import eyerec
import numpy as np
import pandas as pd
import glob

def select_run_from_qc(mp4_list):
    # import QC report as pd
    # filter row where ['DO_NOT_USE']==1
    # list the ['file_number'] not to use
    # filter the mp4_list
    mp4_list = [] # filter
    return mp4_list

def main():
    print("Hello from mariostars-mocet!")
    the_filename = '/u*/e*/n*/mariostars/sourcedata/sub-01/ses-001/sub-01_ses-001_20220601-181020.pupil/task-mariostars_run-01/000/eye0.mp4'
    mp4_list = glob.glob(the_filename)
    for filename in mp4_list: 
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
            print('okey dockey')
            df.to_csv(target_filename, index=False) # update the path to the output folders

'''
The confounds file contains the following 6-DoF motion parameters:
trans_x, trans_y, trans_z: Translation in the x, y, and z directions.
rot_x, rot_y, rot_z: Rotation around the x, y, and z axes.

see also:
Applying MoCET for Avotec/Arrington system
Nonlinear MoCET variants

but it seems prety easy
'''
if __name__ == "__main__":
    main()
