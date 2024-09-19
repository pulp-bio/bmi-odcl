# 
# read_binary_data.py
# 
# Author(s):
# Victor Kartsch <victor.kartsch@iis.ee.ethz.ch>
# Lan Mei <lanmei@student.ethz.ch>
# 
# Copyright (c) 2024 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

""" reads a single trial of eeg data from different devices"""

import numpy as np

def get_data(filename, device, only_eeg_channels=True):
    """
    This function reads out a csv file of recorded EEG data.

    Parameters
    ----------
    filename : string
        The filename/filepath.
    device : string
        The device the data was recorded with.
        Possible arguments are: "unicorn", "biowolf", "mindrove_arc", "mindrove_strip".
    only_eeg_channels : TYPE, optional
        Only returns channels where EEG was recorded. The default is True.

    Returns
    -------
    A numpy array in the form channels x samples

    """
    
    #assert that device argument is viable
    possible_devices = ["unicorn","biowolf","mindrove_arc","mindrove_strip"]
    devices =" -- ".join(possible_devices)
    assert(device in possible_devices), "This is not a supported device. Possible Devices : "+devices
    
    #for every device: get data from csv and then cut irrelevant channels
    if(device=="unicorn"):
        
        X = np.genfromtxt(filename, delimiter=",", skip_header=1)
        if(only_eeg_channels):
            X = X[:,0:8]
            
    elif(device=="biowolf"):
        
        #reading from .csv that was converted as last year, only eeg channels measured
        X = np.genfromtxt(filename ,delimiter=",", skip_header=1)
    
    elif(device=="mindrove_arc"):
        
        X = np.genfromtxt(filename, delimiter=";", skip_header=2)
        if(only_eeg_channels):
            X = X[:,0:6]
            
    elif(device=="mindrove_strip"):
        
        X = np.genfromtxt(filename, delimiter=";", skip_header=2)
        if(only_eeg_channels):
            X = X[:,0:4]
            
    return np.transpose(X)

def convert_biowolf_bin2csv(filename,VoltageScale='uV',TimeStampScale='s'):
    """
    Converts a binary biowolf output file to csv. 

    Parameters
    ----------
    filename : string
        The filename/filepath.
    VoltageScale : TYPE, optional
        The voltage scale from the Biowolf output file.
        Possible arguments are: "V", "mV", "uV".
    TimeStampScale : TYPE, optional
        The timestamp scale from the Biowolf output file.
        Possible arguments are: "s", "ms", "us".

    """
    # Constants and parameters
    lsb_g1 = 7000700
    lsb_g2 = 14000800
    lsb_g3 = 20991300
    lsb_g4 = 27990100
    lsb_g6 = 41994600
    lsb_g8 = 55994200
    lsb_g12 = 83970500
    HEADER_SIZE=7
    bt_pck_size=32

    # Open and read the file
    with open(filename, "rb") as f:
        bytes_read = f.read()

    #Check the input parameters
    voltageScale_dict={
        'V':1,
        'mV':1e3,
        'uV':1e6
    }

    timeStamp_dict={
        's':1,
        'ms':1e3,
        'us':1e6
    }
    vscaleFactor=voltageScale_dict[VoltageScale]
    tscaleFactor=timeStamp_dict[TimeStampScale]

    # Read experimental notes.
    end_of_data=0
    data_recovered=[]
    for ind in range(len(bytes_read)-HEADER_SIZE):
        if  bytes_read[ind]==60 and \
            bytes_read[ind+1]==60 and \
            bytes_read[ind+2]==62 and \
            bytes_read[ind+3]==62 and \
            bytes_read[ind+4]==73 and \
            bytes_read[ind+5]==69 and \
            bytes_read[ind+6]==80 and \
            bytes_read[ind+7]==44:
            data_recovered = bytes_read[ind:].decode("utf-8").split(',') # Convert byte to string then split by commas
            end_of_data=ind-1
    if data_recovered:
        test_name=data_recovered[1][1:]
        subject_name=data_recovered[2][1:]
        try: 
            subject_age=float(data_recovered[3][1:])
        except ValueError:
            subject_age=0
        remarks=data_recovered[4][1:]
        sample_rate=float(data_recovered[5][1:])
        signal_gain=float(data_recovered[6][1:])
    else:
        print('The file does not contain information about the experimental parameters. Hence, conversion of the data to the specified voltage scale is skipped.')
        end_of_data=len(bytes_read)
        sample_rate=0.0
        signal_gain=0.0
        vscaleFactor=1

    # Read data
    def toSigned32(n):
        n = n & 0xffffffff
        return (n ^ 0x80000000) - 0x80000000

    nb_channels=8
    ads=np.array([x for x in bytes_read])
    ads=np.resize(ads,(len(bytes_read[:end_of_data])//bt_pck_size,bt_pck_size))
    channels_array = np.empty((len(bytes_read[:end_of_data])//bt_pck_size,nb_channels))
    for i in range(ads.shape[0]):
        for chan in range(nb_channels):
            channels_array[i,chan]=toSigned32(ads[i,chan*3]*256**3+ads[i,chan*3+1]*256**2+ads[i,chan*3+2]*256**1)

    # Convert adc data into volts
    scaling_dict={
        0:1,
        1:1/lsb_g1,
        2:1/lsb_g2,
        3:1/lsb_g3,
        4:1/lsb_g4,
        6:1/lsb_g6,
        8:1/lsb_g8,
        12:1/lsb_g12
    }
    gain_scaling=scaling_dict[signal_gain]
    channels_array=channels_array/256*gain_scaling*vscaleFactor
    t_trigger=ads[:,31]

    skipped_samples=1
    X=channels_array[skipped_samples:]
    t_trigger=t_trigger[skipped_samples:].reshape(-1,1)
    X_save = np.concatenate((X, t_trigger), axis=1)
    if sample_rate>0:
        step_size=tscaleFactor/sample_rate
        end_time=(channels_array.shape[0]-1)*step_size
        timestamp=np.arange(0,end_time,step_size)
    else:
        timestamp=0

    # Create header
    header=''
    for channel_nb in range(1,9):
        header+=f'EEG{channel_nb},'
    #     print(channel_nb)
    header+=f'Trigger,'
    header=header[:-1]

    #Save as csv
    np.savetxt(fname=f'{filename[:-4]}_fromPy.csv',comments='',X= X_save,header=header, delimiter=",")            
        

