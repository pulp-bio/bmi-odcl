% 
% convert_data_biowolf_GUI_2c.m
% 
% Author(s):
% Victor Kartsch <victor.kartsch@iis.ee.ethz.ch>
% Lan Mei <lanmei@student.ethz.ch>
%
% ======================================================================
% 
% Copyright (c) 2022-2024 ETH Zurich.
% 
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
% 
% http://www.apache.org/licenses/LICENSE-2.0
% 
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
% SPDX-License-Identifier: Apache-2.0


function [ExGData] = convert_data_biowolf_GUI_2c(file_including_path, VoltageScale, TimeStampScale)

  

%% Constants and parameters.
lsb_g1 = 7000700;   %BioWOlfs lsbs for all gains.
lsb_g2 = 14000800;
lsb_g3 = 20991300;
lsb_g4 = 27990100;
lsb_g6 = 41994600;
lsb_g8 = 55994200;
lsb_g12 = 83970500;
HEADER_size = 7;
bt_pck_size =32;

%% Open, read and close the file.
fileID = fopen(file_including_path) ;
A = fread(fileID ,inf, 'uint8');
fclose(fileID);

%% Check the input parameters
switch TimeStampScale
    case 's'
        tscaleFactor = 1;
    case 'ms'
        tscaleFactor = 1e3;
    case 'us'
        tscaleFactor = 1e6;
    otherwise
        error('Enter a valid voltage scale. Available Options: [V], [mV], [uV]')
end

switch VoltageScale
    case 'V'
        vscaleFactor = 1;
    case 'mV'
        vscaleFactor = 1e3;
    case 'uV'
        vscaleFactor = 1e6;
    otherwise
        error('Enter a valid voltage scale. Available Options: [V], [mV], [uV]')
end 

%% Read experimental notes.
ExGData = struct;
end_of_data = 0;

for i = 1:length(A)-HEADER_size     %Find header of the experimental notes.
    if ( (A(i)==60) && (A(i+1)==60) && (A(i+2)==62) && (A(i+3)==62) && (A(i+4)==73) && (A(i+5)==69) && (A(i+6)==80) && (A(i+7)==44))
        data_recovered = char(A(i:end))';
        end_of_data = i-1;
        break;
    end
end

if(end_of_data ~= 0)                        % legacy support for older versions.

    Rparams = strsplit(data_recovered,','); %Split the data.

    for inx = 2:length(Rparams)             %Extract and save.
        t_value = Rparams{inx};
        switch t_value(1)
            case 'T'
                ExGData.TestName = t_value(2:end);
            case 'S'
                ExGData.SubjectName = t_value(2:end);
            case 'A'
                ExGData.SubjectAge = str2double(t_value(2:end));
            case 'R'
                ExGData.Remarks = t_value(2:end);
            case 'F'
                ExGData.SampleRate = str2double(t_value(2:end));
            case 'G'
                ExGData.SignalGain = str2double(t_value(2:end));           
        end
    end
else
    warning('The file does not contain information about the experimetal parameters. Hence, conversion of the data to the specified voltage scale is skipped.');
    %end_of_data = length(A);    %legacy support.
    ExGData.SampleRate = 1000;
    ExGData.SignalGain = 12;
    vscaleFactor = 1e6;
    end_of_data = length(A);


end


%% Read data
ADS = vec2mat(A(1:end_of_data)',bt_pck_size);
ch11 = zeros(size(ADS,1), 1);
ch22 = zeros(size(ADS,1), 1);
ch33 = zeros(size(ADS,1), 1);
ch44 = zeros(size(ADS,1), 1);
ch55 = zeros(size(ADS,1), 1);
ch66 = zeros(size(ADS,1), 1);
ch77 = zeros(size(ADS,1), 1);
ch88 = zeros(size(ADS,1), 1);
acc = zeros(size(ADS,1), 3);

for i = 1 : size(ADS,1)         
 ch11(i,1) =  typecast(uint32((0+ADS(i,1)   *256*256*256  +  ADS(i,2)  *256*256  +  ADS(i,3)  *256)),'int32');
 ch22(i,1) =  typecast(uint32((0+ADS(i,4)   *256*256*256  +  ADS(i,5)  *256*256  +  ADS(i,6)  *256)),'int32');
 ch33(i,1) =  typecast(uint32((0+ADS(i,7)   *256*256*256  +  ADS(i,8)   *256*256  +  ADS(i,9) *256)),'int32');
 ch44(i,1) =  typecast(uint32((0+ADS(i,10)  *256*256*256  +  ADS(i,11) *256*256  +  ADS(i,12) *256)),'int32');
 ch55(i,1) =  typecast(uint32((0+ADS(i,13)  *256*256*256  +  ADS(i,14) *256*256  +  ADS(i,15) *256)),'int32');
 ch66(i,1) =  typecast(uint32((0+ADS(i,16)  *256*256*256  +  ADS(i,17) *256*256  +  ADS(i,18) *256)),'int32');
 ch77(i,1) =  typecast(uint32((0+ADS(i,19)  *256*256*256  +  ADS(i,20) *256*256  +  ADS(i,21) *256)),'int32');
 ch88(i,1) =  typecast(uint32((0+ADS(i,22)  *256*256*256  +  ADS(i,23) *256*256  +  ADS(i,24) *256)),'int32');
 acc(i, 1) =  typecast(uint32((0+ADS(i,25)  *256*256*256   +  ADS(i,26)*256*256 )),'int32');
 acc(i, 2) =  typecast(uint32((0+ADS(i,27)  *256*256*256   +  ADS(i,28)*256*256 )),'int32');
 acc(i, 3) =  typecast(uint32((0+ADS(i,29)  *256*256*256   +  ADS(i,30)*256*256 )),'int32');
 %fprintf('Processing %2.2f\r\n', (i/size(ADS,1))*100);
end

%% Convert adc data into volts.
switch ExGData.SignalGain
    case 0
        gain_scaling = 1;           %% legacy support.
    case 1
        gain_scaling = (1/lsb_g1);
    case 2
        gain_scaling = (1/lsb_g2);
    case 3
        gain_scaling = (1/lsb_g3);
    case 4
        gain_scaling = (1/lsb_g4);
    case 6
        gain_scaling = (1/lsb_g6);
    case 8
        gain_scaling = (1/lsb_g8);
    case 12
        gain_scaling = (1/lsb_g12);
end

t_data(:,1) = (double(ch11(1:end,1))/256)*gain_scaling*vscaleFactor;
t_data(:,2) = (double(ch22(1:end,1))/256)*gain_scaling*vscaleFactor;
t_data(:,3) = (double(ch33(1:end,1))/256)*gain_scaling*vscaleFactor;
t_data(:,4) = (double(ch44(1:end,1))/256)*gain_scaling*vscaleFactor;
t_data(:,5) = (double(ch55(1:end,1))/256)*gain_scaling*vscaleFactor;
t_data(:,6) = (double(ch66(1:end,1))/256)*gain_scaling*vscaleFactor;
t_data(:,7) = (double(ch77(1:end,1))/256)*gain_scaling*vscaleFactor;
t_data(:,8) = (double(ch88(1:end,1))/256)*gain_scaling*vscaleFactor;
t_trigger = ADS(:,32);

skipped_samples = 1; % skip the first sample
ExGData.Data = t_data(skipped_samples+1:end, :);    
ExGData.Trigger = t_trigger(skipped_samples+1:end)';
ExGData.timestamp = 0: (tscaleFactor/ExGData.SampleRate) : (max(size(ExGData.Data))-1)/(ExGData.SampleRate/tscaleFactor);
ExGData.ImuData = acc/(255*255);

end


