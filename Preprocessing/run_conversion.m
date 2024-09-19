% 
% run_conversion_new.m
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


  
% clear all, close all, clc;

% set the paths - all MM data sessions for Subject A
path_data = ["..\DatasetB\SubjectA_1129_S1\MM"; 
             "..\DatasetB\SubjectA_1206_S2\MM";
             "..\DatasetB\SubjectA_1213_S3\MM";
             "..\DatasetB\SubjectA_1220_S4\MM"];

for ci = 1:length(path_data)
    cd(path_data(ci))
    listing=dir("*.bin");
       
    n_files = size(listing);
    for file = 1:n_files
        ExG_data = convert_data_biowolf_GUI_2c(listing(file).name,'uV','s');
        T = table(ExG_data.Data(:,1),ExG_data.Data(:,2),ExG_data.Data(:,3),ExG_data.Data(:,4),...
        ExG_data.Data(:,5),ExG_data.Data(:,6),ExG_data.Data(:,7),ExG_data.Data(:,8),...
        ExG_data.Trigger(:),...
        'VariableNames', ["EEG_1","EEG_2","EEG_3","EEG_4","EEG_5","EEG_6","EEG_7","EEG_8","Trigger"]);
        if not(isfolder('csv'))
            mkdir('csv')
        end
        writetable(T,strcat('csv/',listing(file).name(1:end-3),'csv'),'Delimiter',',');
    end
    cd("../../..")
end
 




