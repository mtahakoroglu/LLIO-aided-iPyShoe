clear all; clc;
dinfo = dir('*.mat');
filename = {dinfo.name};
detector = {}; threshold = zeros(1,length(filename));
for i=1:length(filename)
    data = load(filename{i});
    detector{i} = data.best_detector;
    thresholdString = sprintf('G_%s_opt', detector{i});
    threshold(1,i) = data.(thresholdString);
end

