function [trainSamples,trainLabels]=ExtractData(rawSignal,event)
%% 特征提取参数设定
trialStartPoints = find(event.type>=41 & event.type<=80); % 30*1 30次trial开始的时间点（位置值），401*n+1~11630
numTrials = length(trialStartPoints)-1; % 最后一个字符因为采样不完整舍弃，只用29个
numEpochs = 40; % 屏幕上总共40个字符
numRounds = round( (trialStartPoints(2)-trialStartPoints(1))/numEpochs ); % 每个字符重复10次
epochLength = 150; % 时间窗为150个采样点

filterCoefB = [0.0083   -0.0260    0.0464   -0.0551    0.0580   -0.0551    0.0464   -0.0260    0.0083];
filterCoefA = [1.0000   -5.0983   11.7635  -15.9105   13.7377   -7.7295    2.7616   -0.5718    0.0525];

channelSelected = [9 14 19 24 28:32 34:36]; % NuAmps的通道选择
numChannels = length(channelSelected); % 共12个通道

timeWindowLeft = 25;
timeWindowRight = 125;
timeWindow = timeWindowRight - timeWindowLeft;

%% 读取数据并滤波
% trainFeatureMaps,trainLabels为提取出的训练样本及标签值
trainSamples = zeros(numTrials*numRounds*numEpochs, timeWindow, numChannels); % (29*10*40)*20*12=11600*100*12 11600个卷积训练样本
trainLabels = zeros(numTrials*numRounds*numEpochs, 1); % (29*10*40)*1=11600*1

targetChars = zeros(numTrials, 1);

disp('Extracting features...');
% 三层for循环分别为 1-29 trials * 1-10 rounds * 1-40 epochs
for trial_Iter = 1:numTrials
    targetChars(trial_Iter) = event.type( trialStartPoints(trial_Iter) )-40;
    for round_Iter = 1:numRounds
        for epoch_Iter = trialStartPoints(trial_Iter) + (round_Iter - 1)*numEpochs + 1 : trialStartPoints(trial_Iter) + round_Iter*numEpochs
            % 把这一时刻屏幕所闪字符及36个频道的信号值提取
            flashingCode = event.type(epoch_Iter);
            signalEpoch = rawSignal(event.pos(epoch_Iter):event.pos(epoch_Iter)+epochLength-1, :)';%36*150
            
            if (flashingCode > 0 && flashingCode <= numEpochs)
                signalEpoch = signalEpoch(:,timeWindowLeft+1:timeWindowRight);%36*100 取中间时间窗
                %signalDebased = signalEpoch - kron(mean(signalEpoch(:,2:24),2), ones(1, size(signalEpoch,2)) );
                signalFiltered = filter( filterCoefB, filterCoefA, signalEpoch(channelSelected, :)',[],1);%100*12
                trainSamples( (trial_Iter-1)*numRounds*numEpochs + (round_Iter-1)*numEpochs + flashingCode, :, :) = signalFiltered; %11600*100*12
            end
                
        end
        trainLabels( (trial_Iter-1)*numRounds*numEpochs + (round_Iter-1)*numEpochs + targetChars(trial_Iter) ) = 1;%11600*1
    end
end

%% 去除漏标情况并标准化提取出的数据
featureDim1 = size(trainSamples, 2);
featureDim2 = size(trainSamples, 3);
trainSamples = reshape(trainSamples, size(trainSamples,1), []); % 11600*1200

trainLabels( all(trainSamples==0, 2) ) = []; % 如果有某一特征全为0，则去除
trainSamples( all(trainSamples==0, 2), :) = []; % 11600*1200

% 归一化
trainSamples = zscore(trainSamples')';

% 恢复原结构
trainSamples = reshape(trainSamples, size(trainSamples, 1), featureDim1, featureDim2); % 11600*100*12

% 调整数据结构以输入CNN
trainSamples = permute(trainSamples, [2 3 1]); % 100*12*11600
trainLabels = [trainLabels'; ~trainLabels']; % 2*11600
end