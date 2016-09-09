%% =======================训练模型主程序==============================
% 本程序流程：
% 1、读取原始EEG信号，预处理得100（时域）*12（频道）的特征片段（ReadData文件夹）
% (已废弃)用稀疏自编码器学习得到卷积核W 100*240（AutoEncoder文件夹）
% 2、将每一个特征片段输入CNN中训练
% 3、训练完成，先用训练集测试，再用测试集测试
%
% 注：1、预处理中去除克罗内克积步骤
%     2、正负样本比例
%
%
%
% =========================================================================
%% 设置路径及要训练的数据文件（可调）
clear;clc;
addpath data/
% cntTrainFileName = 'kangyarui_20150709_train_1.cnt';
% cntTestFileName = 'kangyarui_20150709_test_1.cnt';

cntTrainFileName = 'chenzhubing_20150627_train_1.cnt';
cntTestFileName = 'chenzhubing_20150627_test_1.cnt';

%% 提取数据
addpath ReadData/

% 读取训练数据
disp('Reading train data...');
[rawSignalTrain, eventTrain] = readcnt(cntTrainFileName); % rawSignal 136410*36 36个频道的采样值
[trainData, trainLabel] = ExtractData(rawSignalTrain, eventTrain);
% 读取测试数据
disp('Reading test data...');
[rawSignalTest, eventTest] = readcnt(cntTestFileName);
[testData, testLabel] = ExtractData(rawSignalTest, eventTest);

%% 稀疏自编码器提取特征用作CNN卷积核
% addpath AutoEncoder/
% 
% % 可调参数，稀疏自编码器训练卷积核用
% visibleSize=1*12; % 滑动卷积只沿时域轴进行，故卷积核覆盖全部频道
% hiddenSize = 6; % 自编码器隐藏层神经元数，亦为feature map数
% opt.numPatches=100000; % 可调参数，选取的学习样本数,须小于11600*20
% opt.sparsityParam = 0.01; % 稀疏性参数
% opt.lambda = 0.01; % regularization
% opt.beta = 1; % 稀疏性惩罚因子
% opt.maxIteration = 1000; % 最大迭代次数
% 
% % 稀疏自编码器学学习得到卷积核
% W = SparseEncoderLearn(trainFeatures, visibleSize, hiddenSize, opt);
% W = reshape(W,size(W,1),1,visibleSize);%hiddenSize*1*12
% clearvars -except W trainFeatures trainLabels 
% % 至此，已训练得到hiddenSize个1*visibleSize的卷积核，用于CNN卷积输入

%% 取正负样本
% 经过数据预处理得到 trainData 100*20*numData 和 trainLabel 2*numData
% 现需分别提取出等量的正负样本用作CNN训练集
posIndex = find(trainLabel(1,:)); % 1*290
negIndex = find(trainLabel(2,:)); % 1*11600
numPos = size(posIndex,2); % 290个正样本
numNeg = size(negIndex,2); % 11310个负样本
randNegIndex = randperm(numNeg, numPos); % 从负样本中随机抽取与正样本等量个训练

% preallocate for speed
inputX = zeros( size(trainData,1), size(trainData,2), 2*numPos); % 100*12*580
inputY = zeros( size(trainLabel,1), 2*numPos); % 2*580
for pos_Iter = 1:numPos % 290个正样本
    inputX(:, :, pos_Iter) =  trainData(:, :, posIndex(pos_Iter) );
    inputY(:, pos_Iter) = trainLabel(:, posIndex(pos_Iter) );
end
for neg_Iter = 1:numPos % 290个负样本
    inputX(:, :, neg_Iter + numPos) = trainData(:, :, randNegIndex(neg_Iter) );
    inputY(:, neg_Iter + numPos) = trainLabel(:, randNegIndex(neg_Iter) );
end

inputX = permute(inputX,[3,1,2]);
inputY = inputY';
clearvars -except inputX inputY trainData trainLabel testData testLabel cntTrainFileName cntTestFileName

% %% CNN训练
% % 网络结构见CNNInitParam.m文件
% 
% addpath CNN/
% % 设定训练参数opt（可调）
% CNNOpt.numIteration = 3000;
% CNNOpt.lambda = 0;
% CNNOpt.alpha = 5e-4; % 学习速率
% % 初始化网络对象
% CNN.layers = {
%     struct('type','L0', 'dimension','100x12')
%     struct('type','L1', 'numMaps',10, 'mapSize',100, 'kernelSize',12)
%     struct('type','L2', 'numMaps',5, 'mapSize',5, 'kernelSize',20)
%     struct('type','L3', 'hiddenSize',100)
%     struct('type','L4', 'dimension',2)
% };
% %rng(0);
% % 初始化网络参数
% CNN = CNNInitParam(CNN);
% % 训练网络
% CNN = CNNTrain(CNN, inputX, inputY, CNNOpt);
% 
% %% 测试网络
% fprintf('Train completed.\nTesting training set...\n');
% 
% testTrainingSet = CNNTest(CNN, trainData, trainLabel); % 测试训练集
% testTestSet = CNNTest(CNN, testData, testLabel); % 测试测试集
% 
% % 每次迭代后网络在训练集上的准确率g
% figure; plot(testTrainingSet.accuracyPerTrain);
% xlabel('Iteration'); ylabel('Output accuracy on training set');
% 
% % 每个batch后的损失函数Loss
% figure; plot(testTrainingSet.Loss);
% xlabel('batch'); ylabel('Loss');
% 
% % 记录实验参数
% NOTE = {'normalization', true;...
%         'kronecker', false;...
%         'lambda', CNNOpt.lambda;...
%         'alpha', CNNOpt.alpha;...
%         'iter',CNNOpt.numIteration;...
%         'trainFile',cntTrainFileName;...
%         'testFile', cntTestFileName};
