function W = SparseEncoderLearn(trainFeatures,visibleSize,hiddenSize,opt)
fprintf('Learning features with sparse auto encoder, max iteration %d \n',opt.maxIteration);

%从样本中随机选取numPatches个1*nChannel的patch用作稀疏自编码器输入，以训练出卷积核
patches=SamplePatches(trainFeatures,opt.numPatches);%numPatches*visibleSize 10000*12

theta=SparseEncoderInitParam(visibleSize,hiddenSize);

%开始迭代学习
addpath minFunc/;
options.Method = 'lbfgs'; 
options.maxIter = opt.maxIteration;
options.display = 'on';

[opttheta, ~] = minFunc( @(p) SparseEncoderCost(p,visibleSize, hiddenSize,...
    opt.lambda, opt.sparsityParam, opt.beta, patches'),theta,options);

W = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
end