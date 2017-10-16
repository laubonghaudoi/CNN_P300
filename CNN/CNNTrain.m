%% CNN训练函数
% 输入：
%   X 100*12*560
%   Y 2*560
%   opt numIteration lambda alpha
function net = CNNTrain(net, X, Y, opt)
%% 迭代训练
disp('Training network...');
sample = 1;
net.RMSe = 0;
figure; hold on;
for iter = 1:opt.numIteration
    tic;
    disp(['Iteration ' num2str(iter) '/' num2str(opt.numIteration)]);
    
    % 开始迭代训练
    randIndex = randperm(size(Y,2)); % 打乱训练样本的输入顺序
    numCorrect = 0;
    for sample_Iter = 1:size(X,3) % 每个训练样本
        batchX = X(:, :, randIndex(sample_Iter) );
        batchY = Y(:, randIndex(sample_Iter) );
        state = Y(1, sample_Iter); % 1代表P300，0代表非P300
        
        net = CNNFeedforward(net, batchX, batchY); % 前馈
        net = CNNBackPropagation(net, batchX, batchY, opt); % BP更新权值
        
        % 对BP后的新网络前馈输出检验是否符合标签值
        netCheck = CNNFeedforward(net, batchX, batchY);
        
        % 记录网络输出值及训练标签
        net.trainOutput(:, sample_Iter) = netCheck.layers{5}.a;
        net.trainY(:, sample_Iter) = batchY;
        
        % 确定输出是否P300
        if net.trainOutput(1, sample_Iter) > net.trainOutput(2, sample_Iter)
            output = 1;
        else
            output = 0;
        end
        
        % 测试在训练集上的准确率
        if state == output
            numCorrect = numCorrect + 1;
        end
        
        net.Loss(:, sample) = netCheck.loss;
        sample = sample + 1;
    end
    net.accuracyPerTrain(iter) = numCorrect/size(X, 3); % 记录此次迭代所得网络在训练集上的准确率
    fprintf('Accuracy on training set: %d%% \n', net.accuracyPerTrain(iter)*100 );
    toc;
end

end