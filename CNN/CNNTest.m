%%  CNN测试函数
% 
% 
% 
% 
% 
% 
function net = CNNTest( net, X, Y )
numTP = 0;   numFN = 0;   numFP = 0;   numTN = 0;
net.TPindex = []; net.TNindex = []; net.FPindex = []; net.FNindex = [];

for sample_Iter = 1:size(X, 3)
    batchX = X(:, :, sample_Iter);
    batchY = Y(:, sample_Iter);
    state = Y(1, sample_Iter);
    
    % 前馈得到预测结果
    net = CNNFeedforward(net, batchX, batchY);
    net.testOutput(:, sample_Iter) = net.layers{5}.a;
    net.testLabel(:, sample_Iter) = Y(:, sample_Iter);
    
    if (net.testOutput(1, sample_Iter) > net.testOutput(2, sample_Iter))
        net.isP300(sample_Iter) = 1;
    else
        net.isP300(sample_Iter) = 0;
    end
    
    % 计算ROC曲线
    if (net.isP300(sample_Iter) == 1)&&(state == 1)
        numTP = numTP + 1;
        net.TPindex(numTP) = sample_Iter; 
    elseif (net.isP300(sample_Iter) == 0)&&(state == 0)
        numTN = numTN + 1;
        net.FNindex(numTN) = sample_Iter; 
    elseif (net.isP300(sample_Iter) == 1)&&(state == 0)
        numFP = numFP + 1;
        net.FPindex(numFP) = sample_Iter; 
    elseif (net.isP300(sample_Iter) == 0)&&(state == 1)
        numFN = numFN + 1;
        net.TNindex(numFN) = sample_Iter; 
    end
end

net.confusion(1,1) = numTP;
net.confusion(2,2) = numTN;
net.confusion(2,1) = numFP;
net.confusion(1,2) = numFN;

net.precision = numTP/(numTP+numFP);
net.recall = numTP/(numTP+numFN);

net.TPR = numTP/(numTP+numFN);
net.FPR = numFP/(numFP+numTN);

net.F1score = 2 * net.precision * net.recall / (net.precision + net.recall);
net.ROC = (1 - net.TPR)^2 + (1 - net.FPR)^2;
net.testAccuracy = (numTP+numTN)/(numTP+numTN+numFP+numFN);
end