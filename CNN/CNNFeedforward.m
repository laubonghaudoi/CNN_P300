%% CNN前馈函数
% 输入：
%   X 100*12
%   Y 2*1
%
% 网络参数：
%   L0  dimension   '100x12'
%   net.layers{1}
%
%	L1  numMaps 10  mapSize 100 kernelSize  12
%   net.layers{2}    k{10} 1*12  b 1*10
%
%	L2  numMaps 5   mapSize 5   kernelSize  20
%   net.layers{3}    k{10}{5} 1*20   b 10*5
%
%	L3  dimension   100
%   net.layers{4}    k{100} 1*250    b 1*100
%
%	L4  dimension   2
%   net.layers{5}    k{2} 1*100  b 1*2
function net = CNNFeedforward(net, batchX, batchY)
%% L0到L1前馈
w = reshape( cell2mat(net.layers{2}.k), net.layers{2}.kernelSize, net.layers{2}.numMaps)'; % L1权重 10 numMaps * 12 kernels
net.layers{2}.a = tanh_opt( bsxfun(@plus,w * batchX', net.layers{2}.b') ); % 10*100 L1激活值，10个100*1卷积后时域特征
%% L1到L2前馈
% L2输出a为50个5*1的特征图
map_Iter = 1;
% L1传来的10个100*1特征图，需要将其卷积池化为5*1大小，卷积核为1*20
for mapL1_Iter = 1:net.layers{2}.numMaps % 10个特征图
    z1 = reshape( net.layers{2}.a(mapL1_Iter, :), net.layers{3}.kernelSize, net.layers{3}.numMaps); % 将1*100重塑为20*5
    bias = net.layers{3}.b(mapL1_Iter,:); % 1*5
    for mapL2_Iter = 1:net.layers{3}.numMaps % 每个特征图又卷积池化出5个特征图，故共50个
        net.layers{3}.a{map_Iter} = tanh_opt( bsxfun(@plus, net.layers{3}.k{mapL1_Iter}{mapL2_Iter} * z1, bias(mapL2_Iter)) );
        map_Iter = map_Iter+1;
    end
end
%% L2到L3前馈
z2 = cell2mat( net.layers{3}.a ); % 1*250 50个1*5片段
w = reshape( cell2mat(net.layers{4}.k), size(z2,2), net.layers{4}.hiddenSize )'; % 100*250
net.layers{4}.a = sigmoid( w*z2' + net.layers{4}.b' ); % 100*1
%% L3到L4前馈
w = [ net.layers{5}.k{1} ; net.layers{5}.k{2} ]; % 2*100
net.layers{5}.a = sigmoid( w*net.layers{4}.a + net.layers{5}.b' ); % 2*1
%% 计算误差
net.e = batchY - net.layers{5}.a; % 2*1
net.loss = 1/2 * (sumsqr(net.e))^2;
end

%% sigmoid函数
function sig = sigmoid(x)
sig = 1 ./ ( 1 + exp(-x) );
end
%% tanh_opt函数
function  f = tanh_opt(x)
    f = 1.7159 * tanh( 2/3 .*x);
end