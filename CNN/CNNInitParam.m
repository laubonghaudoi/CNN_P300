%% 初始化CNN每层网络参数
% 网络结构：
% L0输入层 每个输入为100*12的特征图 
% L1卷积层 卷积核为1*12，得到10个100*1的时域特征
% L2卷积池化层 卷积核为20*1，由每个L1所得特征卷积得到5个5*1的特征，即共50个5*1特征
% 本层亦为BP网络输入层，共250个神经元
% L3为BP网络隐藏层，与L2全连通
% L4输出层，二分类
function net = CNNInitParam(net)
%% 设定规范化参数
r1 = 2 * sqrt( 6 / ( 10*100 + 12 + 1));     % L1卷积层
r2 = 2 * sqrt( 6 / ( 10*100 + 50*20 + 1));  % L2卷积池化层
r3 = 2 * sqrt( 6 / ( 100 + 50*20 +1));      % L3全连通BP网络隐藏层
r4 = 2 * sqrt( 6 / ( 100 + 2 + 1));         % L4输出层
%% L1 卷积层 10个特征map
for map_Iter = 1:net.layers{2}.numMaps % L1特征数
    %每个卷积核为1*12大小，将频域卷积
    net.layers{2}.k{map_Iter} = (rand(1,12)-0.5) * r1;
    net.layers{2}.b(map_Iter) = 0;
end
%% L2 卷积池化层，共10*5个特征map
for map_Iter1 = 1:net.layers{2}.numMaps
    for map_Iter2 = 1:net.layers{3}.numMaps
        %每个卷积核大小为20*1，将时域卷积并池化
        net.layers{3}.k{map_Iter1}{map_Iter2} = (rand(1,20)-0.5) * r2;
        net.layers{3}.b(map_Iter1, map_Iter2) = 0;
    end
end
%% L3 全连通层，100个神经元
for map_Iter = 1:net.layers{4}.hiddenSize
    net.layers{4}.k{map_Iter} = (rand(1, net.layers{2}.numMaps * net.layers{3}.numMaps * net.layers{3}.mapSize) - 0.5) * r3;
    net.layers{4}.b(map_Iter) = 0;
end
%% L4 输出层 二分类
net.layers{5}.k{1} = (rand(1,100)-0.5) * r4;
net.layers{5}.b(1) = 0;
net.layers{5}.k{2} = (rand(1,100)-0.5) * r4;
net.layers{5}.b(2) = 0;
end