%% QUICKSTART  免训练快速出图：用包内已训练好的原型直接生成（每数据集约 1 分钟）
%
%  用法：先 cd 到 matlab_reproduction 目录，然后
%        >> quickstart
%
%  依赖：Deep Learning Toolbox + Image Processing Toolbox，无需 GPU。
%  产物：fig_generation_mnist.png / fig_generation_fashion.png

if ~isfile('generate_mixture.m') || ~isfile('prototypes_mnist.mat')
    error('请在 matlab_reproduction 目录下运行 quickstart（需同时存在 generate_mixture.m 与 prototypes_*.mat）');
end

fprintf('\n===== [1/2] MNIST：场混合生成 10 类 =====\n');
setenv('OGEN_DATASET', 'mnist');
run('generate_mixture.m');

fprintf('\n===== [2/2] Fashion-MNIST：场混合生成 10 类 =====\n');
setenv('OGEN_DATASET', 'fashion');
run('generate_mixture.m');

fprintf('\n完成。结果图：fig_generation_mnist.png / fig_generation_fashion.png\n');
fprintf('如需从零重训，见 使用说明.md 第四节。\n');
