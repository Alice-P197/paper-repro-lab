function download_data(ds)
%DOWNLOAD_DATA  下载 MNIST / Fashion-MNIST 原始 IDX 文件到 ../data/<ds>
%
%  用法：
%    download_data('mnist')      下载到 ../data/mnist     （约 12 MB）
%    download_data('fashion')    下载到 ../data/fashion   （约 26 MB）
%
%  下载后 train_prototypes / train_encoder_distill 会自动读取（load_idx.m
%  内部解压 .gz，无需手动处理）。两个数据集都提供备用镜像，主镜像失败会自动
%  切换到 GitHub 镜像。

if nargin < 1, ds = 'mnist'; end
ds = lower(char(ds));

switch ds
    case 'mnist'
        mirrors = {'https://ossci-datasets.s3.amazonaws.com/mnist/', ...
                   'https://raw.githubusercontent.com/fgnt/mnist/master/'};
    case {'fashion', 'fashion-mnist', 'fashionmnist'}
        ds = 'fashion';
        mirrors = {'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', ...
                   'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'};
    otherwise
        error('download_data: 只支持 ''mnist'' 或 ''fashion''');
end

files = {'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', ...
         't10k-images-idx3-ubyte.gz',  't10k-labels-idx1-ubyte.gz'};

outDir = fullfile('..', 'data', ds);
if ~exist(outDir, 'dir'), mkdir(outDir); end
fprintf('目标目录：%s\n', outDir);

for i = 1:numel(files)
    f   = files{i};
    out = fullfile(outDir, f);
    if isfile(out) && dir(out).bytes > 1000
        fprintf('[skip] %-30s 已存在\n', f); continue;
    end
    ok = false;
    for m = 1:numel(mirrors)
        url = [mirrors{m} f]; %#ok<AGROW>
        fprintf('[get ] %s\n', url);
        try
            websave(out, url);
            if isfile(out) && dir(out).bytes > 1000, ok = true; break; end
        catch ME
            fprintf('[fail] %s\n', ME.message);
            if isfile(out), delete(out); end
        end
    end
    if ~ok
        error('下载失败：%s —— 请手动下载后放入 %s', f, outDir);
    end
end

fprintf('\n完成：%s（%d 个文件）\n', outDir, numel(files));
if strcmp(ds, 'mnist')
    fprintf('下一步：setenv(''OGEN_DATASET'',''mnist''); train_prototypes\n');
else
    fprintf('下一步：setenv(''OGEN_DATASET'',''fashion''); train_prototypes\n');
end
end
