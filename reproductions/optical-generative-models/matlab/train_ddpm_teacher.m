%% TRAIN_DDPM_TEACHER  (Optional) Teacher DDPM for the optical generative model
%  Paper Methods: "we first train a teacher digital generative model based on
%  the denoising diffusion probabilistic model (DDPM)... Once trained, the
%  learned DDPM is frozen, and it continuously generates the noise-image data
%  pairs used to train the snapshot optical generative model."
%
%  Settings follow the paper: T = 1000, linear beta from 1e-4 to 0.02,
%  AdamW lr = 1e-4. NOTE: a paper-quality teacher needs a GPU and far more
%  iterations; the defaults give a quick, coarse teacher for the pipeline.
%
%  Outputs:
%    ddpm_teacher.mat     - network weights
%    teacher_samples.mat  - 32x32x1xN samples in [-1,1] + labels,
%                           directly usable by train_optical_gen.m

clear; clc; close all; rng(1);

% ---------------- config ----------------
T       = 1000;
beta    = linspace(1e-4, 0.02, T);        % linear schedule (paper)
abar    = cumprod(1 - beta);
nCh     = [64 128 256];                   % small U-Net widths
batch   = 64;
iters   = 2000;
lr      = 1e-4;
nSample = 100;                            % CPU: sampling is 1000 U-Net evals/image
if ~isempty(getenv('DDPM_ITERS')),   iters   = str2double(getenv('DDPM_ITERS'));   end
if ~isempty(getenv('DDPM_NSAMPLE')), nSample = str2double(getenv('DDPM_NSAMPLE')); end
useGPU  = gpuDeviceCount > 0;
fprintf('DDPM teacher: %d iters, GPU=%d\n', iters, useGPU);

% ---------------- data ----------------
[X, Y] = digitTrain4DArrayData;
X = imresize(single(X), [32 32]);
X = 2*X - 1;
if iscategorical(Y) || isstring(Y), Y = double(string(Y)); else, Y = double(Y); end
Y = Y(:);
N = size(X, 4);

% ---------------- init ----------------
params = unet_init(nCh);
if useGPU
    params = to_gpu_struct(params);
end
avgG = []; avgS = [];

% ---------------- training loop ----------------
lossHist = zeros(iters, 1);
tic;
for it = 1:iters
    idx = randi(N, batch, 1);
    x0  = X(:,:,:,idx);
    x0  = dlarray(reshape(x0, 32, 32, 1, batch), 'SSCB');
    if useGPU, x0 = gpuArray(x0); end

    t    = randi(T, batch, 1);
    eps0 = randn(32, 32, 1, batch, 'single');
    at   = reshape(single(abar(t)), 1, 1, 1, batch);
    if useGPU, eps0 = gpuArray(eps0); at = gpuArray(at); end
    xt = sqrt(at).*x0 + sqrt(1-at).*eps0;             % forward diffusion

    [loss, grads] = dlfeval(@ddpm_loss, params, xt, eps0, t, T);
    lossHist(it) = double(gather(extractdata(loss)));

    [params, avgG, avgS] = adamupdate(params, grads, avgG, avgS, it, lr);

    if mod(it, 25) == 0 || it == 1
        fprintf('ddpm iter %5d | eps-MSE %.4f | %.1f s\n', it, lossHist(it), toc);
    end
    if mod(it, 500) == 0 || it == iters
        save('ddpm_teacher.mat', 'params', 'nCh', 'T', 'beta', 'abar');
    end
end
save('ddpm_teacher.mat', 'params', 'nCh', 'T', 'beta', 'abar');

figure('Color','w'); plot(movmean(lossHist,25)); grid on;
xlabel('Iteration'); ylabel('\epsilon MSE'); title('DDPM teacher training');
print('-dpng','-r200','fig_ddpm_loss.png');

% ---------------- ancestral sampling ----------------
fprintf('Sampling %d teacher images...\n', nSample);
Xs = zeros(32,32,1,nSample,'single'); Ys = zeros(nSample,1);
for k = 1:nSample
    xt = randn(32,32,1,1,'single');
    for t = T:-1:1
        xt = ddpm_step(params, xt, t, beta, abar, T, useGPU);
    end
    Xs(:,:,:,k) = single(gather(xt));
    Ys(k) = mod(k-1, 10);
    if mod(k,50) == 0, fprintf('  sampled %d/%d\n', k, nSample); end
end
save('teacher_samples.mat', 'Xs', 'Ys');
fprintf('Saved teacher_samples.mat\n');

figure('Color','w');
nShow = min(25, nSample);
for j = 1:nShow
    subplot(5,5,j); imshow((Xs(:,:,:,j)+1)/2, []); axis off;
end
sgtitle('Teacher DDPM samples');
print('-dpng','-r200','fig_ddpm_samples.png');

%% ---------------- local functions ----------------
function [loss, grads] = ddpm_loss(params, xt, eps0, t, T)
    epred = unet_forward(params, xt, t, T);
    loss  = mean((epred - eps0).^2, 'all');
    grads = dlgradient(loss, params);
end

function xt = ddpm_step(params, xt, t, beta, abar, T, useGPU) %#ok<INUSD>
    e = dlfeval(@unet_forward, params, dlarray(single(xt),'SSCB'), t, T);
    e = reshape(gather(extractdata(e)), 32,32,1,1);
    z = 0;
    if t > 1, z = randn(32,32,1,1,'single'); end
    xt = (1/sqrt(1-beta(t))) .* (xt - (beta(t)/sqrt(1-abar(t))).*e) ...
         + sqrt(beta(t)).*z;
    if useGPU, xt = gpuArray(xt); end
end

function params = unet_init(nCh)
    c1 = nCh(1); c2 = nCh(2); c3 = nCh(3); teD = 64;
    p = struct();
    % time embedding MLP: 32 sinusoid dims -> teD -> teD
    p.teW1 = dlarray(single(randn(teD, 32)*sqrt(2/32)));
    p.teB1 = dlarray(single(zeros(teD,1)));
    p.teW2 = dlarray(single(randn(teD, teD)*sqrt(2/teD)));
    p.teB2 = dlarray(single(zeros(teD,1)));
    % per-block learnable time projections: C x teD  (FiLM-free additive bias)
    names = {'tb0','tb1','tb2','tb3','tb4','tb5'};
    Cs    = {c1, c1, c2, c3, c2, c1};
    for q = 1:6
        p.(names{q}) = dlarray(single(0.02*randn(Cs{q}, teD)));
    end
    % stem
    p.c0 = conv_init(3, 1, c1);
    % encoder (downsampling convs also change channel count so that each
    % residual block has matching input/output channels for the skip add)
    p.r1a = conv_init(3, c1, c1);  p.r1b = conv_init(3, c1, c1);
    p.d1  = conv_init(3, c1, c2);              % stride-2: 32x32xc1 -> 16x16xc2
    p.r2a = conv_init(3, c2, c2);  p.r2b = conv_init(3, c2, c2);
    p.d2  = conv_init(3, c2, c3);              % stride-2: 16x16xc2 -> 8x8xc3
    p.r3a = conv_init(3, c3, c3);  p.r3b = conv_init(3, c3, c3);
    % decoder
    p.u3  = conv_init(3, c3, c2);              % up: 8 -> 16
    p.r4a = conv_init(3, c2+c2, c2); p.r4b = conv_init(3, c2, c2);
    p.u4  = conv_init(3, c2, c1);              % up: 16 -> 32
    p.r5a = conv_init(3, c1+c1, c1); p.r5b = conv_init(3, c1, c1);
    p.cOut = conv_init(3, c1, 1);
    params = p;
end

function W = conv_init(f, cin, cout)
    w = single(randn(f, f, cin, cout) * sqrt(2/(f*f*cin)));
    W = dlarray(w, 'SSCU');     % U labels the output-channel dimension
end

function y = convf(x, W)
    % dlconv with zero bias (bias format 'C', device-matched)
    cout = size(W, 4);
    wd = extractdata(W);
    b = dlarray(zeros(cout, 1, 'like', wd), 'C');
    y = dlconv(x, W, b, 'Padding', 'same');
end

function s = to_gpu_struct(s)
    f = fieldnames(s);
    for q = 1:numel(f)
        s.(f{q}) = gpuArray(s.(f{q}));
    end
end

function epred = unet_forward(params, x, t, T)
    B = size(x, 4);
    % sinusoidal time embedding (32 dims) -> MLP
    half = 16;
    freqs = exp(-log(T) * (1:half)/half);
    ang = single(t(:)) * (2*pi*freqs);              % B x half
    te = dlarray([sin(ang), cos(ang)]');            % 32 x B
    te = relu(params.teW1*te + params.teB1);
    te = params.teW2*te + params.teB2;              % teD x B
    if isgpuarray(extractdata(x)) && ~isgpuarray(extractdata(te))
        te = gpuArray(te);
    end

    tb = @(nm, C) time_bias(params.(nm), te, C, B); % 1 x 1 x C x B

    h  = lrelu(gn(convf(x, params.c0)) + tb('tb0', 64));
    r1 = resblock(params, 'r1a', 'r1b', 'tb1', h, te, 64, B);   % 32x32x64
    h2 = stride2(r1, params.d1);                                 % 16x16x64
    r2 = resblock(params, 'r2a', 'r2b', 'tb2', h2, te, 128, B);  % 16x16x128
    h3 = stride2(r2, params.d2);                                 % 8x8x128
    r3 = resblock(params, 'r3a', 'r3b', 'tb3', h3, te, 256, B);  % 8x8x256
    u3 = conv_up(r3, params.u3);                                 % 16x16x128
    r4 = resblock2(params, 'r4a', 'r4b', 'tb4', cat(3, u3, r2), te, 128, B);
    u4 = conv_up(r4, params.u4);                                 % 32x32x64
    r5 = resblock2(params, 'r5a', 'r5b', 'tb5', cat(3, u4, r1), te, 64, B);
    epred = convf(r5, params.cOut);
end

function y = time_bias(Wtb, te, C, B)
    modv = Wtb * te;                                 % C x B
    y = dlarray(reshape(modv, 1, 1, C, B), 'SSCB');   % broadcasts over S,S2
end

function y = resblock(params, na, nb, tn, x, te, C, B)
    h = lrelu(gn(convf(x, params.(na))) + time_bias(params.(tn), te, C, B));
    h = gn(convf(h, params.(nb)));
    y = x + h;
end

function y = resblock2(params, na, nb, tn, x, te, C, B)
    h = lrelu(gn(convf(x, params.(na))) + time_bias(params.(tn), te, C, B));
    y = gn(convf(h, params.(nb)));
end

function y = stride2(x, W)
    y = convf(x, W);                  % keep resolution, then subsample
    y = y(1:2:end, 1:2:end, :, :);
end

function y = conv_up(x, W)
    y = repelem(x, 2, 1, 1, 1);       % nearest-neighbour upsampling
    y = repelem(y, 1, 2, 1, 1);
    y = gn(convf(y, W));
end

function y = lrelu(x), y = max(x,0) + 0.2*min(x,0); end

function y = gn(x)
    % per-sample, per-position channel normalization (GroupNorm with 1 group)
    m = mean(x, 3);
    v = mean((x - m).^2, 3);
    y = (x - m) ./ sqrt(v + 1e-5);
end
