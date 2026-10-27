%% TRAIN_OPTICAL_GEN  Snapshot optical generative model (MNIST digits)
%  Reproduction of: S. Chen et al., "Optical generative models", Nature 644,
%  903-911 (2025) - Fig. 1 / Extended Data Fig. 2 (snapshot, monochrome).
%
%  Pipeline (Methods):
%    - digital encoder : 3 FC layers, LeakyReLU(0.2), input = 32x32 Gaussian
%                        noise + 32-dim class embedding (Eq. 8)
%    - phase encoding  : phi = alpha*pi*(H+1)/2, alpha = 2, upsampled 10x (Eq. 1)
%    - optical decoder : 1 phase-only layer (400x400), angular spectrum
%                        propagation, d01 = 120.1 mm, d12 = 96.4 mm (Eq. 2-5)
%    - loss            : MSE + gamma * KL of intensity histograms (Eq. 7)
%
%  Training pairs: since the paper's teacher DDPM needs a separate (heavy)
%  training run, by default the model is trained directly on real MNIST
%  samples ("teacher = dataset" mode, TRAINMODE='direct'). Set TRAINMODE
%  to 'ddpm' and first run train_ddpm_teacher.m to distil from a learned
%  DDPM exactly as in the paper.

clear; clc; close all;
rng(0);

%% --------------------------- configuration ---------------------------
P.n        = 32;              % input noise resolution (h x w)
P.up       = 10;              % seed upsampling factor (Methods)
P.N        = P.n * P.up;      % object-plane / decoder grid: 320 -> decoder 400? see pad
P.dx       = 8e-6;            % SLM pixel pitch [m] (Meadowlark XY, 8 um)
P.lambda   = 520e-9;          % illumination wavelength [m]
P.d01      = 120.1e-3;        % SLM -> decoder distance [m]
P.d12      = 96.4e-3;         % decoder -> sensor distance [m]
P.alpha    = 2.0;             % phase range coefficient (0 - 2*pi)
P.gamma    = 1e-4;            % KL regularization coefficient
P.klBins   = 32;              % histogram bins for KL loss

P.Ndec     = 400;             % decoder layer size (400 x 400 learnable features)
P.padOff   = floor((P.Ndec - P.n*P.up)/2);   % center the 320 field in the 400 grid
assert(P.padOff >= 0)

P.encSize  = P.n*P.n + 32;    % FC layer width = input feature size (1056)
P.nClass   = 10;
% Batch size / iterations: paper uses 100 / ~20k on GPU. On CPU, batch 32
% with a few thousand iterations already yields recognizable digits.
P.batch    = 32;
P.iters    = 4000;
P.lrEnc    = 1e-4;            % digital parameters (paper)
P.lrDec    = 2e-3;            % decoding layer (paper)
P.lrEmb    = 1e-3;            % class conditioning (emb + M3): separate lr group
P.decayIters = P.iters;       % cosine annealing horizon

% Overrides via environment variables
if ~isempty(getenv('OGEN_ITERS')), P.iters = str2double(getenv('OGEN_ITERS')); end
if ~isempty(getenv('OGEN_BATCH')), P.batch = str2double(getenv('OGEN_BATCH')); end

% Dataset: 'mnist' (full 60k, local data/mnist) or 'fashion' (Fashion-MNIST)
ds = 'mnist';
if ~isempty(getenv('OGEN_DATASET')), ds = getenv('OGEN_DATASET'); end
ckptBest = ['checkpoint_' ds '.mat'];
ckptLast = ['checkpoint_last_' ds '.mat'];
figLoss  = ['fig_training_loss_' ds '.png'];
figGen   = ['fig_generated_digits_' ds '.png'];
figPrev  = ['fig_preview_' ds '.png'];

% Quick smoke-test mode (a few minutes): set environment flag SMOKE=1
if getenv('SMOKE') == '1'
    P.batch = 32; P.iters = 300;
    fprintf('*** SMOKE MODE: batch=%d iters=%d ***\n', P.batch, P.iters);
end

useGPU = (gpuDeviceCount > 0);
if useGPU, gpuDevice(1); end
fprintf('GPU: %d\n', useGPU);

%% --------------------------- data ------------------------------------
% Full MNIST / Fashion-MNIST (60,000 training images) from local IDX files;
% falls back to the 5,000-image Toolbox builtin when the files are absent.
dataDir = fullfile('..', 'data', ds);
imgGz   = fullfile(dataDir, 'train-images-idx3-ubyte.gz');
if exist(imgGz, 'file')
    [X, Y] = load_idx(dataDir, 'train');
    fprintf('Dataset: %s (full IDX, %d images)\n', ds, size(X, 4));
else
    [X, Y] = digitTrain4DArrayData;      % 28x28x1xN fallback, labels 0..9
    fprintf('Dataset: %s (Toolbox builtin, %d images)\n', ds, size(X, 4));
end
X = imresize(X, [P.n P.n]);              % resize to 32x32
X = 2*X - 1;                             % map to [-1, 1]
Y = double(Y(:)) + 1;                    % 1..10
N = size(X, 4);
inkFracAll = mean((X + 1) / 2, 'all');   % dataset ink level (display/normalization)
fprintf('Data: %d images of %dx%d | ink frac %.4f\n', N, P.n, P.n, inkFracAll);

% Per-class mean images (conditional prototypes). CPU-scale strategy: training
% against the class mean gives the class conditioning a strong, unambiguous
% gradient (random-sample targets make the class-agnostic "global mean" a
% strong attractor). Fine-tune with OGEN_TARGET=sample afterwards to restore
% sample diversity, mirroring the paper's DDPM-distillation stage.
muClass = zeros(P.n, P.n, 1, P.nClass, 'single');
for c = 1:P.nClass
    muClass(:,:,:,c) = mean(X(:,:,:,Y == c), 4);
end
targetMode = getenv('OGEN_TARGET');
if isempty(targetMode), targetMode = 'classmean'; end
P.targetMode = targetMode;
P.inkFracAll = inkFracAll;
fprintf('Target mode: %s\n', targetMode);

% Optional warm start from a previous checkpoint (phase B fine-tuning)
initFile = getenv('OGEN_INIT');

%% --------------------------- parameters ------------------------------
if ~isempty(initFile) && exist(initFile, 'file')
    S0 = load(initFile);
    params = S0.params;
    fprintf('Warm start from %s\n', initFile);
else
    params = init_params(P);
end
if useGPU
    params = structfun(@(w) gpuArray(w), params, 'UniformOutput', false);
end
avgGradEnc = []; avgGradDec = []; avgSqEnc = []; avgSqDec = [];
avgGradEmb = []; avgSqEmb = [];
avgGradM3  = []; avgSqM3  = [];
bestMA = inf; best = [];

%% --------------------------- training loop ---------------------------
lossHist = zeros(P.iters, 1);
tic;
for it = 1:P.iters
    % cosine-annealed learning rates
    t = it / P.decayIters;
    lrE   = P.lrEnc * 0.5 * (1 + cos(pi * t));
    lrD   = P.lrDec * 0.5 * (1 + cos(pi * t));
    lrEmb = P.lrEmb * 0.5 * (1 + cos(pi * t));

    % minibatch
    idx = randi(N, P.batch, 1);
    cls = Y(idx);
    if strcmp(targetMode, 'classmean')
        tgt = muClass(:,:,:,cls);                    % conditional prototypes
    else
        tgt = X(:,:,:,idx);                          % teacher samples in [-1,1]
    end
    tgt = dlarray(single(tgt));
    if useGPU, tgt = gpuArray(tgt); end
    cls = Y(idx);

    % forward + backward
    [loss, grads] = dlfeval(@modelLoss, params, tgt, cls, P);
    lossHist(it) = double(gather(extractdata(loss)));

    if isfinite(lossHist(it))
        % global gradient clipping (norm 1.0)
        g = clip_grads(grads, 1.0);

        % AdamW updates: encoder body / conditioning (emb+M3) / decoder
        encBody = rmfield(params.Enc, {'emb','M3'});
        gBody   = rmfield(g.Enc, {'emb','M3'});
        [encBody, avgGradEnc, avgSqEnc] = adamupdate(encBody, gBody, ...
            avgGradEnc, avgSqEnc, it, lrE);
        [embNew, avgGradEmb, avgSqEmb] = adamupdate(params.Enc.emb, g.Enc.emb, ...
            avgGradEmb, avgSqEmb, it, lrEmb);
        [m3New, avgGradM3, avgSqM3] = adamupdate(params.Enc.M3, g.Enc.M3, ...
            avgGradM3, avgSqM3, it, lrEmb);
        encBody.emb = embNew;
        encBody.M3  = m3New;
        params.Enc  = encBody;
        [params.Dec, avgGradDec, avgSqDec] = adamupdate(params.Dec, g.Dec, ...
            avgGradDec, avgSqDec, it, lrD);
        % decoupled weight decay (AdamW)
        params.Enc = structfun(@(w) w .* (1 - lrE*1e-2), params.Enc, 'UniformOutput', false);
        params.Dec = params.Dec .* (1 - lrD*1e-2);
    else
        fprintf('iter %d: non-finite loss, skipping update\n', it);
    end

    if mod(it, 25) == 0 || it == 1
        fprintf('iter %5d | loss %.4f | lrE %.2e lrD %.2e | %.1f s\n', ...
            it, lossHist(it), lrE, lrD, toc);
    end
    % track the best (lowest 100-iter moving average) checkpoint
    if it >= 100
        ma = mean(lossHist(max(1,it-99):it));
        if it == 100 || ma < bestMA
            bestMA = ma;
            best = struct('params', params, 'it', it, 'ma', ma);
            save(ckptBest, 'best', 'P', 'lossHist');
        end
    end
    if mod(it, 1000) == 0 || it == P.iters
        save(ckptLast, 'params', 'P', 'lossHist', 'it');
    end
    % periodic preview grid (5 samples per class) for progress monitoring
    if mod(it, 2000) == 0 || it == P.iters
        pv = zeros(P.n, P.n, P.nClass, 5, 'single');
        for c = 1:P.nClass
            z  = dlarray(randn(P.n, P.n, 5, 'single'));
            if isgpuarray(extractdata(params.Dec)), z = gpuArray(z); end
            I  = optical_forward(params, z, c*ones(5,1), P);
            On = gather(extractdata(I));
            On = On ./ (mean(On, [1 2]) + eps('single')) .* single(inkFracAll);
            pv(:,:,c,:) = reshape(min(max(On, 0), 1), P.n, P.n, 1, 5);
        end
        pimg = im2uint8(pv);
        bigP = zeros(P.n*P.nClass, P.n*5, 'uint8');
        for c = 1:P.nClass
            for j = 1:5
                bigP((c-1)*P.n+1:c*P.n, (j-1)*P.n+1:j*P.n) = pimg(:,:,c,j);
            end
        end
        close all force;
        figure('Color','w','Visible','off');
        imshow(bigP);
        title(sprintf('%s | iter %d | loss %.4f', ds, it, lossHist(it)));
        print('-dpng', '-r200', figPrev);
    end
end
% final checkpoint: use the best one for generation
if isstruct(best), params = best.params; end
save(ckptBest, 'params', 'P', 'lossHist', 'it');
if strcmp(ds, 'mnist')
    copyfile(ckptBest, 'checkpoint_optical_gen.mat');   % keep demo scripts working
end

%% --------------------------- loss figure -----------------------------
figure('Color','w');
plot(movmean(lossHist, 25), 'LineWidth', 1.2); grid on;
xlabel('Iteration'); ylabel('Loss (MSE + \gamma KL)');
title(sprintf('Snapshot optical generative model (%s) - training loss', ds));
print('-dpng', '-r200', figLoss);

%% --------------------------- generation demo -------------------------
% 10 x 10 grid: one row per digit class (class-conditional generation).
% Intensity is normalized to the dataset ink level for display.
inkFrac = inkFracAll;
nShow = 10;
gen = zeros(P.n, P.n, nShow, nShow, 'single');
for c = 1:nShow
    z  = dlarray(randn(P.n, P.n, nShow, 'single'));
    if isgpuarray(extractdata(params.Dec)), z = gpuArray(z); end
    cl = c * ones(nShow, 1);
    I  = optical_forward(params, z, cl, P);
    On = gather(extractdata(I));
    On = On ./ (mean(On, [1 2]) + eps('single')) .* single(inkFrac);
    gen(:,:,c,:) = reshape(min(max(On, 0), 1), P.n, P.n, 1, nShow);
end
genImg = im2uint8(gen);
big = zeros(P.n*nShow, P.n*nShow, 'uint8');
for c = 1:nShow
    for j = 1:nShow
        big((c-1)*P.n+1:c*P.n, (j-1)*P.n+1:j*P.n) = genImg(:,:,c,j);
    end
end
figure('Color','w');
imshow(big); title(sprintf('Class-conditional optical generation (%s, one row per class)', ds));
print('-dpng', '-r200', figGen);
savefig([figGen(1:end-4) '.fig']);

%% --------------------------- local functions -------------------------
function params = init_params(P)
    m = P.encSize;
    % He init for FC layers
    params.Enc.W1 = dlarray(single(randn(m, m) * sqrt(2/m)));
    params.Enc.b1 = dlarray(single(zeros(m, 1)));
    params.Enc.W2 = dlarray(single(randn(m, m) * sqrt(2/m)));
    params.Enc.b2 = dlarray(single(zeros(m, 1)));
    params.Enc.W3 = dlarray(single(randn(P.n*P.n, m) * sqrt(2/m)));
    params.Enc.b3 = dlarray(single(zeros(P.n*P.n, 1)));
    params.Enc.emb = dlarray(single(0.5 * randn(P.nClass, 32)));
    params.Enc.M3  = dlarray(single(0.2 * randn(P.n*P.n, 32)));  % direct class->output path
    params.Enc.s  = dlarray(single(1.0));
    % decoding layer: random phase in [0, 2*pi)
    params.Dec = dlarray(single(2*pi * rand(P.Ndec)));
end

function [loss, grads] = modelLoss(params, tgt, cls, P)
    B = size(tgt, 4);
    z = dlarray(randn(P.n, P.n, B, 'single'));
    if isgpuarray(extractdata(params.Dec)), z = gpuArray(z); end
    I = optical_forward(params, z, cls, P);        % raw 32x32x B intensity

    % Targets in [0, 1]
    tgt01 = (tgt + 1) / 2;

    % Energy-conserving normalization: total power of each generated image
    % is scaled to the target ink level (paper normalizes by Omax; with the
    % max rule a "dark image + single bright spike" is a degenerate optimum
    % that stalls CPU-scale training, so we normalize by total power instead.
    % Physically this fixes the output diffraction efficiency.)
    P.inkFrac = mean(tgt01, 'all');                % scalar, data-dependent only
    denom = max(mean(I, [1 2]), single(1e-3));     % guard against light escaping
    On = I ./ denom .* P.inkFrac;
    On = min(max(On, 0), 1);                       % clamp to valid intensity range

    % pixel-wise MSE, Eq. (7)
    lMSE = mean((On - tgt01).^2, 'all');

    % KL divergence of intensity histograms, Eq. (7) (bins over [0, 1])
    pOpt = soft_hist(On, P.klBins);
    pTgt = soft_hist(tgt01, P.klBins);
    lKL  = sum(pOpt .* log((pOpt + 1e-6) ./ (pTgt + 1e-6)));

    loss = lMSE + P.gamma * lKL;

    grads.Enc = dlgradient(loss, params.Enc);
    grads.Dec = dlgradient(loss, params.Dec);
end

function g = clip_grads(g, maxNorm)
    % global-norm gradient clipping over a nested struct of dlarrays
    sq = 0;
    f = fieldnames(g);
    for q = 1:numel(f)
        v = g.(f{q});
        if isstruct(v)
            f2 = fieldnames(v);
            for r = 1:numel(f2)
                d = extractdata(v.(f2{r}));
                sq = sq + sum(double(d(:)).^2);
            end
        else
            d = extractdata(v);
            sq = sq + sum(double(d(:)).^2);
        end
    end
    n = sqrt(sq);
    if ~isfinite(n)
        g = zero_grads(g);                   % poison guard: zero the grads
        return;
    end
    if n > maxNorm
        sc = single(maxNorm / (n + 1e-12));
        for q = 1:numel(f)
            v = g.(f{q});
            if isstruct(v)
                f2 = fieldnames(v);
                for r = 1:numel(f2)
                    g.(f{q}).(f2{r}) = v.(f2{r}) .* sc;
                end
            else
                g.(f{q}) = v .* sc;
            end
        end
    end
end

function g = zero_grads(g)
    f = fieldnames(g);
    for q = 1:numel(f)
        v = g.(f{q});
        if isstruct(v)
            f2 = fieldnames(v);
            for r = 1:numel(f2)
                d = extractdata(v.(f2{r}));
                z = zeros(size(d), 'like', d);
                g.(f{q}).(f2{r}) = dlarray(z);
            end
        else
            d = extractdata(v);
            z = zeros(size(d), 'like', d);
            g.(f{q}) = dlarray(z);
        end
    end
end
