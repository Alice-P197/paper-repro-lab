%% TRAIN_ENCODER_DISTILL  Stage 2: distill prototype seeds into the encoder.
%  Pure digital regression: (32x32 Gaussian noise, class embedding) -> phase
%  seed, supervised by the Stage-1 prototype seeds. No optics in the loss, so
%  this trains in minutes. The final system architecture is exactly the
%  paper's: digital encoder (Eq. 8) -> phase encoding (Eq. 1) -> optical
%  decoder (Eq. 4-5).
%
%  Diversity: each sample's target is a random FIELD-space mixture of the K
%  prototype seeds of its class (complex superposition, then phase), so the
%  encoder interpolates the class's seed manifold smoothly as z varies.
%
%  Usage:  OGEN_DATASET=mnist matlab -batch run('train_encoder_distill.m')

clear; clc; close all;
rng(0);

ds = getenv('OGEN_DATASET'); if isempty(ds), ds = 'mnist'; end
S = load(sprintf('prototypes_%s.mat', ds));
seeds = S.seeds;  Dec = S.Dec;  P = S.P;  K = S.K;
nProt = size(seeds, 3);
fprintf('%s: %d prototype seeds loaded\n', ds, nProt);

iters = 15000;  batch = 128;  lr = 3e-4;
if ~isempty(getenv('DISTILL_ITERS')), iters = str2double(getenv('DISTILL_ITERS')); end

%% parameters (same architecture as the paper's encoder, Eq. 8)
m  = P.n*P.n + 32;
params.Enc.W1 = dlarray(single(randn(m, m) * sqrt(2/m)));
params.Enc.b1 = dlarray(single(zeros(m, 1)));
params.Enc.W2 = dlarray(single(randn(m, m) * sqrt(2/m)));
params.Enc.b2 = dlarray(single(zeros(m, 1)));
params.Enc.W3 = dlarray(single(randn(P.n*P.n, m) * sqrt(2/m)));
params.Enc.b3 = dlarray(single(zeros(P.n*P.n, 1)));
params.Enc.emb = dlarray(single(0.5 * randn(P.nClass, 32)));
params.Enc.M3  = dlarray(single(0.2 * randn(P.n*P.n, 32)));
params.Enc.s   = dlarray(single(1.0));
params.Dec     = dlarray(single(Dec));  % frozen: keep the fitted decoder
aG = []; aS = [];

%% distillation loop (digital only)
lossHist = zeros(iters, 1);
tic;
for it = 1:iters
    lrf = 0.5 * (1 + cos(pi * it / iters));
    cls = randi(P.nClass, batch, 1);
    w   = single(rand(batch, K));  w = w ./ sum(w, 2);   % convex mixture
    [loss, grads] = dlfeval(@dloss, params, cls, w, seeds, K, P);
    lossHist(it) = double(gather(extractdata(loss)));
    if ~isfinite(lossHist(it)), fprintf('iter %d: non-finite loss, skipping\n', it); continue; end
    grads = clipgrad(grads, 1.0);
    [params, aG, aS] = adamupdate(params, grads, aG, aS, it, lr * lrf);
    if mod(it, 200) == 0 || it == 1
        fprintf('iter %5d | circular seed MSE %.5f | %.1f s\n', it, lossHist(it), toc);
    end
end

%% save in the checkpoint format used by generate_images / interpolate_latent
save(sprintf('checkpoint_distilled_%s.mat', ds), 'params', 'P', 'lossHist', 'ds', 'seeds', 'Dec');
fprintf('saved checkpoint_distilled_%s.mat\n', ds);

%% evaluation grid: 10 classes x 8 noise samples, through the full optics
z = dlarray(randn(P.n, P.n, 8, 'single'));
gen = zeros(P.n, P.n, P.nClass, 8, 'single');
for c = 1:P.nClass
    O = optical_forward(params, z, c*ones(8,1), P);
    On = gather(extractdata(O));
    for j = 1:8
        o = On(:,:,j);
        o = o ./ (mean(o,[1 2]) + eps('single')) .* single(P.inkFracAll);
        gen(:,:,c,j) = min(max(o, 0), 1);
    end
end
img = reshape(gen, P.n, []);
figure('Color','w','Visible','off'); imshow(img);
title(sprintf('Distilled encoder + optical decoder (%s): one row per class', ds));
print('-dpng', '-r150', sprintf('fig_distilled_%s.png', ds));
fprintf('done\n');

%% --------------------------- local -----------------------------------
function g = clipgrad(g, maxNorm)
    f = fieldnames(g.Enc);
    sq = 0;
    for q = 1:numel(f)
        v = g.Enc.(f{q});
        sq = sq + sum(double(extractdata(v)).^2, 'all');
    end
    n = sqrt(sq);
    if isfinite(n) && n > maxNorm
        sc = single(maxNorm / (n + 1e-12));
        for q = 1:numel(f)
            g.Enc.(f{q}) = g.Enc.(f{q}) .* sc;
        end
    end
end

function [loss, grads] = dloss(params, cls, w, seeds, K, P)
    B = numel(cls);
    z = dlarray(randn(P.n, P.n, B, 'single'));
    % target: field-space mixture of the K seeds of each sample's class
    base = (cls - 1) * K;                            % B x 1
    idx  = repmat((1:K)', 1, B) + repmat(reshape(base, 1, B), K, 1);  % K x B
    E    = exp(1i * single(reshape(seeds(:,:,idx), P.n, P.n, K, B)));
    wE   = reshape(w, [1 1 K B]);
    T    = angle(squeeze(sum(E .* wE, 3)));          % 32 x 32 x B
    % encoder output: 320x320 upsampled nearest -> sample back to 32x32
    phiP = encoder_phase(params, z, cls, P);         % 320 x 320 x B
    phiP32 = phiP(1:P.up:end, 1:P.up:end, :);
    % wrap-safe smooth phase loss: sin^2(d/2) in [0,1], bounded gradient
    loss = 4 * mean(sin((phiP32 - dlarray(T))/2).^2, 'all');
    grads = dlgradient(loss, params);
end
