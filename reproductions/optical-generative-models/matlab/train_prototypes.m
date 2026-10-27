%% TRAIN_PROTOTYPES  Stage 1 of the CPU-scale reproduction strategy.
%  Direct hologram-style optimization: treat K phase seeds per class (plus the
%  shared diffractive decoder) as free parameters and fit them to real dataset
%  samples through the paper's optical forward model (Eq. 2-5). Gradients flow
%  directly to the seeds with no encoder indirection, which converges fast on
%  CPU. Stage 2 (train_encoder_distill.m) then distills the seeds into the
%  digital encoder, giving the full paper architecture.
%
%  Usage:  OGEN_DATASET=mnist OGEN_ITERS=2000 matlab -batch run('train_prototypes.m')

clear; clc; close all;
rng(0);

P.n      = 32;  P.up = 10;  P.dx = 8e-6;  P.lambda = 520e-9;
P.alpha  = 2.0;
P.d01    = 120.1e-3;  P.d12 = 96.4e-3;
P.Ndec   = 400;  P.padOff = floor((P.Ndec - P.n*P.up)/2);
P.nClass = 10;

K     = 5;                       % prototype seeds per class
iters = 2000;
lr    = 5e-3;
if ~isempty(getenv('OGEN_ITERS')), iters = str2double(getenv('OGEN_ITERS')); end
ds = getenv('OGEN_DATASET'); if isempty(ds), ds = 'mnist'; end
fprintf('Dataset %s | K=%d seeds/class | %d iters\n', ds, K, iters);

%% data
[X, Y] = load_idx(fullfile('..', 'data', ds), 'train');
X = imresize(X, [P.n P.n]);  X = 2*X - 1;
Y = double(Y(:));
inkFrac = mean((X+1)/2, 'all');
P.inkFracAll = inkFrac;

%% prototype targets: class mean blended with individual samples
%  Pure sample targets have fine strokes the optical bandwidth cannot fit
%  (loss stalls ~0.06); pure class means are smooth but leave no diversity.
%  mix = 0.5 balances both. OGEN_MIX overrides.
mix = 0.5;
if ~isempty(getenv('OGEN_MIX')), mix = str2double(getenv('OGEN_MIX')); end
muClass = zeros(P.n, P.n, 1, P.nClass, 'single');
for c = 1:P.nClass, muClass(:,:,:,c) = mean(X(:,:,:,Y == c-1), 4); end

nProt = P.nClass * K;
tgt   = zeros(P.n, P.n, 1, nProt, 'single');
clsOf = zeros(nProt, 1);
for c = 1:P.nClass
    pool = find(Y == c-1);
    pick = pool(randperm(numel(pool), K));
    for j = 1:K
        s = (c-1)*K + j;
        tgt(:,:,:,s) = (1-mix)*muClass(:,:,:,c) + mix*X(:,:,:,pick(j));
        clsOf(s) = c;
    end
end
tgt01 = (tgt + 1) / 2;
tgtMean = mean(tgt01, [1 2]);                    % 1 x 1 x nProt, per-sample brightness
fprintf('%d prototype targets (mix=%.2f) assigned\n', nProt, mix);

%% parameters: free phase seeds + shared decoder
pr.U   = dlarray(single(2*pi * rand(P.n, P.n, nProt)));  % direct phase params
pr.Dec = dlarray(single(2*pi * rand(P.Ndec)));
bestLoss = inf; best = struct();
lossHist = zeros(iters, 1);
tic;

%% PHASE A: task-stream cycling (B=1, U+Dec co-trained, small Dec lr)
nEpochs = 3;  T = 60;
lrDec   = 0.1 * lr;                        % slow decoder -> stable shared platform
aU = cell(nProt,1); aSU = cell(nProt,1);
aD = []; aSD = [];
tstep = 0; tic;
for ep = 1:nEpochs
    order = randperm(nProt);
    for kk = 1:nProt
        sIdx = order(kk);
        Uk = dlarray(pr.U(:,:,sIdx));
        tgt1 = tgt01(:,:,:,sIdx);  tm1 = tgtMean(:,:,:,sIdx);
        for tt = 1:T
            tstep = tstep + 1;
            [loss, g] = dlfeval(@ptask, Uk, pr.Dec, tgt1, tm1, P);
            lossHist(tstep) = double(gather(extractdata(loss))) / 1e6;
            [Uk, aU{sIdx}, aSU{sIdx}] = adamupdate(Uk, g.U, aU{sIdx}, aSU{sIdx}, tstep, lr);
            [pr.Dec, aD, aSD] = adamupdate(pr.Dec, g.Dec, aD, aSD, tstep, lrDec);
        end
        pr.U(:,:,sIdx) = gather(extractdata(Uk));
    end
    fprintf('phase A epoch %d | running loss %.4f | %.1f s\n', ep, mean(lossHist(max(1,tstep-nProt*T+1):tstep)), toc);
end

%% PHASE B: warm-start polish — freeze the decoder, re-fit each seed
T2 = 150;
lr2 = 0.5 * lr;
finalLoss = zeros(nProt,1);
for kk = 1:nProt
    Uk = dlarray(pr.U(:,:,kk));
    tgt1 = tgt01(:,:,:,kk);  tm1 = tgtMean(:,:,:,kk);
    for tt = 1:T2
        tstep = tstep + 1;
        [loss, g] = dlfeval(@ptask, Uk, pr.Dec, tgt1, tm1, P);
        lossHist(tstep) = double(gather(extractdata(loss))) / 1e6;
        [Uk, aU{kk}, aSU{kk}] = adamupdate(Uk, g.U, aU{kk}, aSU{kk}, tstep, lr2);
    end
    pr.U(:,:,kk) = gather(extractdata(Uk));
    finalLoss(kk) = lossHist(tstep);
end
fprintf('phase B done | mean final loss %.4f | max %.4f | %.1f s\n', ...
    mean(finalLoss), max(finalLoss), toc);

Dec   = real(gather(extractdata(pr.Dec)));
seeds = single(mod(real(gather(extractdata(pr.U))), 2*pi));   % [0, 2pi)
save(sprintf('prototypes_%s.mat', ds), 'seeds', 'Dec', 'tgt', 'clsOf', 'P', 'K', 'lossHist', 'finalLoss');
fprintf('saved prototypes_%s.mat\n', ds);

%% progress figure
I = forward_from_phase(dlarray(pr.U), struct('Dec', dlarray(single(Dec))), P);
In = gather(extractdata(I));
gen = zeros(size(tgt), 'single');
for s = 1:nProt
    o = In(:,:,s);
    gen(:,:,:,s) = min(max(o ./ (mean(o,[1 2]) + eps('single')) .* tgtMean(:,:,:,s), 0), 1);
end
top = reshape(min(max(tgt01,0),1), P.n, []);
bot = reshape(gen,                 P.n, []);
figure('Color','w','Visible','off');
subplot(2,1,1); imshow(top); title(sprintf('%s targets (each block = one class, %d seeds)', ds, K));
subplot(2,1,2); imshow(bot); title('Optically fitted prototype seeds');
print('-dpng', '-r150', sprintf('fig_prototypes_%s.png', ds));
fprintf('done\n');

seeds = single(mod(real(gather(extractdata(pr.U))), 2*pi));   % [0, 2pi)
Dec   = real(gather(extractdata(pr.Dec)));
save(sprintf('prototypes_%s.mat', ds), 'seeds', 'Dec', 'tgt', 'clsOf', 'P', 'K', 'lossHist', 'best');
fprintf('saved prototypes_%s.mat\n', ds);

%% progress figure: targets vs fitted prototypes
I = forward_from_phase(dlarray(seeds), struct('Dec', dlarray(Dec)), P);
In = gather(extractdata(I));
gen = zeros(size(tgt));
for s = 1:nProt
    o = In(:,:,s);
    o = o ./ (mean(o,[1 2]) + eps('single')) .* single(inkFrac);
    gen(:,:,:,s) = min(max(o, 0), 1);
end
top = reshape(min(max(tgt01,0),1), P.n, []);
bot = reshape(gen,                P.n, []);
figure('Color','w','Visible','off');
subplot(2,1,1); imshow(top); title(sprintf('%s targets (each block = one class, %d seeds)', ds, K));
subplot(2,1,2); imshow(bot); title('Optically fitted prototype seeds');
print('-dpng', '-r150', sprintf('fig_prototypes_%s.png', ds));
fprintf('done\n');

%% --------------------------- local -----------------------------------
function [loss, g] = ptask(Uk, Dec, tgt1, tm1, P)
    I = forward_from_phase(Uk, struct('Dec', Dec), P);
    On = I ./ max(mean(I,[1 2]), single(1e-6)) .* tm1;
    loss = 1e6 * mean((On - tgt1).^2, 'all');
    g = dlgradient(loss, struct('U', Uk, 'Dec', Dec));
    g.U = g.U;  %#ok<NASGU>
end

function [loss, grads] = ploss(pr, tgt01, tgtMean, P)
    phi = pr.U;                                      % direct phase (exp(1i*phi) is periodic, gradient never vanishes)
    I = forward_from_phase(phi, pr, P);              % 32 x 32 x nProt
    denom = max(mean(I, [1 2]), single(1e-6));
    On = I ./ denom .* tgtMean;                      % match each target's brightness
    loss = 1e6 * mean((On - tgt01).^2, 'all');       % scale up: float32 gradient of the raw loss (~1e-7) drowns in rounding noise
    grads = dlgradient(loss, pr);
end

function saveProt(fname, best, P, tgt, clsOf, ds, lossHist)
    seeds = single(mod(real(gather(extractdata(best.pr.U))), 2*pi));
    Dec   = real(gather(extractdata(best.pr.Dec)));
    save(fname, 'seeds', 'Dec', 'tgt', 'clsOf', 'P', 'ds', 'lossHist', 'best');
end
