%% TRAIN_PROTOTYPES_GS  Gerchberg-Saxton solver for the optical prototypes.
%  Stage 1 (CPU-scale strategy): solve the 50 phase seeds directly with a
%  double-constraint iterative algorithm (no autodiff, fully batched, fast).
%
%  Forward operator (paper Eq. 2-5), exactly matching forward_from_phase.m:
%  seed phase (32x32, amplitude 1) -> 10x upsample -> embed in 400x400
%  dark aperture (zero outside) -> AS propagation d01 -> decoding phase
%  Dec -> AS propagation d12 -> sensor intensity (crop + 10x downsample).
%
%  Constraints: seed plane = phase-only (amplitude 1, annulus phase 0);
%  sensor plane amplitude = sqrt(target) inside the central 320x320 window.
%
%  The decoder Dec is loaded from the task-stream snapshot if available.

clear; clc; close all;
rng(0);

P.n      = 32;  P.up = 10;  P.dx = 8e-6;  P.lambda = 520e-9;
P.d01    = 120.1e-3;  P.d12 = 96.4e-3;
P.Ndec   = 400;  P.padOff = floor((P.Ndec - P.n*P.up)/2);
P.nClass = 10;
K        = 5;                       % prototype seeds per class
nSweeps  = 80;
ds = getenv('OGEN_DATASET'); if isempty(ds), ds = 'mnist'; end
fprintf('GS solver | dataset %s | K=%d/class | %d sweeps\n', ds, K, nSweeps);

%% data
[X, Y] = load_idx(fullfile('..', 'data', ds), 'train');
X = imresize(X, [P.n P.n]);  X = 2*X - 1;
Y = double(Y(:));

mix = 0.5;                            % class-mean + sample blend
muClass = zeros(P.n, P.n, 1, P.nClass, 'single');
for c = 1:P.nClass, muClass(:,:,:,c) = mean(X(:,:,:,Y == c-1), 4); end
nProt = P.nClass * K;
tgt   = zeros(P.n, P.n, 1, nProt, 'single');
for c = 1:P.nClass
    pool = find(Y == c-1);
    pick = pool(randperm(numel(pool), K));
    for j = 1:K
        tgt(:,:,:, (c-1)*K+j) = (1-mix)*muClass(:,:,:,c) + mix*X(:,:,:,pick(j));
    end
end
tgt01 = (tgt + 1) / 2;
tgtMean = mean(tgt01, [1 2]);

%% decoder
if isfile('prototypes_progress.mat')
    Spr = load('prototypes_progress.mat', 'best');
    Dec = real(extractdata(Spr.best.pr.Dec));
    fprintf('decoder loaded from prototypes_progress.mat\n');
else
    Dec = 2*pi * rand(P.Ndec);
    fprintf('decoder: random init\n');
end
DecExp = exp(1i * single(Dec));
DecCon = conj(DecExp);

%% GS solver (batched over all prototypes), 400x400 grid
N   = P.Ndec;  ns = P.n * P.up;
off = P.padOff;
ampTgt = reshape(single(sqrt(max(repelem(tgt01, P.up, P.up, 1), 0))), ns, ns, []);  % 320 x 320 x nProt

M01p = as_mask(N,  P.d01, P.lambda, P.dx);
M01m = as_mask(N, -P.d01, P.lambda, P.dx);
M12p = as_mask(N,  P.d12, P.lambda, P.dx);
M12m = as_mask(N, -P.d12, P.lambda, P.dx);

seed = 2*pi * rand(P.n, P.n, nProt, 'single');
histMSE = zeros(nSweeps, 1);
tic;
for sw = 1:nSweeps
    % --- forward ---
    U = embed(exp(1i * repelem(seed, P.up, P.up, 1)), N, off);  % N x N x nProt
    U = ifft2( fft2(U) .* M01p );
    U = U .* DecExp;
    U = ifft2( fft2(U) .* M12p );
    % --- sensor constraint: amplitude = target inside central window ---
    win = U(off+1:off+ns, off+1:off+ns, :);
    U(off+1:off+ns, off+1:off+ns, :) = ampTgt .* exp(1i * angle(win));
    % --- backward ---
    U = ifft2( fft2(U) .* M12m );
    U = U .* DecCon;
    U = ifft2( fft2(U) .* M01m );
    % --- seed-plane constraint: phase-only, annulus flat ---
    inner = U(off+1:off+ns, off+1:off+ns, :);
    seed = angle( blockmean_cplx(inner, P.up) );
    % --- monitor ---
    I = gs_forward(seed, DecExp, M01p, M12p, P);
    In = I ./ max(mean(I,[1 2]), single(1e-6)) .* tgtMean;
    histMSE(sw) = mean((In - tgt01).^2, 'all');
    if mod(sw, 10) == 0, fprintf('sweep %2d | MSE %.4f | %.1f s\n', sw, histMSE(sw), toc); end
end
fprintf('final MSE %.4f\n', histMSE(end));

seeds = single(mod(seed, 2*pi));
save(sprintf('prototypes_%s.mat', ds), 'seeds', 'Dec', 'tgt', 'P', 'K', 'histMSE', 'ds', 'mix');
fprintf('saved prototypes_%s.mat\n', ds);

%% progress figure
I = gs_forward(seeds, DecExp, M01p, M12p, P);
gen = zeros(size(tgt), 'single');
for s = 1:nProt
    o = I(:,:,s);
    gen(:,:,:,s) = min(max(o ./ (mean(o,[1 2]) + eps('single')) .* tgtMean(:,:,:,s), 0), 1);
end
top = reshape(min(max(tgt01,0),1), P.n, []);
bot = reshape(gen,                 P.n, []);
figure('Color','w','Visible','off');
subplot(2,1,1); imshow(top); title(sprintf('%s targets (each block = one class, %d seeds)', ds, K));
subplot(2,1,2); imshow(bot); title('Optically fitted prototype seeds (GS)');
print('-dpng', '-r150', sprintf('fig_prototypes_%s.png', ds));
fprintf('done\n');

%% --------------------------- local -----------------------------------
function U = embed(inner, N, off)
    % dark aperture: zero field outside the illuminated 320x320 area
    U = zeros(N, N, size(inner,3), 'like', inner);
    U(off+1:off+size(inner,1), off+1:off+size(inner,2), :) = inner;
end

function M = as_mask(n, d, lambda, dx)
    fx = (-floor(n/2):ceil(n/2)-1) / (n*dx);
    fy = fx;
    [FX, FY] = meshgrid(fx, fy);
    k = 2*pi/lambda;
    arg = 1 - (lambda*FX).^2 - (lambda*FY).^2;
    M = complex(zeros(n, n, 'single'));
    M(arg > 0) = single(exp(1i * k * d * sqrt(arg(arg > 0))));
    M = ifftshift(M);
end

function I = gs_forward(seed, DecExp, M01, M12, P)
    U = embed(exp(1i * repelem(seed, P.up, P.up, 1)), P.Ndec, P.padOff);
    U = ifft2( fft2(U) .* M01 );
    U = U .* DecExp;
    U = ifft2( fft2(U) .* M12 );
    ns = P.n * P.up;
    I = abs(U(P.padOff+1:P.padOff+ns, P.padOff+1:P.padOff+ns, :)).^2;
    I = blockmean_cplx(I, P.up);
end

function y = blockmean_cplx(x, f)
    [h, w, B] = size(x);
    x = reshape(x, h/f, f, w/f, f, B);
    y = squeeze(mean(mean(x, 2), 4));
    y = reshape(y, h/f, w/f, B);
end
