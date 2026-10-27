%% GENERATE_MIXTURE  Class-conditional generation by field-mixture sampling.
%  Samples random field-space mixtures of each class's Stage-1 prototype
%  seeds (physically: superposed SLM phase patterns), pushes them through
%  the fitted optical decoder, and renders a Fig-3c-style grid.
%
%  Usage:  OGEN_DATASET=mnist matlab -batch run('generate_mixture.m')

clear; clc; close all;
rng(1);

ds = getenv('OGEN_DATASET'); if isempty(ds), ds = 'mnist'; end
S = load(sprintf('prototypes_%s.mat', ds));
seeds = S.seeds;  Dec = S.Dec;  P = S.P;  K = S.K;
P.nClass = 10;
nProt = size(seeds, 3);
nPer  = 8;                       % generated samples per class
fprintf('%s: %d seeds | %d samples per class\n', ds, nProt, nPer);

%% sample mixtures and push through the optics (batched, B = 10*nPer)
Btot = P.nClass * nPer;
mixseed = zeros(P.n, P.n, Btot, 'single');
for c = 1:P.nClass
    base = (c-1)*K;
    for j = 1:nPer
        b = (c-1)*nPer + j;
        w  = single(rand(K,1));  w = w ./ sum(w);          % random mixture
        acc = zeros(P.n, P.n, 'single');
        for k = 1:K
            acc = acc + w(k) * exp(1i * seeds(:,:,base+k));  % field superposition
        end
        mixseed(:,:,b) = angle(acc);
    end
end

I = forward_from_phase(dlarray(mixseed), struct('Dec', dlarray(single(Dec))), P);
In = gather(extractdata(I));

%% normalize and render
gen = zeros(P.n, P.n, Btot, 'single');
for b = 1:Btot
    o = In(:,:,b);
    gen(:,:,b) = min(max(o ./ (mean(o,[1 2]) + eps('single')) .* single(P.inkFracAll), 0), 1);
end
% per-image contrast stretch for display (keep relative ink levels)
genDisp = gen;
for b = 1:Btot
    m = max(genDisp(:,:,b), [], 'all');
    if m > 0, genDisp(:,:,b) = genDisp(:,:,b) / m; end
end
big = zeros(P.nClass*96, (P.n*3+4)*nPer, 'single');
sc  = @(a) imresize(a, 3, 'nearest');
for c = 1:P.nClass
    row = [];
    for j = 1:nPer
        b = (c-1)*nPer + j;
        row = [row, sc(genDisp(:,:,b)), nan(96,4)];  %#ok<AGROW>
    end
    big((c-1)*96+1:c*96, 1:size(row,2)) = row;  %#ok<AGROW>
end
figure('Color','w','Visible','off','Position',[50 50 1800 1000]);
imshow(big, 'InitialMagnification', 'fit');
title(sprintf('%s: field-mixture generation, one row per class (%d samples each)', ds, nPer));
print('-dpng', '-r150', sprintf('fig_generation_%s.png', ds));
save(sprintf('generation_%s.mat', ds), 'gen', 'mixseed', 'P');
fprintf('saved fig_generation_%s.png\n', ds);
