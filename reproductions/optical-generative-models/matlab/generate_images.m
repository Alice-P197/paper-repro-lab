%% GENERATE_IMAGES  Load a trained snapshot optical generative model and
%  synthesize class-conditional digit images (paper Fig. 1a / Ext. Data Fig. 2a).
%
%  Usage:  generate_images            % loads checkpoint_optical_gen.mat
%          generate_images('file.mat', 20, 16)  % file, images per class, classes

function generate_images(ckptFile, nPerClass, nClass)
    if nargin < 1 || isempty(ckptFile), ckptFile = 'checkpoint_optical_gen.mat'; end
    if nargin < 2, nPerClass = 10; end
    if nargin < 3, nClass = 10; end

    S = load(ckptFile);
    params = S.params;  P = S.P;

    useGPU = isgpuarray(extractdata(params.Dec));

    gen = zeros(P.n, P.n, nClass, nPerClass, 'single');
    if isfield(P, 'inkFracAll'), inkFrac = P.inkFracAll; else, inkFrac = 0.16; end
    for c = 1:nClass
        z  = dlarray(randn(P.n, P.n, nPerClass, 'single'));
        if useGPU, z = gpuArray(z); end
        O  = optical_forward(params, z, c*ones(nPerClass,1), P);
        On = gather(extractdata(O));
        On = On ./ (mean(On, [1 2]) + eps('single')) .* single(inkFrac);
        gen(:,:,c,:) = reshape(min(max(On, 0), 1), P.n, P.n, 1, nPerClass);
    end

    img = im2uint8(gen);
    % gamma correction for display, as in the experiments (Gamma = 0.454)
    img = uint8(255 * (double(img)/255).^0.454);

    big = zeros(P.n*nClass, P.n*nPerClass, 'uint8');
    for c = 1:nClass
        for j = 1:nPerClass
            blk = img(:,:,c,j);
            big((c-1)*P.n+1:c*P.n, (j-1)*P.n+1:j*P.n) = blk;
        end
    end

    figure('Color','w'); imshow(big);
    title(sprintf('Optical generation, %s (rows = digit classes)', ckptFile));
    print('-dpng', '-r200', 'fig_generation.png');
    fprintf('Saved fig_generation.png\n');
end
