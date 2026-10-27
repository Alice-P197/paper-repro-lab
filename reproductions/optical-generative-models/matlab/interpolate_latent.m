function interpolate_latent(ckptFile, nSteps, c1, c2)
%INTERPOLATE_LATENT  Latent-space interpolation of a snapshot optical
%   generative model (Extended Data Fig. 5 / Supplementary Fig. 26).
%
%   J_g = g*J1 + (1-g)*J2, and the class embedding is interpolated the same
%   way. The interpolated inputs are encoded into optical generative seeds
%   and decoded by the fixed diffractive decoder.
%
%   interpolate_latent(ckptFile, nSteps, c1, c2)

    if nargin < 1 || isempty(ckptFile), ckptFile = 'checkpoint_optical_gen.mat'; end
    if nargin < 2, nSteps = 8; end
    if nargin < 3, c1 = 3; end
    if nargin < 4, c2 = 8; end

    S = load(ckptFile);
    params = S.params;  P = S.P;
    useGPU = isgpuarray(extractdata(params.Dec));

    nRows = 6;
    gam = linspace(0, 1, nSteps);
    tiles = zeros(P.n, P.n, nRows, nSteps, 'single');
    inkFrac = 0.16;

    emb = gather(extractdata(params.Enc.emb));   % 10 x 32 host copy
    e1 = emb(c1, :);  e2 = emb(c2, :);

    for r = 1:nRows
        J1 = randn(P.n, P.n, 'single');
        J2 = randn(P.n, P.n, 'single');
        for k = 1:nSteps
            % input noise and class embedding are interpolated identically
            Jg = gam(k)*J1 + (1-gam(k))*J2;
            eg = gam(k)*e1 + (1-gam(k))*e2;      % 1 x 32
            z  = dlarray(reshape(Jg, P.n, P.n, 1));
            ec = dlarray(single(eg'));
            if useGPU, z = gpuArray(z); ec = gpuArray(ec); end
            O  = optical_forward(params, z, ec, P);
            On = gather(extractdata(O));
            On = On ./ (mean(On, [1 2]) + eps('single')) .* single(inkFrac);
            tiles(:,:,r,k) = min(max(On, 0), 1);
        end
    end

    img = im2uint8(tiles);
    img = uint8(255 * (double(img)/255).^0.454);      % gamma 0.454 (Methods)

    big = zeros(P.n*nRows, P.n*nSteps, 'uint8');
    for r = 1:nRows
        for k = 1:nSteps
            big((r-1)*P.n+1:r*P.n, (k-1)*P.n+1:k*P.n) = img(:,:,r,k);
        end
    end
    figure('Color','w'); imshow(big);
    title(sprintf('Latent interpolation: digit %d -> %d', c1-1, c2-1));
    print('-dpng', '-r200', 'fig_interpolation.png');
    fprintf('Saved fig_interpolation.png\n');
end
