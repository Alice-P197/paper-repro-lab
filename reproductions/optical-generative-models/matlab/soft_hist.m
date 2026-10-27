function softp = soft_hist(x, nb)
%SOFT_HIST  Differentiable (soft-binned) histogram of values in x over [0,1].
%   Each value is linearly distributed between its two nearest bin centres,
%   so the gradient flows through the interpolation weights.
%   Returns normalized probability vector (nb x 1).

    c1 = 0; c2 = 1; bw = (c2 - c1) / (nb - 1);

    v = x(:);
    nv = numel(v);

    % numeric bin indices (constants, no gradient needed)
    vd = double(gather(extractdata(v)));
    tn = min(max((vd - c1) / bw + 1, 1), nb);
    i0 = min(max(floor(tn), 1), nb);
    i1 = min(i0 + 1, nb);

    % one-hot assignment matrices on the same device as x
    onGPU = isgpuarray(extractdata(v));
    A0 = zeros(nv, nb, 'single');  A1 = A0;
    lin = (1:nv)';
    A0(sub2ind([nv nb], lin, i0)) = 1;
    A1(sub2ind([nv nb], lin, i1)) = 1;
    if onGPU, A0 = gpuArray(A0); A1 = gpuArray(A1); end %#ok<*GVMIS>

    % soft weights (gradient flows through w)
    t = (v - c1) / bw + 1;
    t = min(max(t, 1), nb);
    w = t - floor(t);                       % nv x 1 dlarray in [0,1)

    h = sum((1 - w) .* A0 + w .* A1, 1)';   % nb x 1
    softp = h / (sum(h) + eps(single(1)));
end
