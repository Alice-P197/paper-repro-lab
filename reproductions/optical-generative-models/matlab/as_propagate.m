function Uout = as_propagate(Uin, d, lambda, dx)
%AS_PROPAGATE  Free-space propagation by the angular spectrum method.
%   Paper Eq. (2)-(3):  U_out = F^-1{ F{U_in} .* M(fx,fy;;d) }
%
%   Uin    : complex array, h x w x B (batch along the 3rd dimension)
%   d      : axial propagation distance [m]
%   lambda : wavelength [m]
%   dx     : lateral pixel pitch [m]
%
%   Evanescent components (fx^2+fy^2 > 1/lambda^2) are set to zero, as in
%   Eq. (3). The transfer function is cached (it depends only on h, w, d).

    persistent cache
    if isempty(cache), cache = containers.Map('KeyType','char','ValueType','any'); end

    [h, w, ~] = size(Uin);
    key = sprintf('%dx%d_d%.9g_lam%.9g_dx%.9g', h, w, d, lambda, dx);

    if isKey(cache, key)
        M = cache(key);
    else
        fx = (-floor(w/2):ceil(w/2)-1) / (w*dx);     % spatial frequencies [1/m]
        fy = (-floor(h/2):ceil(h/2)-1) / (h*dx);
        [FX, FY] = meshgrid(fx, fy);

        k = 2*pi/lambda;
        arg = 1 - (lambda*FX).^2 - (lambda*FY).^2;   % 1 - (fx^2+fy^2)/fc^2
        valid = arg > 0;
        M = complex(zeros(h, w, 'single'));
        M(valid) = single(exp(1i * k * d * sqrt(arg(valid))));
        M = ifftshift(M);
        cache(key) = M;
        clear FX FY arg valid
    end

    if isgpuarray(extractdata_compat(Uin)) && ~isgpuarray(M)
        M = gpuArray(M);
    end

    % fft2 on complex dlarray requires batch dim > 1: duplicate a B=1 batch
    if size(Uin, 3) == 1 && isa(Uin, 'dlarray') && ~isreal(extractdata_compat(Uin))
        U2 = cat(3, Uin, Uin);
        Uout = ifft2( fft2(U2) .* M );
        Uout = Uout(:, :, 1);
    else
        Uout = ifft2( fft2(Uin) .* M );              % fft2 acts on dims 1-2
    end
end

function y = extractdata_compat(x)
    if isa(x, 'dlarray'), y = extractdata(x); else, y = x; end
end
