function Iout = forward_from_phase(phi, params, P)
%FORWARD_FROM_PHASE  Optical forward model from a 32 x 32 phase seed.
%   Paper Eq. (2)-(5): phase seed -> 10x upsample -> AS propagation ->
%   diffractive decoding layer -> AS propagation -> sensor intensity.
%
%   phi   : 32 x 32 x B phase seed [rad]
%   params: struct with field Dec (Ndec x Ndec learnable phase)
%   Returns Iout: 32 x 32 x B raw intensity (normalization in the loss).

    % 1) Up-sample the seed profile by 10x (Methods)
    phi = repelem(phi, P.up, P.up, 1);               % 320 x 320 x B

    % 2) Embed in the 400 x 400 aperture; outside the illuminated area the
    %    field is zero (dark) — a flat-phase annulus would inject a spurious
    %    seed-independent background that dominates the sensor.
    U = pad_field_exp(phi, P.Ndec, P.padOff);

    % 3) Propagate SLM -> decoding layer (d0,1), Eq. (2)
    U = as_propagate(U, P.d01, P.lambda, P.dx);

    % 4) Phase-only decoding layer, Eq. (4)
    dec3 = reshape(params.Dec, P.Ndec, P.Ndec, 1);
    U = U .* exp(1i * dec3);

    % 5) Propagate decoding layer -> sensor plane (d1,2), Eq. (2)
    U = as_propagate(U, P.d12, P.lambda, P.dx);

    % 6) Sensor intensity, Eq. (5)
    I = real(U).^2 + imag(U).^2;

    % 7) Crop central sensor region and downsample by 10x (Methods)
    ns = P.n * P.up;
    I = I(P.padOff+1:P.padOff+ns, P.padOff+1:P.padOff+ns, :);
    Iout = blockmean(I, P.up);                       % 32 x 32 x B
end

function y = pad_field_exp(phi, N, off)
    inner = exp(1i * phi);
    y = zeros(N, N, size(phi,3), 'like', inner);
    y(off+1:off+size(phi,1), off+1:off+size(phi,2), :) = inner;
end

function y = blockmean(x, f)
    [h, w, B] = size(x);
    x = reshape(x, h/f, f, w/f, f, B);
    y = squeeze(mean(mean(x, 2), 4));
    y = reshape(y, h/f, w/f, B);
end
