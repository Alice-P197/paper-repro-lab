function Iout = optical_forward(params, noise, cls, P)
%OPTICAL_FORWARD  Full forward model of the snapshot optical generative model.
%   Paper Eq. (1)-(5): encoder phase seed -> AS propagation -> diffractive
%   decoding layer -> AS propagation -> sensor intensity.
%
%   Returns Iout: 32 x 32 x B raw intensity after 10x downsampling
%   (normalization is done in the loss, see train_optical_gen.m).

    % 1) Encoded phase pattern (optical generative seed), Eq. (1) & (8)
    phi = encoder_phase(params, noise, cls, P);      % 320 x 320 x B

    % 2) Incident complex field U0 = cos(phi) + 1i*sin(phi) (Methods).
    %    Outside the illuminated SLM area the field is ZERO (no light) —
    %    padding the phase with zeros would inject a spurious flat-field
    %    annulus whose seed-independent diffraction dominates the sensor.
    U = pad_field_exp(phi, P.Ndec, P.padOff);        % 400 x 400 x B

    % 3) Propagate SLM -> decoding layer (d0,1), Eq. (2)
    U = as_propagate(U, P.d01, P.lambda, P.dx);

    % 4) Phase-only decoding layer, Eq. (4): U .* exp(1j*phi_dec)
    dec3 = reshape(params.Dec, P.Ndec, P.Ndec, 1);   % broadcast over the batch
    U = U .* exp(1i * dec3);

    % 5) Propagate decoding layer -> sensor plane (d1,2), Eq. (2)
    U = as_propagate(U, P.d12, P.lambda, P.dx);

    % 6) Sensor intensity, Eq. (5): O = |U|^2
    I = real(U).^2 + imag(U).^2;

    % 7) Crop central sensor region (320 x 320 = seed resolution), then
    %    downsample by 10x for the loss (Methods)
    ns = P.n * P.up;                                 % 320 x 320 sensor pixels
    I = I(P.padOff+1:P.padOff+ns, P.padOff+1:P.padOff+ns, :);
    Iout = blockmean(I, P.up);                       % 32 x 32 x B
end

function y = pad_field_exp(phi, N, off)
    % embed exp(1i*phi) into a zero (dark) N x N canvas at offset (off, off)
    inner = exp(1i * phi);
    y = zeros(N, N, size(phi,3), 'like', inner);
    y(off+1:off+size(phi,1), off+1:off+size(phi,2), :) = inner;
end

function y = blockmean(x, f)
    % f x f average pooling of h x w x B  (h, w divisible by f)
    [h, w, B] = size(x);
    x = reshape(x, h/f, f, w/f, f, B);
    y = squeeze(mean(mean(x, 2), 4));
    y = reshape(y, h/f, w/f, B);
end
