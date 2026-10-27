function phi = encoder_phase(params, noise, cls, P)
%ENCODER_PHASE  Shallow digital phase encoder -> optical generative seed.
%   Paper Eq. (1) & (8): flatten(noise) (+ class embedding) -> 3 FC layers
%   (LeakyReLU 0.2 on the first two) -> normalized H -> phase seed.
%
%   params : struct with fields W1,b1,W2,b2,W3,b3,s (all dlarray)
%   noise  : 32 x 32 x B (dlarray, Gaussian)
%   cls    : B x 1 class indices (1..10), numeric double
%   P      : config struct
%
%   phi    : 320 x 320 x B encoded phase pattern (rad)

    B = size(noise, 3);
    enc = params.Enc;
    % Eq. (8): x(0) = [flatten(I_xy), embedding(C)]  (l = 32)
    % cls: class indices (numeric B x 1) or a dlarray embedding (32 x B),
    % which is used for latent-space interpolation of the class condition.
    if isa(cls, 'dlarray')
        e = cls;                         % already a 32 x B embedding
    else
        e = enc.emb(cls, :)';            % 32 x B (dlarray, 10 x 32 embedding)
    end
    x = reshape(noise, [], B);           % 1024 x B
    x = cat(1, x, e);                    % 1056 x B   (Eq. 8)

    % 3 fully connected layers, LeakyReLU(0.2) after first two (Methods).
    % FiLM-lite class conditioning: the embedding additionally feeds the
    % output layer through a learned linear map (direct additive path), so
    % class information cannot be drowned by the 1024-dim noise input.
    x = leakyrelu(enc.W1*x + enc.b1, 0.2);
    x = leakyrelu(enc.W2*x + enc.b2, 0.2);
    H = enc.W3*x + enc.b3 + enc.M3*e;    % 1024 x B

    % Learnable scaling factor s (Methods, "predicted scaling factor")
    H = tanh(H) .* tanh(enc.s);

    % Eq. (1): phi = alpha*pi*(H + 1)/2  -> [0, alpha*pi], alpha = 2
    h32 = reshape(H, P.n, P.n, B);
    phi = (P.alpha*pi/2) * (h32 + 1);

    % Up-sample the normalized profile by 10x in x and y (Methods):
    % 32x32 -> 320x320 at 8 um pitch (object plane 2.56 mm x 2.56 mm)
    phi = upsample_phase(phi, P.up);
end

function y = leakyrelu(x, a)
    y = max(x, 0) + a * min(x, 0);
end

function y = upsample_phase(x, f)
    % nearest-neighbour f x upsampling of h x w x B
    y = repelem(x, f, f, 1);
end
