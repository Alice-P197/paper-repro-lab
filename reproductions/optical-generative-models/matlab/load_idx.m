function [X, Y] = load_idx(dataDir, prefix)
%LOAD_IDX  Parse IDX-format MNIST/Fashion-MNIST gzip archives.
%   Returns X: 28x28xN single in [0,1], Y: 1xN double labels 0..9.

    tmp = tempname; mkdir(tmp);
    cleanup = onCleanup(@() cleanup_dir(tmp));

    fi = gunzip(fullfile(dataDir, [prefix '-images-idx3-ubyte.gz']), tmp);
    fl = gunzip(fullfile(dataDir, [prefix '-labels-idx1-ubyte.gz']), tmp);

    fid = fopen(fi{1}, 'r');
    assert(fread(fid, 1, 'int32', 0, 'b') == 2051, 'bad idx3 magic');
    n  = fread(fid, 1, 'int32', 0, 'b');
    r  = fread(fid, 1, 'int32', 0, 'b');
    c  = fread(fid, 1, 'int32', 0, 'b');
    X  = fread(fid, n*r*c, 'uint8=>single', 0, 'b') / 255;
    X  = reshape(X, r, c, n);
    X  = permute(X, [2 1 4 3]);        % idx is row-major -> transpose; make 28x28x1xN
    fclose(fid);

    fid = fopen(fl{1}, 'r');
    assert(fread(fid, 1, 'int32', 0, 'b') == 2049, 'bad idx1 magic');
    fread(fid, 1, 'int32', 0, 'b');
    Y = double(fread(fid, n, 'uint8', 0, 'b'))';
    fclose(fid);
end

function cleanup_dir(d)
    try, rmdir(d, 's'); catch, end   % best effort; locked files are ignorable
end
