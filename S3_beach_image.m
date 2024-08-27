%
% Codes reproducing the numerical examples of MPBTV, DGFM3, and GenLouvain
% [1] from [Section 7.3, 2] on image data created by the authors of [2].
% The same image has previously been studied in semi-supervised node
% classification [3]. We construct multiplex networks with L = 2 layers
% that separately record RGB channels and xy pixel coordinates. Identifying
% each pixel with a physical node, this separate treatment allows the
% construction of unweighted nearest-neighbor graphs with 40 neighbors in
% the RGB layer and 10 neighbors in the xy layer.
%
% References:
% [1] https://github.com/GenLouvain/GenLouvain
% [2] Kai Bergermann and Martin Stoll. Gradient flow-based modularity
%   maximization for community detection in multiplex networks, Preprint,
%   2024.
% [3] Kai Bergermann, Martin Stoll, and Toni Volkmer. Semi-supervised
%   learning for aggregated multilayer graphs using diffuse interface
%   methods and fast matrix-vector products. SIAM J. Math. Data Sci.,
%   3(2):758–785, 2021.
%
% Kai Bergermann, 2024
% 

addpath('GenLouvain-master/HelperFunctions')
addpath('GenLouvain-master/')
addpath('subroutines')

%% Choose resolution: '37_65', '37_65'
resolution = '37_65';

load(['networks/beach_',resolution,'_adj.mat'])

%% Set parameters
n = size(A{1},1);
L = 2;
nL = n*L;

gamma = 0.1*ones(L,1);
n_c = 4;
k_MPBTV = 9;
k_DGFM3 = 9;

omega = 10;
dt = 1;
max_iter = 300;
tolit = 1e-08;
n_runs = 100;

%% Assemble matrices
A_intra = blkdiag(A{1},A{2});
m_intra = [sum(sum(A{1}))/2, sum(sum(A{2}))/2];
d = cell(L,1); d{1} = sum(A{1},2); d{2} = sum(A{2},2);

A_inter = sparse(ones(L,L)) - speye(L);

L_supra = (spdiags(sum(A_intra,2),0,nL,nL) - A_intra) + omega*kron((diag(sum(A_inter,2)) - A_inter),speye(n,n));
C = omega*kron(A_inter,speye(n,n));

%% Offline computations
% Set up multiplex modularity matrix
[M_mult,twom] = multicat_f(A,gamma,omega);

% Compute eigenvalues and -vectors
tic
[phi_MPBTV,lambda_MPBTV] = eigs(@(v) MVprod(L_supra,gamma,m_intra,d,v), nL, k_DGFM3, 'smallestreal', 'SubspaceDimension', 3*k_MPBTV, 'IsFunctionSymmetric', true, 'IsSymmetricDefinite', true);
eig_time_MPBTV = toc;

tic
[phi_DGFM3,lambda_DGFM3] = eigs(@(v) MVprod_M_mult(A_intra,C,gamma,m_intra,d,v), nL, k_MPBTV, 'largestreal');
eig_time_DGFM3 = toc;

fprintf('\nRuntimes eigenvalue computations:\nMPBTV:\t%.3f sec\tk = %d\nDGFM3:\t%.3f sec\tk = %d\n', eig_time_MPBTV, k_MPBTV, eig_time_DGFM3, k_DGFM3)

%% Per run computations
rng(0)

acc_GenLouvain = zeros(n_runs,1); Q_GenLouvain = zeros(n_runs,1); NMI_GenLouvain = zeros(n_runs,1); runtime_GenLouvain = zeros(n_runs,1);
acc_DGFM3 = zeros(n_runs,1); Q_DGFM3 = zeros(n_runs,1); NMI_DGFM3 = zeros(n_runs,1); runtime_DGFM3 = zeros(n_runs,1);
acc_MPBTV = zeros(n_runs,1); Q_MPBTV = zeros(n_runs,1); NMI_MPBTV = zeros(n_runs,1); runtime_MPBTV = zeros(n_runs,1);

bar=waitbar(0,'Looping over runs...');
for i=1:n_runs
    % Initial conditions
    U0 = rand(nL,n_c);
    [~,U0_max_ind] = max(U0,[],2);
    U0(sub2ind(size(U0),1:size(U0,1),U0_max_ind')) = 1;
    U0(U0~=1) = 0;
    
    % GenLouvain
    tic
    [S,Q] = iterated_genlouvain(M_mult, 'verbose', false);
    runtime_GenLouvain(i) = toc;
    Q_GenLouvain(i) = getMultiplexModularity(S,A,gamma,omega);
    acc_GenLouvain(i) = getAccuracy(GT,S);
    NMI_GenLouvain(i) = getNMI(GT,S);

    % MPBTV
    tic
    [U_MPBTV,~] = MBO(U0, -lambda_MPBTV, phi_MPBTV, dt, max_iter, tolit);
    runtime_MPBTV(i) = toc;
    [~, S_MPBTV] = max(U_MPBTV,[],2);
    Q_MPBTV(i) = getMultiplexModularity(S_MPBTV,A,gamma,omega);
    acc_MPBTV(i) = getAccuracy(GT,S_MPBTV);
    NMI_MPBTV(i) = getNMI(GT,S_MPBTV);
    
    % DGFM3
    tic
    [U_DGFM3,it] = MBO(U0, lambda_DGFM3, phi_DGFM3, dt, max_iter, tolit);
    runtime_DGFM3(i) = toc;
    [~, S_DGFM3] = max(U_DGFM3,[],2);
    Q_DGFM3(i) = getMultiplexModularity(S_DGFM3,A,gamma,omega);
    acc_DGFM3(i) = getAccuracy(GT,S_DGFM3);
    NMI_DGFM3(i) = getNMI(GT,S_DGFM3);
    
    waitbar(i/n_runs,bar);
end
close(bar)

%% Print results
fprintf('\nBeach image network with %d runs, and %d layers.\n', n_runs, L)
fprintf('\t\tmax mod\tmax acc\tmax NMI\tavg runtime\n')
fprintf('MPBTV: \t\t%.3f \t%.3f \t%.3f \t%.3f\n', max(Q_MPBTV), max(acc_MPBTV), max(NMI_MPBTV), mean(runtime_MPBTV))
fprintf('DGFM3: \t\t%.3f \t%.3f \t%.3f \t%.3f\n', max(Q_DGFM3), max(acc_DGFM3), max(NMI_DGFM3), mean(runtime_DGFM3))
fprintf('GenLouvain: \t%.3f \t%.3f \t%.3f \t%.3f\n\n', max(Q_GenLouvain), max(acc_GenLouvain), max(NMI_GenLouvain), mean(runtime_GenLouvain))

%% function handles
function w = MVprod(L_supra,gamma,m_intra,d,v)
    L = length(gamma);
    nL = size(L_supra,1);
    n = nL/L;
    w1 = L_supra*v;
    w2 = zeros(size(w1));
    for l=1:L
        w2((l-1)*n+1:l*n) = (gamma(l)/m_intra(l)) * (d{l}'*v((l-1)*n+1:l*n)) * d{l};
    end
    w = w1+w2;
end

function w = MVprod_M_mult(A_intra,C,gamma,m_intra,d,v)
    L = length(gamma);
    nL = size(A_intra,1);
    n = nL/L;
    w1 = A_intra*v + C*v;
    w2 = zeros(size(w1));
    for l=1:L
        w2((l-1)*n+1:l*n) = (gamma(l)/(2*m_intra(l))) * (d{l}'*v((l-1)*n+1:l*n)) * d{l};
    end
    w = w1-w2;
end
