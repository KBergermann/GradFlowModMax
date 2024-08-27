%
% Codes reproducing the numerical examples of MPBTV, DGFM3, and GenLouvain
% [1] from [Section 7.2, 2] on seven data sets containing feature vector
% data from different data sources as well as ground truth labels for all
% data points. We constructed small- to medium-sized multiplex networks by
% identifying each data source with a layer and each data point with a
% physical node. Analogously to [3], we construct sparse weighted
% nearest-neighbor graphs for each layer where distances are measured and
% edges are weighted by the feature vectors' Pearson correlations.
%
% References:
% [1] https://github.com/GenLouvain/GenLouvain
% [2] Kai Bergermann and Martin Stoll. Gradient flow-based modularity
%   maximization for community detection in multiplex networks, Preprint,
%   2024.
% [3] Pedro Mercado, Antoine Gautier, Francesco Tudisco, Matthias Hein.
%   The power mean Laplacian for multilayer graph clustering. In
%   International Conference on Artificial Intelligence and Statistics,
%   pages 1828–1838. PMLR, 2018.
%
% Kai Bergermann, 2024
% 
addpath('GenLouvain-master/HelperFunctions')
addpath('GenLouvain-master/')
addpath('subroutines')

%% Choose network: '3sources', 'BBC', 'BBCS', 'citeseer', 'cora', 'webkb', 'wikipedia'
network = '3sources';

% Load network and set network-specific parameters
switch network
    case '3sources'
        load networks/adj_3sources.mat
        L = size(A,1);
        gamma = 1.2*ones(L,1);
        n_c = 6;
        k_MPBTV = 10;
        k_DGFM3 = 12;
        
    case 'BBC'
        load networks/adj_BBC.mat
        L = size(A,1);
        gamma = 0.8*ones(L,1);
        n_c = 5;
        k_MPBTV = 4;
        k_DGFM3 = 11;
        
    case 'BBCS'
        load networks/adj_BBCS.mat
        L = size(A,1);
        gamma = 0.6*ones(L,1);
        n_c = 5;
        k_MPBTV = 34;
        k_DGFM3 = 11;
        
    case 'citeseer'
        load networks/adj_citeseer.mat
        L = size(A,1);
        gamma = 0.6*ones(L,1);
        n_c = 6;
        k_MPBTV = 20;
        k_DGFM3 = 27;
        
    case 'cora'
        load networks/adj_cora.mat
        L = size(A,1);
        gamma = 0.8*ones(L,1);
        n_c = 7;
        k_MPBTV = 18;
        k_DGFM3 = 28;
        
    case 'webkb'
        load networks/adj_webkb.mat
        L = size(A,1);
        gamma = 0.6*ones(L,1);
        n_c = 5;
        k_MPBTV = 3;
        k_DGFM3 = 6;
        
    case 'wikipedia'
        load networks/adj_wikipedia.mat
        L = size(A,1);
        gamma = 1*ones(L,1);
        n_c = 10;
        k_MPBTV = 13;
        k_DGFM3 = 18;
end

%% Set remaining parameters
n = size(A{1},1);
nL = n*L;
omega = 1;
dt = 0.4;
max_iter = 300;
tolit = 1.0e-08;
n_runs = 20;

%% Assemble matrices
m_intra = zeros(L,1);
A_intra = sparse(n*L,n*L);
K = sparse(n*L,n*L);
for l=1:L
    m_intra(l) = sum(sum(A{l}))/2;
    A_intra((l-1)*n+1:n*l,(l-1)*n+1:n*l) = A{l};
    d_l = sum(A{l},2);
    K((l-1)*n+1:n*l,(l-1)*n+1:n*l) = (gamma(l)/m_intra(l)) * (d_l*d_l');
end

A_inter = ones(L,L) - eye(L);

L_supra = (spdiags(sum(A_intra,2),0,n*L,n*L) - A_intra) + omega*kron((diag(sum(A_inter,2)) - A_inter),speye(n,n));

%% Offline computations
% Set up multiplex modularity matrix
[M_mult,twom] = multicat(A,gamma,omega);

% Compute eigenvalues and -vectors
tic
if strcmp(network,'citeseer') || strcmp(network,'cora')
    [phi_MPBTV,lambda_MPBTV,flag] = eigs(L_supra + K, k_MPBTV, 'smallestreal', 'SubspaceDimension', 4*k_MPBTV);
else
    [phi_MPBTV,lambda_MPBTV,flag] = eigs(L_supra + K, k_MPBTV, 'smallestreal');
end
eig_time_MPBTV = toc;

tic
[phi_DGFM3,lambda_DGFM3] = eigs(M_mult, k_DGFM3, 'largestreal');
eig_time_DGFM3 = toc;

fprintf('\nRuntimes eigenvalue computations:\nMPBTV:\t%.3f sec\tk = %d\nDGFM3:\t%.3f sec\tk = %d\n', eig_time_MPBTV, k_MPBTV, eig_time_DGFM3, k_DGFM3)

%% Per run computations
rng(0)

acc_GenLouvain = zeros(n_runs,1); Q_GenLouvain = zeros(n_runs,1); NMI_GenLouvain = zeros(n_runs,1); runtime_GenLouvain = zeros(n_runs,1);
acc_DGFM3 = zeros(n_runs,1); Q_DGFM3 = zeros(n_runs,1); NMI_DGFM3 = zeros(n_runs,1); runtime_DGFM3 = zeros(n_runs,1);
acc_MPBTV = zeros(n_runs,1); Q_MPBTV = zeros(n_runs,1); NMI_MPBTV = zeros(n_runs,1); runtime_MPBTV = zeros(n_runs,1);

bar=waitbar(0,'Looping over runs...');
for i=1:n_runs
    % GenLouvain
    tic
    [S,Q] = iterated_genlouvain(M_mult, 'verbose', false);
    runtime_GenLouvain(i) = toc;
    Q_GenLouvain(i) = getMultiplexModularity(S,A,gamma,omega);
    acc_GenLouvain(i) = getAccuracy(GT,S);
    NMI_GenLouvain(i) = getNMI(GT,S);

    % Initial conditions
    U0 = rand(nL,n_c);
    [~,U0_max_ind] = max(U0,[],2);
    U0(sub2ind(size(U0),1:size(U0,1),U0_max_ind')) = 1;
    U0(U0~=1) = 0;

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
    [U_DGFM3,~] = MBO(U0, lambda_DGFM3, phi_DGFM3, dt, max_iter, tolit);
    runtime_DGFM3(i) = toc;
    [~, S_DGFM3] = max(U_DGFM3,[],2);
    Q_DGFM3(i) = getMultiplexModularity(S_DGFM3,A,gamma,omega);
    acc_DGFM3(i) = getAccuracy(GT,S_DGFM3);
    NMI_DGFM3(i) = getNMI(GT,S_DGFM3);

    waitbar(i/n_runs,bar);
end
close(bar)

%% Print results
fprintf('\n%s network with %d runs, %d layers, and %d classes.\n', network, n_runs, L, n_c)
fprintf('\t\tmax mod\tmax acc\tmax NMI\tavg runtime\n')
fprintf('MPBTV: \t\t%.3f \t%.3f \t%.3f \t%.3f\n', max(Q_MPBTV), max(acc_MPBTV), max(NMI_MPBTV), mean(runtime_MPBTV))
fprintf('DGFM3: \t\t%.3f \t%.3f \t%.3f \t%.3f\n', max(Q_DGFM3), max(acc_DGFM3), max(NMI_DGFM3), mean(runtime_DGFM3))
fprintf('GenLouvain: \t%.3f \t%.3f \t%.3f \t%.3f\n\n', max(Q_GenLouvain), max(acc_GenLouvain), max(NMI_GenLouvain), mean(runtime_GenLouvain))
