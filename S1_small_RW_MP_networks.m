%
% Codes reproducing the numerical examples of MPBTV, DGFM3, and GenLouvain
% [1] from [Section 7.1, 2] on five small, unweighted, and publicly
% available [2] real-world multiplex networks from genetic and social
% applications.
%
% References:
% [1] https://github.com/GenLouvain/GenLouvain
% [2] Kai Bergermann and Martin Stoll. Gradient flow-based modularity
%   maximization for community detection in multiplex networks, Preprint,
%   2024.
% [3] https://manliodedomenico.com/data.php
%
% Kai Bergermann, 2024
% 

addpath('GenLouvain-master/HelperFunctions')
addpath('GenLouvain-master/')
addpath('subroutines')

%% Choose network: 'daniorerio', 'florentine', 'hepatitus', 'humanherpes', 'oryctolagus'
network = 'hepatitus';

% Load network and set network-specific parameters
switch network
    case 'daniorerio'
        load networks/danioRerio_genetic_adj.mat
        L = size(A_single,1);
        gamma = 1.2*ones(L,1);
        n_c = 16;
        k_MPBT = 30;
        k_DGFM3 = 27;
        
    case 'florentine'
        load networks/Florentine_Families_adj.mat
        L = size(A_single,1);
        gamma = 0.6*ones(L,1);
        n_c = 3;
        k_MPBT = 4;
        k_DGFM3 = 7;
        
    case 'hepatitus'
        load networks/hepatitusC_genetic_adj.mat
        L = size(A_single,1);
        gamma = 1.5*ones(L,1);
        n_c = 40;
        k_MPBT = 90;
        k_DGFM3 = 80;

    case 'humanherpes'
        load networks/humanHerpes_adj.mat
        L = size(A_single,1);
        gamma = 1*ones(L,1);
        n_c = 11;
        k_MPBT = 13;
        k_DGFM3 = 12;
        
    case 'oryctolagus'
        load networks/oryctolagus_genetic_adj.mat
        L = size(A_single,1);
        gamma = 0.4*ones(L,1);
        n_c = 13;
        k_DGFM3 = 18;
        k_MPBT = 18;
end

%% Set remaining parameters
n = size(A_single{1},2);
nL = n*L;
omega = 1;
dt = 0.4;
max_iter = 300;
tolit = 1e-08;
n_runs = 50;

%% Assemble matrices
m_intra = zeros(L,1);
A_intra = sparse(n*L,n*L);
K = sparse(n*L,n*L);
for l=1:L
    m_intra(l) = sum(sum(A_single{l}))/2;
    A_intra((l-1)*n+1:n*l,(l-1)*n+1:n*l) = A_single{l};
    d_l = sum(A_single{l},2);
    K((l-1)*n+1:n*l,(l-1)*n+1:n*l) = (gamma(l)/m_intra(l)) * (d_l*d_l');
end

A_inter = ones(L,L) - eye(L);

L_supra = (spdiags(sum(A_intra,2),0,n*L,n*L) - A_intra) + omega*kron((diag(sum(A_inter,2)) - A_inter),speye(n,n));

%% Offline computations
% Set up multiplex modularity matrix
[M_mult,twom] = multicat(A_single,gamma,omega);

% Compute eigenvalues and -vectors
if strcmp(network, 'oryctolagus')
    [phi_MPBTV,lambda_MPBTV] = eigs(L_supra + K, k_MPBT, 'smallestreal', 'SubSpaceDimension', 5*k_MPBT);
else
    [phi_MPBTV,lambda_MPBTV] = eigs(L_supra + K, k_MPBT, 'smallestreal');
end

[phi_DGFM3,lambda_DGFM3] = eigs(M_mult,k_DGFM3,'largestreal');

%% Per run computations
rng(0)

Q_GenLouvain = zeros(n_runs,1); Q_MPBTV = zeros(n_runs,1); Q_DGFM3 = zeros(n_runs,1);
for i=1:n_runs
    % GenLouvain
    [S,Q] = iterated_genlouvain(M_mult, 'verbose', false);
    Q_GenLouvain(i) = Q/twom;

    % Initial conditions
    U0 = rand(nL,n_c);
    [~,U0_max_ind] = max(U0,[],2);
    U0(sub2ind(size(U0),1:size(U0,1),U0_max_ind')) = 1;
    U0(U0~=1) = 0;
    
    % MPBTV
    [U_MPBTV,~] = MBO(U0, -lambda_MPBTV, phi_MPBTV, dt, max_iter, tolit);
    [~, S_MPBTV] = max(U_MPBTV,[],2);
    Q_MPBTV(i) = getMultiplexModularity(S_MPBTV,A_single,gamma,omega);
    
    % DGFM3
    [U_DGFM3,~] = MBO(U0, lambda_DGFM3, phi_DGFM3, dt, max_iter, tolit);
    [~, S_DGFM3] = max(U_DGFM3,[],2);
    Q_DGFM3(i) = getMultiplexModularity(S_DGFM3,A_single,gamma,omega);
end

% Print results
fprintf('\n%s network.\n', network)
fprintf('Q_MPBTV \t %.3f\n', max(Q_MPBTV))
fprintf('Q_DGFM3 \t %.3f\n', max(Q_DGFM3))
fprintf('Q_GenLouvain \t %.3f\n', max(Q_GenLouvain))
