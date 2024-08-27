function Q = getMultiplexModularity(C,A,gamma,omega)
% 
% This function computes the multiplex modularity [1] of a partition C of
% the node-layer pairs of the multiplex network corresponding to the
% intra-layer adjacency matrices A with resolution parameters gamma and
% all-to-all inter-layer coupling with layer-coupling parameter omega.
% 
% Input: 
%   C: community affiliation of each node-layer pair.
%   A: cell of intra-layer adjacency matrices.
%   gamma: vector of resolution parameters per layer.
%   omega: inter-layer coupling (all-to-all coupling).
% 
% Output: 
%   Q: Multiplex modularity.
% 
% Reference:
% [1] Peter J Mucha, Thomas Richardson, Kevin Macon, Mason A Porter, and
% Jukka-Pekka Onnela. Community structure in time-dependent, multiscale,
% and multiplex networks. Science, 328(5980):876–878, 2010.
% 
% Kai Bergermann, 2024
% 

    
    L = size(A,1);
    n = size(A{1},1);
    
    d = zeros(n,L);
    twom = zeros(L,1);
    for l=1:L
        d(:,l) = full(sum(A{l},2));
        twom(l) = sum(sum(A{l}));
    end
    nu = sum(twom) + L*(L-1)*n*omega;
    
    % binary class affiliation matrix
    c = length(unique(C));
    S = [];
    for k=1:c
        S = [S, C==k];
    end
    
    % intra-layer contribution
    A_sum = zeros(c,c);
    dd_sum = zeros(c,c);
    for l=1:L
        Sl = S(n*(l-1)+1:n*l,:);
        A_sum = A_sum + Sl'*A{l}*Sl;
        dlTSl = d(:,l)'*Sl;
        dd_sum = dd_sum + (gamma(l)/twom(l))*(dlTSl'*dlTSl);
    end
    
    % inter-layer coupling
    A_coupling = ones(L,L) - eye(L);
    
    Q = (1/nu) * trace(A_sum - dd_sum + omega*S'*kron(A_coupling,speye(n,n))*S);
end