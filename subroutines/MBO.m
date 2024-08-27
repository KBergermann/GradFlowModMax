function [U,i] = MBO(U, lambda, phi, dt, maxIter, tol)
% 
% This function implements the graph Merriman--Bence--Osher (MBO) scheme
% [1,2] that is part of [Algorithm 1, 3] to solve the non-linear
% reaction-diffusion equations [Eq.'s (4.7) and (5.2), 3] on graphs. It
% consists of two iteratively repeated stages: a diffusion step integrating
% the linear part of the ODE and a thresholding step that maps each node’s
% community affiliation vector to the closest unit vector, i.e., the
% community it most likely belongs to.
% 
% Input:
%   U: matrix of initial conditions.
%   lambda: subset of eigenvalues of the discrete linear differential
%           operator of the ODE.
%   phi: eigenvectors corresponding to lambda.
%   dt: time step size of the diffusion step.
%   maxIter: maximum number of iterations.
%   tol: tolerance between consecutive iterates below which the scheme is
%           terminated.
% 
% Output:
%   U: steady-state solution of the ODE/community partition.
%   i: number of iterations.
% 
% References:
% [1] Barry Merriman, James K Bence, and Stanley J Osher. Motion of
%   multiple junctions: A level set approach. J. Comput. Phys.,
%   112(2):334–363, 1994.
% [2] Ekaterina Merkurjev, Tijana Kostic, and Andrea L Bertozzi. An MBO
%   scheme on graphs for classification and image processing. SIAM J.
%   Imaging Sci., 6(4):1903–1930, 2013.
% [3] Kai Bergermann and Martin Stoll. Gradient flow-based modularity
%   maximization for community detection in multiplex networks, Preprint,
%   2024.
% 
% Kai Bergermann, 2024
% 

    for i=1:maxIter
        % diffusion
        U_new = phi * (expm(dt * lambda) * (phi' * U));
        
        % thresholding
        [~,U_new_max_ind] = max(U_new,[],2);
        U_new = zeros(size(U_new));
        U_new(sub2ind(size(U_new),1:size(U_new,1),U_new_max_ind')) = 1;
        
        % stopping criterion
        rel_change = norm(U_new-U)/norm(U);
        U = U_new;
        if rel_change < tol
            break
        end
    end
end