function accuracy = getAccuracy(A,B)
% 
% This function maps each class of the prediction B to the ground truth
% class with the largest overlap (without replacement) and computes the
% classification accuracy of B. We define classification accuracy as the
% proportion of correctly classified node-layer pairs in B with respect to
% the ground truth labels from A.
% 
% Input: 
%   A: ground truth.
%   B: prediction.
% 
% Output: 
%   accuracy: classification accuracy of the prediction B.
% 
% Kai Bergermann, 2024
% 
    
    if size(A) ~= size(B)
        error('Error: arrays A and B must have equal size!')
    end
    n = length(A);
    
    nA = length(unique(A));
    nB = length(unique(B));
    
    availA = (1:nA)';
    availB = (1:nB)';
    
    G = zeros(nA,nB);
    for i=1:nA
        for j=1:nB
            G(i,j) = sum((A==i) == (B==j));
        end
    end
    
    GList = reshape(G,[size(G,1)*size(G,2),1]); % column under column
    [~,I] = sort(GList,'descend');
    
    commList = zeros(nA,1);
    
    for i=1:length(I)
        if isempty(availA) || isempty(availB)
            break
        end
        
        if mod(I(i),nA)==0
            r = nA; % row index
            c = I(i)/nA; % col index
        else
            r = mod(I(i),nA); % row index
            c = ceil(I(i)/nA); % col index
        end
        
        if sum(availA==r)>0 && sum(availB==c)>0
            commList(r) = c;
            availA(availA==r) = [];
            availB(availB==c) = [];
        end
    end
    
    accuracy = 0;
    for k=1:nA
        accuracy = accuracy + (1/n)*sum((A==k) & (B==commList(k)));
    end
    % if nA<nB, then all nodes w/o predicted class assigned to them are
    % incorrectly classified since their community in commList is 0.
    
end