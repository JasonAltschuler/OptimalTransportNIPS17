%% 
% Implementation of classical Sinkhorn algorithm for matrix scaling.
% Each iteration simply alternately updates (projects) all rows or
% all columns to have correct marginals.
% 
% Input parameters:
%  -- A:  positive square matrix to project onto U_{r,c} transport polytope (dims: nxn)
%  -- r:  desired row sums (marginals)         (dims: nx1)
%  -- c:  desired column sums (marginals)      (dims: 1xn)
%  -- T:  number of full Sinkhorn iterations (normalize ALL row or cols)
%  -- compute_otvals: flag whether to compute otvals (slow but used in some plots)
%  -- C:  cost matrix for OT
%
% Output:
%  -- P:   final scaled matrix
%  -- err: sum of row and column violations at each iteration
%  -- ot:  values of optimal transport of matrix iterates

function [P, err, otvals] = sinkhorn(A,r,c,T,compute_otvals,C)
P = A;
err = zeros(T+1,1);
r_P = sum(P,2);
c_P = sum(P,1);
err(1) = norm(r_P-r,1)+norm(c_P-c,1);

if compute_otvals
    % initialize OT
    otvals = zeros(T+1,1);
    otvals(1) = frobinnerproduct(round_transpoly(P,r,c),C);
end

for t=1:T
    if mod(t,2)==1
        % rescale rows
        r_P = sum(P,2);
        P   = bsxfun(@times,P,r./r_P);
        
        r_P = sum(P,2);
        c_P = sum(P,1);
        err(t+1) = norm(r_P-r,1)+norm(c_P-c,1);
    else
        % rescale columns
        c_P = sum(P,1);
        P   = bsxfun(@times,P,c./c_P);
        
        r_P = sum(P,2);
        c_P = sum(P,1);
        err(t+1) = norm(r_P-r,1)+norm(c_P-c,1);
    end
    if compute_otvals
        otvals(t+1) = frobinnerproduct(round_transpoly(P,r,c),C);
    end
end
end