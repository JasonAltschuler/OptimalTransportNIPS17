%%
% Computes Optimal Transport using a MATLAB linear programming solver.
%

function lp_val = computeot_lp( C,r,c,n )
% vectorize P and C by: column 1, column 2, etc.
Aeq = zeros(2*n,n*n);
beq = [c';r];

% column sums correct
for row=1:n
    for t=1:n
        Aeq(row,(row-1)*n+t)=1;
    end
end

% row sums correct
for row=n+1:2*n
    for t=0:n-1
        Aeq(row,(row-n)+t*n) = 1;
    end
end

% ensure positivity of each entry
lb = zeros(n*n,1);

% solve OT LP using linprog
cost = reshape(C,n*n,1);
[lp_sol,lp_val] = linprog(cost,[],[],Aeq,beq,lb,[]);
end