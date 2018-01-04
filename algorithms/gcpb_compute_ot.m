%%
% SUMMARY: computes OT cost from current iterate of GCPB algorithm.
% Does this by extracting the iterate in primal space (see Prop 2.1 of
% their paper), and then just computing the OT cost with (a
% feasible version) of that transportation map.
% 

function ot_cost = gcpb_compute_ot( n,r,c,C,z,eps )

%% Extracting primal OT variables from SAG dual variables
% Extract u
u = zeros(n,1);
for i=1:n
    temp_sum = 0;
    for l=1:n
        temp_sum = temp_sum + (c(l) * exp( (z(l) - C(i,l))/eps ));
    end
    u(i) = -1 * eps * log(temp_sum);
end

% Extract P using u
P = zeros(n,n);
for i=1:n
    for j=1:n
        P(i,j) = r(i)*c(j)*exp( (u(i) + z(j) - C(i,j))/eps );
    end
end

%% Computing the actual OT cost using primal variables
ot_cost = frobinnerproduct(round_transpoly(P,r,c),C);

end

