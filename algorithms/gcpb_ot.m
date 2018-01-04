%% 
% SUMMARY: Implements the algorithm from GCPB NIPS 2016 paper:
% https://arxiv.org/abs/1605.08527
%

function ot_vals = gcpb_ot(r,c,small_iters,C,eps,downsampling,stepsize)

% parameters in their paper (renamed)
n = size(C,1);
z = zeros(n,1);
d = zeros(n,1);
g = zeros(n,n);

% for sampling from distribution r
r_cumsum = cumsum(r);

% initialize OT
ot_vals = zeros(1 + small_iters,1);
ot_vals(1) = gcpb_compute_ot(n,r,c,C,z,eps);

% MAIN LOOP FOR THEIR ALGORITHM
for t=1:small_iters
    % SAG UPDATE
    % sample i from r
    %     ran = rand;
    %     for i = 1:n
    %         if ran < r_cumsum(i)
    %             break;
    %         end
    %     end
    i = randi(n);
    if t < n
        i = t;
    end

    % update d
    d = d - g(i,:)';
    
    % compute new gradient
    M = max(z - C(i,:)');
    temp_sum = 0;
    for l=1:n
        temp_sum = temp_sum + (c(l) * exp( (z(l) - C(i,l) - M)/eps ));
    end
    for j=1:n
        % update g(i,j)   
        temp = exp((z(j)-C(i,j)-M)/eps);
        g(i,j) = r(i)*c(j) * (1 - temp/temp_sum);
    end
    
    % update 
    d = d + g(i,:)';
    z = z + stepsize * d;
    %     z = z + (3/(L*n))*d;
    
    % Compute actual OT cost
    if mod(t,downsampling) == 0
        ot_vals(t+1) = gcpb_compute_ot(n,r,c,C,z,eps);
    end
end
end