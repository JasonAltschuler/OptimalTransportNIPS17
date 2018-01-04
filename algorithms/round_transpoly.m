%%
% Implementation of our algorithm for rounding a matrix onto the U_{r,c}
% transport polytope. See Section 2 of the paper for details.
%

function A = round_transpoly( X,r,c )

A=X;
n=size(A,1);
r_A = sum(A,2);
for i=1:n
    scaling = min(1,r(i)/r_A(i));
    A(i,:)=scaling*A(i,:);
end

c_A = sum(A,1);
for j=1:n
    scaling = min(1,c(j)/c_A(j));
    A(:,j)=scaling*A(:,j);
end

r_A = sum(A,2);
c_A = sum(A,1);
err_r = r_A - r;
err_c = c_A - c;

A = A + err_r*err_c/sum(abs(err_r));

end

