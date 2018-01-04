%%
% Given two images, creates an OT input instance (A,r,c). See paper for details.
%

function [A,r,c,C] = ot_input_between_imgs(flattened_img_1,flattened_img_2,eta,m,n)
r=flattened_img_1/sum(sum(flattened_img_1));
c=flattened_img_2/sum(sum(flattened_img_2));

C=zeros(n,n);
for i_A=1:m
    for j_A=1:m
        for i_B=1:m
            for j_B=1:m
                i = m*(j_A-1)+i_A;
                j = m*(j_B-1)+i_B;
                C(i,j)=abs(i_A-i_B)+abs(j_A-j_B);
            end
        end
    end
end
C = C/m; % renormalization because otherwise A~=0
A=exp(-1 * eta * C);
A=A/sum(sum(A));
end