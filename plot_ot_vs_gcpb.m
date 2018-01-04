%% 
% SUMMARY: compares rate of convergence for computing OT between
% our new Greenkhorn algorithm, and the GCBP algorithm from
% the NIPS 2016 paper "Stochastic optimization for large-scale optimal
% transport" by Genevay, Cuturi, Peyre, and Bach.
%

clear all

%% Experiment parameters
eta = 5;                          % Greenkhorn's regularization parameter
small_iters = 2500;               % number of updates in experiment
print_progress_updates = true;


%% Create input
addpath(genpath('input_generation/'));
addpath(genpath('input_generation/mnist'));
m=20; % images are of dim mxm
n=m*m;
fraction_fg = 0.2; % parameter: 20% of image area will be foreground
img_1 = synthetic_img_input(m, fraction_fg);
img_2 = synthetic_img_input(m, fraction_fg);
flattened_img_1 = reshape(img_1,n,1);
flattened_img_2 = reshape(img_2,n,1)';
[A,r,c,C] = ot_input_between_imgs(flattened_img_1,flattened_img_2,eta,m,n);

% Initialize at same point as GCPB for apples-to-apples comparison
u = zeros(n,1);
for i=1:n
    temp_sum = 0;
    for l=1:n
        temp_sum = temp_sum + (c(l) * exp( (-1* C(i,l))*eta ));
    end
    u(i) = -1 * log(temp_sum)/eta;
end

% Extract A
A = zeros(n,n);
for i=1:n
    for j=1:n
        A(i,j) = r(i)*c(j)*exp( eta*(u(i) + - C(i,j)) );
    end
end

%% Run algorithms
% Run Greenkhorn algorithm
addpath(genpath('algorithms/'));
if print_progress_updates
    disp('Beginning to run Greenkhorn.')
end
compute_otvals = true;
[P_greedy,err_greedy,greedy_ot] = greenkhorn(A,r,c,small_iters,compute_otvals,C,1);
greedy_ots = greedy_ot';

% Run GCPB algorithm
if print_progress_updates
    disp('Beginning to run GCPB.')
end
% compute stepsizes
r_max = max(r);
eps = 1/eta;
L = r_max/eps;
stepsizes = [1/(L*n); 3/(L*n); 5/(L*n)];

runs = size(stepsizes,1);
gcpb_ots = [];
for run=1:runs
    if print_progress_updates
        disp([' --> Beginning run ',num2str(run),' of ',num2str(runs)])
    end
    stepsize = stepsizes(run);
    gcpb_ot_output = gcpb_ot(r,c,small_iters,C,eps,1,stepsize);
    gcpb_ots = [gcpb_ots; gcpb_ot_output'];
end

%% Make plot
figure

% Load MIT colors
mit_red    = [163, 31, 52]/255;
mit_grey   = [138, 139, 140]/255;

% Plot GREENKHORN
linestyle='-';
plot(1:(small_iters+1), greedy_ots,'DisplayName','GREENKHORN','Color',mit_red,'LineStyle',linestyle,'LineWidth',2);
hold('all')

% Plot GBCP
for run=1:runs
    stepsize = stepsizes(run);
    if run==1
        linestyle='-.';
        stepsizestr='1';
    elseif run==2
        linestyle='--';
        stepsizestr='3';
    else
        linestyle=':';
        stepsizestr='5';
    end 
    plot(1:(small_iters+1), gcpb_ots(run,:),'DisplayName',['GCPB, stepsize=',num2str(stepsizestr),'/(Ln)'],'Color',mit_grey,'LineStyle',linestyle,'LineWidth',2);
end
hold('all')
hold('off');
legend('show');
ylabel('Value of OT');
xlabel('row/col updates');
title(strcat('GREENKHORN vs GCPB for OT, eta=',num2str(eta)));