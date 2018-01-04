%%
% SUMMARY: This script considers the role of the regularization parameter eta.
% Our theoretical analysis requires taking eta of order (log n)/eps, but
% [Cuturi '13] observed that in practice eta can be much smaller.
% Cuturi showed that Sinkhorn outperforms state-of-the art techniques for
% computing OT distance even when eta is a small constant, and
% this script demonstrates that Greenkhorn runs faster than Sinkhorn in
% this regime with no loss in accuracy.
%
% This script generates a plot like that in the right hand side of Figure 3
% in our revised NIPS paper.
%
% More specifically, we perform OT between a pair of images (either
% synthetically generated or from MNIST). Then for various values of eta, 
% we plot the OT error for both Sinkhorn and Greenkhorn as a function
% of # of updated rows/cols.
%

%% Experiment parameters
etas = [1,5,9];            % different etas with which to run matrix-scaling-based OT
full_iters = 4;            % # of full Sinkhorn iterations in each experiment
greedy_downsampling = 50;  % downsample error vals from Greenkhorn -> faster and cleans plot.
                           % Larger value makes the plot more precise, but takes longer.
                           % (Note that implementation requires this to divide small_iters=n*full_iters)
use_synth_imgs = true;     % flag for whether to use synthetic or MNIST images as input
print_progress_updates = true; 

%% Create input
addpath(genpath('input_generation/'));
addpath(genpath('input_generation/mnist'));
if use_synth_imgs
    % SYNTHETIC IMAGE INPUT
    m=25; % imgs are of dim mxm
    n=m*m;
    fraction_fg = 0.2; % parameter: 20% of image area will be foreground
    img_1 = synthetic_img_input(m, fraction_fg);
    img_2 = synthetic_img_input(m, fraction_fg);
    flattened_img_1 = reshape(img_1,n,1);
    flattened_img_2 = reshape(img_2,n,1)';
else
    % MNIST IMAGE INPUT
    m=28; % imgs are of dim mxm
    n=m*m;
    imgs = loadMNISTImages('t10k-images-idx3-ubyte');
    labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
    
    % choose random pair of mnist images
    nimages = size(imgs,2);
    idx_1 = randi(nimages);
    idx_2 = randi(nimages);
    if idx_2 == idx_1 %% ensure distinctness
        idx_2 = idx_1 + 1;
    end
    
    % flatten images and add small background so nonzero entries
    flattened_img_1 = imgs(:,idx_1) +0.01*ones(n,1);
    flattened_img_2 = imgs(:,idx_2)'+0.01*ones(1,n);
end

%% Run for each different value of epsilon
addpath(genpath('algorithms/'));
num_runs = size(etas,2);
small_iters = full_iters*n; % # of row/col updates used in full_iters # of
                            % Sink iterations, for apples-to-apples comparison 
sinkhorn_ots = [];
greedy_ots   = [];
for run=1:num_runs
    eta = etas(run);
    if print_progress_updates
        disp(['Beginning experiment for eta=',num2str(eta)])
    end
    
    % create OT input instance from images
    [A,r,c,C] = ot_input_between_imgs(flattened_img_1,flattened_img_2,eta,m,n);
    
    % Run algorithms
    compute_otvals = true;
    [P_sink,err_sink,sink_ot] = sinkhorn(A,r,c,full_iters,compute_otvals,C);
    [P_greedy,err_greedy,greedy_ot] = greenkhorn(A,r,c,small_iters,compute_otvals,C,greedy_downsampling);
    sinkhorn_ots = [sinkhorn_ots; sink_ot'];
    greedy_ots   = [greedy_ots; greedy_ot'];
end % big run loop

%% Compute gold standard: linear program to solve OT
lp_opt = computeot_lp(C,r,c,n);

%% Make plot
figure

% Prepare to plot
downsample_indices        = linspace(1,small_iters+1,full_iters+1);
greedy_downsample_indices = linspace(1,small_iters+1,1+small_iters/greedy_downsampling);

% Load MIT colors
mit_red    = [163, 31, 52]/255;
mit_grey   = [138, 139, 140]/255;

% Plot gold standard (LP)
lp_opt_vec = lp_opt*ones(1,1+small_iters/greedy_downsampling);
plot(greedy_downsample_indices, lp_opt_vec,'DisplayName','True optimum','Color','blue','LineWidth',2); 
hold('all')

% Plot Greenkhorn performance
for run=1:num_runs
    eta=etas(run);
    
    % line styles for different etas
    if run==1
        linestyle='-';
    elseif run==2
        linestyle='--';
    else
        linestyle=':';
    end
    
    plot(greedy_downsample_indices, greedy_ots(run,greedy_downsample_indices),'DisplayName',['GREENKHORN, eta=',num2str(eta)],'Color',mit_red,'LineStyle',linestyle,'LineWidth',2);
    hold('all')
end

% Plot Sinkhorn performance
for run=1:num_runs
    eta=etas(run);
    
    % line styles for different etas
    if run==1
        linestyle='-';
    elseif run==2
        linestyle='--';
    else
        linestyle=':';
    end
    
    plot(downsample_indices, sinkhorn_ots(run,:),'DisplayName',['SINKHORN, eta=',num2str(eta)],'Color',mit_grey,'LineStyle',linestyle,'LineWidth',2);
    hold('all')
end

hold('off');
legend('show');
ylabel('Value of OT');
xlabel('row/col updates');
title('GREENKHORN vs SINKHORN for OT');