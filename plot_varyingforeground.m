%%
% SUMMARY: experimentally demonstrates that our new Greenkhorn algorithm
% converges faster than the classical Sinkhorn algorithm, for the matrix
% scaling problem. (See paper for details of these algorithms.)
%
% More specifically, these experiments investigate the speed of convergence
% of these two algorithms, when the amount of "salient" data varies.
% This script generates a plot like that in the left hand side of
% Figure 3 in our revised NIPS paper.
%
% The matrix-scaling input is generated from the matrix-scaling step used
% in solving OT. The OT input instance is created from doing OT between
% pairs of synthetically generated images, which have varying foreground
% sizes. (See Section 5 of paper for further details.)
%

clear all

%% Experiment parameters (more details in documentation at top)
m         = 20;              % images are of dim mxm
fracs_fg  = [0.2, 0.5, 0.8]; % fraction of pixels that are foreground
num_runs  = 10;              % # of runs to "average" over, for each fg level
full_iters = 10;             % # of full Sinkhorn iterations in each experiment         
print_progress_updates = true; 

%% Run the experiments
addpath(genpath('input_generation/'));
addpath(genpath('input_generation/mnist'));
addpath(genpath('algorithms/'));
n = m*m;
total_err_sink_runs = [];
total_err_greedy_runs = [];
for frac_fg_idx=1:3
    frac_fg = fracs_fg(frac_fg_idx);
    if print_progress_updates
        disp(['Beginning experiment on fraction_fg level ',num2str(frac_fg)])
    end
    
    % For each fraction_fg level, do 'num_runs' # of runs 
    err_sink_runs   = [];
    err_greedy_runs = [];
    for run=1:num_runs
        if print_progress_updates
            disp(['  --> beginning run # ',num2str(run)])
        end
        
        %% Create input
        img_1 = synthetic_img_input(m, frac_fg);
        img_2 = synthetic_img_input(m, frac_fg);
        flattened_img_1 = reshape(img_1,n,1);
        flattened_img_2 = reshape(img_2,n,1)';
        
        % The theory in our paper says roughly 6*log(n)/eps, but experimentally
        % smaller eta suffices. See [Cuturi '13] or our paper for details.
        eta = log(n);
        [A,r,c] = ot_input_between_imgs(flattened_img_1,flattened_img_2,eta,m,n);
        
        %% Run algorithms
        small_iters = full_iters*n;  % # of row/col updates used in full_iters # of
                                     %  Sink iterations, for apples-to-apples comparison
        [P_sink, err_sink]     = sinkhorn(A,r,c,full_iters,false,0);
        [P_greedy, err_greedy] = greenkhorn(A,r,c,small_iters,false,0);
        
        err_sink_runs   = [err_sink_runs; err_sink'];
        err_greedy_runs = [err_greedy_runs; err_greedy'];
    end % big run loop
    total_err_sink_runs = [total_err_sink_runs; err_sink_runs];
    total_err_greedy_runs = [total_err_greedy_runs; err_greedy_runs];
end % big fraction_fg loop


%% Make plots
figure

% downsample error vals from Greenkhorn to be same as Sinkhorn; cleans plot
downsample_indices = linspace(1,small_iters+1,full_iters+1);

% load pretty colors for plotting
mit_red   = [163, 31, 52]/255;
mit_grey  = [138, 139, 140]/255;
mit_blue  = [51,165,214]/255;
fill_pink = [240,199,199]/255;
fill_grey = [216,215,212]/255;
fill_blue = [177,219,237]/255;
line_colors = [mit_red; mit_grey; mit_blue];

h=zeros(1,3);
for frac_fg_idx = 1:3
    line_color = line_colors(frac_fg_idx,:);
    frac_fg = fracs_fg(frac_fg_idx);
    err_sink_runs   = total_err_sink_runs((frac_fg_idx-1)*3+1:(frac_fg_idx-1)*3+num_runs,:);
    err_greedy_runs = total_err_greedy_runs((frac_fg_idx-1)*3+1:(frac_fg_idx-1)*3+num_runs,:);
    
    series = zeros(num_runs, full_iters+1);
    downsampled_err_greedy_runs = err_greedy_runs(:,downsample_indices);
    for run=1:num_runs
        series(run,:)=log(err_sink_runs(run,:)./downsampled_err_greedy_runs(run,:));
    end
    
    % plot median
    median_ratio = median(series);
    plot_indices = downsample_indices - ones(size(downsample_indices));
    h(frac_fg_idx) = plot(plot_indices,median_ratio, 'color', line_color, 'LineWidth',4,'DisplayName',['Median:',' ',num2str(frac_fg*100),'% FG']);
    hold('all')
end
hold('off')
legend(h);

xlabel('number of rows and columns updated');
set(findobj('type','axes'),'fontsize',18)

xlabel('row/col updates')
ylabel('ln(compet ratio)')
title('competitive ratio with varying foreground')