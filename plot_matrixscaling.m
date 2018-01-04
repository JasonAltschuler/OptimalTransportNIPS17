%%
% SUMMARY: experimentally demonstrates that our new Greenkhorn algorithm
% converges faster than the classical Sinkhorn algorithm, for the matrix
% scaling problem. (See paper for details of these algorithms.)
%
% More specifically, this script generates plot like those in Figure 2 in
% our revised NIPS paper. 
%
% The matrix-scaling input is generated from the matrix-scaling step used
% in solving OT. The OT input instance is created from doing OT between
% pairs of images. (See paper for further details.)
%
%
% DESCRIPTION OF MAIN EXPERIMENT PARAMETERS/FLAGS:
%
% -- "use_synth_imgs": if true, use synthetic random images as input for OT;
% if false, use MNIST images. (See Section 5 of paper for further details.)
%
% -- "plot_compratio": if true, then plots log of competitive ratio between
% performances of Greenkhorn and Sinkhorn. This generates plots like the 2
% on the right hand side of our paper's Figure 2. If false, then plots the
% two algorithm's performances separately (with error bars). This generates
% plots like the 2 on the left hand side of Figure 2. (See caption of the
% figure for more details.)
%

clear all;

%% Experiment parameters (more details in documentation at top)
num_runs = 10;          % number of runs to "average" over. Should be >= 3.
use_synth_imgs = true;  % flag whether to use synthetic (true) or MNIST images (false) as input
plot_compratio = true;  % whether to plot distance to polytope (if false)
                        % or spread of log of competitive ratio (if true)
                        % False --> LHS plots in paper's figure 2; true --> RHS plots.
                        % See documentation at top for more details.
full_iters = 15;        % # of full Sinkhorn iterations in each experiment
print_progress_updates = true; 

%% Run the experiments
addpath(genpath('input_generation/'));
addpath(genpath('input_generation/mnist'));
addpath(genpath('algorithms/'));
err_sink_runs   = [];
err_greedy_runs = [];
for run=1:num_runs
    if print_progress_updates
        disp(['Beginning run # ',num2str(run)])
    end
    
    %% Create input
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
        if idx_2 == idx_1
            idx_2 = idx_1 + 1;
        end
        
        % flatten images and add small background so nonzero entries
        flattened_img_1 = imgs(:,idx_1) +0.01*ones(n,1);
        flattened_img_2 = imgs(:,idx_2)'+0.01*ones(1,n);
    end
    
    % The theory in our paper says roughly 6*log(n)/eps, but experimentally 
    % smaller eta suffices. See [Cuturi '13] or our paper for details.
    eta = log(n);
    [A,r,c] = ot_input_between_imgs(flattened_img_1,flattened_img_2,eta,m,n);
  
    
    %% Run algorithms
    small_iters = full_iters*n; % # of row/col updates used in full_iters # of
                                % Sink iterations, for apples-to-apples comparison
 
    [P_sink, err_sink]     = sinkhorn(A,r,c,full_iters,false,0);
    [P_greedy, err_greedy] = greenkhorn(A,r,c,small_iters,false,0);
    
    err_sink_runs   = [err_sink_runs; err_sink'];
    err_greedy_runs = [err_greedy_runs; err_greedy'];
end % big run loop


%% Make plots
figure

% downsample error vals from Greenkhorn to be same as Sinkhorn; cleans plot
downsample_indices = linspace(1,small_iters+1,full_iters+1);

% load pretty colors for plotting
mit_red  = [163, 31, 52]/255;
mit_grey = [138, 139, 140]/255;

if plot_compratio
    %% Competitive ratio spread plot (see documentation at top for details)
    series = zeros(num_runs, full_iters+1);
    downsampled_err_greedy_runs = err_greedy_runs(:,downsample_indices);
    for run=1:num_runs
        series(run,:)=log(err_sink_runs(run,:)./downsampled_err_greedy_runs(run,:));
    end
    
    max_ratio    = max(series);
    min_ratio    = min(series);
    median_ratio = median(series);
    
    h=zeros(1,3);
    
    % Shade in between
    plot_indices=downsample_indices - ones(size(downsample_indices));
    polygon_x = [plot_indices, fliplr(plot_indices)];
    polygon_y = [min_ratio, fliplr(max_ratio)];
    mit_pink = [240,199,199]/255;
    fill(polygon_x,polygon_y,mit_pink);
    hold('all')
    
    % plot max
    h(1)=plot(plot_indices,max_ratio, 'color', mit_red, 'LineWidth',2,'DisplayName','Max');
    hold('all')
    
    % plot median
    h(2)=plot(plot_indices,median_ratio, 'color', mit_red, 'LineWidth',4,'DisplayName','Median');
    hold('all')
    
    % plot min
    h(3)=plot(plot_indices,min_ratio, 'color', mit_red, 'LineWidth',2,'DisplayName','Min');
    hold('off')
    
    plot_legend = legend(h);
    set(plot_legend,'FontSize',14);
    xlabel('row/col updates')
    ylabel('ln(compet ratio)')
    title('spead of ln(compet ratio)')
    set(findobj('type','axes'),'fontsize',14)
else
    %% Error bar plot for distance to polytope (see documentation at top for details)
    % Specific plot settings
    plot_with_err_bars = true;   % flag whether to plot error bars
    plot_logs          = false;  % flag whether to make y-axis log scale
    
    % mean error
    avg_err_sink   = mean(err_sink_runs);
    avg_err_greedy = mean(err_greedy_runs);
    
    % std dev of error
    std_err_sink   = std(err_sink_runs);
    std_err_greedy = std(err_greedy_runs);
    
    % Prepare to plot
    plot_sink       = avg_err_sink;
    plot_greedy     = avg_err_greedy;
    plot_std_sink   = std_err_sink;
    plot_std_greedy = std_err_greedy;
    
    if plot_logs
        plot_sink     = log(plot_sink);
        plot_greedy   = log(plot_greedy);
        plot_std_sink = log(plot_std_sink);
        plot_std_greedy = log(plot_std_greedy);
    end
    
    if plot_with_err_bars
        % get handles of plots so that legend doesn't show 2nd greedy
        h=zeros(1,3);
        
        % plot all of greedy
        h(1)=plot(plot_greedy, 'color', mit_red, 'LineWidth',3);
        hold('all')
        
        % plot downsampled version of greedy, with error bars
        downsampled_err_greedy = plot_greedy(downsample_indices);
        downsampled_std_greedy = plot_std_greedy(downsample_indices);
        %     h(2)=errorbar(downsample_indices, downsampled_err_greedy, downsampled_std_greedy,'Marker', 'None','Markersize',20,'Color', mit_red,'DisplayName','Greedy Sinkhorn');
        h(2)=errorbar(downsample_indices, downsampled_err_greedy, downsampled_std_greedy,'.','DisplayName','GREENKHORN','LineWidth',3,'Color',mit_red);
        hold('all')
        
        % plot the few points of sinkhorn, with error bars
        %         h(3)=errorbar(downsample_indices,plot_sink,plot_std_sink,'Marker', '.', 'DisplayName', 'Vanilla Sinkhorn', 'color', mit_grey,'LineWidth',3,'LineStyle','--');
        h(3)=errorbar(downsample_indices,plot_sink,plot_std_sink,'Marker', '.', 'DisplayName', 'SINKHORN', 'color', mit_grey,'LineWidth',3);        
        hold('off')
        plot_legend = legend(h(2:3));
        set(plot_legend,'FontSize',14);
    else
        % plot without error bars
        plot(plot_greedy, 'DisplayName', 'GREENKHORN', 'Color', mit_red);
        hold('all')
        plot(downsample_indices, plot_sink, '-s', 'Markersize',10, 'DisplayName', 'SINKHORN', 'Color', mit_grey); % temp_sink.Color= 'blue';
        hold('off')
        plot_legend = legend('show');
        set(plot_legend,'FontSize',14);
    end
    
    if plot_logs
        ylabel('log(|r(P)-r|_1 + |c(P)-c|_1)');
    else
        ylabel('|r(P)-r|_1 + |c(P)-c|_1');
    end
    xlabel('row/cols updated');
    title('distance to polytope');
    set(findobj('type','axes'),'fontsize',14)
end