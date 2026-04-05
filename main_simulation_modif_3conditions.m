%% main_simulation_modif3conditions.m

% Outputs:
%   - Full trial-by-trial trajectories for every subject & condition
%   - Values needed for post-sim computations (delta, abs(delta), cumAbsDelta, asymAbsDelta, etc.)



clear; clc;
rng(1);

%% Experiment-level settings


cfg = struct();

cfg.L      = 10;        % states per dimension: {1,...,L}
cfg.Dmax   = 2;         % maximum # belief space dimensionality supported
cfg.Tmax   = 120;       % max steps per subject per condition

% Reward parameters
cfg.c_step  = 0.00;     % step cost (set at 0 at this moment)
cfg.R_goal  = 0.00;     % optional terminal (also set at 0)
cfg.Hmax = 120;

% Agent parameters
cfg.beta_fixed = 1.0;   % FIXED inverse temperature
A_list = [1, 1.5, 3, 10, 20, 50, 100];  % asymmetry ratios: alpha_neg = A * alpha_pos

% Subjects
nSubj = 1000;

% Subject-level gamma sampling
gamma_range = [0.50, 0.99];
gamma_subj  = gamma_range(1) + rand(nSubj,1) * (gamma_range(2) - gamma_range(1));

% conditions


% Condition 1: Benchmark (fixed low dimensionality, no rejection)
cond1 = struct();
cond1.name = "Benchmark: fixed 1D";
cond1.type = "dim_fixed";
cond1.D0   = 1;          % fixed active dims
cond1.addEvery = inf;    % never add dimensions

cond1.rejection = struct();
cond1.rejection.mode = "none";         % {"none","terminal","backward"}
cond1.rejection.p    = 0.00;           % probability of rejection event per step

% Condition 2: Dimensionality increase (adds orthogonal goal dimensions)
cond2 = struct();
cond2.name = "Dim-add";
cond2.type = "dim_add";
cond2.D0   = 1;         
cond2.addEvery = 5;     
cond2.rejection = struct();
cond2.rejection.mode = "none";
cond2.rejection.p    = 0.00;

% Condition 3: Distance/belief warping via "rejection"
%   - backward: rejection pushes you away from goal (increasing distance)
cond3 = struct();
cond3.name = "Belief/Distance: rejection backward";
cond3.type = "dim_fixed";
cond3.D0   = 1;
cond3.addEvery = inf;

cond3.rejection = struct();
cond3.rejection.mode = "backward";  % {"terminal","backward"}
cond3.rejection.p    = 0.2;        % tune as needed
cond3.rejection.backSteps = 3;      % only used for "backward"

conditions = {cond1, cond2, cond3};


% Run simulations: META loop over anhedonia parameters

% levels    
dimLevels = [2];  
rejPs     = [0.2 0.5 0.8];    % for rejection-backward condition

% Where to save (your local folder)
baseOutDir = "C:\Users\zyfxl\OneDrive - UC Irvine\UCI\CCNL Lab\Belief Warping Project";
if ~exist(baseOutDir,"dir"); mkdir(baseOutDir); end

metaRes = cell(numel(dimLevels), numel(rejPs));   % optional: keep everything in memory too

for id = 1:numel(dimLevels)
    for ip = 1:numel(rejPs)

        % minimally modify cfg/condition
        cfg_i = cfg;
        cfg_i.Dmax = dimLevels(id);   

        cond1_i = cond1;

        cond2_i = cond2;              % dim-add

        cond3_i = cond3;              % rejection backward
        cond3_i.rejection.p = rejPs(ip);

        conditions_i = {cond1_i, cond2_i, cond3_i};

        %run 
        res = run_all_conditions(cfg_i, conditions_i, A_list, nSubj, gamma_subj);

        tag = sprintf("Dmax%d_rejP%.1f", cfg_i.Dmax, cond3_i.rejection.p);
        outFile = fullfile(baseOutDir, "sim_" + tag + ".mat");

        save(outFile, "cfg_i", "conditions_i", "A_list", "gamma_subj", "res", "-v7.3");

        %master cell array
        metaRes{id, ip} = struct( ...
            "cfg", cfg_i, ...
            "conditions", {conditions_i}, ...
            "A_list", A_list, ...
            "gamma_subj", gamma_subj, ...
            "res", res, ...
            "tag", tag, ...
            "file", outFile);
    end
end

% Optional: save the whole meta collection
save(fullfile(baseOutDir, "meta_all_runs_s0.7_negative_asym_hrl.mat"), "dimLevels", "rejPs", "A_list", "gamma_subj", "metaRes", "-v7.3");



%% functions (add once)

function res = run_all_conditions(cfg, conditions, A_list, nSubj, gamma_subj)
    nCond = numel(conditions);
    nA    = numel(A_list);

    res = struct();
    res.conditions = conditions;
    res.A_list = A_list;

    % Preallocate cell arrays: {cond, A} each storing trajectories
    res.traj = cell(nCond, nA);

    for ic = 1:nCond
        for ia = 1:nA
            A = A_list(ia);
            fprintf("Running %s | A=%.3g ...\n", conditions{ic}.name, A);

            res.traj{ic, ia} = run_condition(cfg, conditions{ic}, A, nSubj, gamma_subj);
            %res.traj{ic, ia} = run_condition_hrl(cfg, conditions{ic}, A, nSubj, gamma_subj); % activate this for homeostatic RL
        end
    end
end


function out = run_condition(cfg, cond, A, nSubj, gamma_subj)

    S = 0.1;                     % specified total learning-rate mass
    % alpha_pos = S / (1 + A);
    % alpha_neg = S * A / (1 + A);

    % activate for positive asymmetry
    alpha_neg = S / (1 + A);
    alpha_pos = S * A / (1 + A);

    nStates = cfg.L ^ cfg.Dmax;

    % Preallocate trajectory logs (subject x time)
    Tmax = cfg.Tmax;
    % Size: nSubj x Tmax x Dmax
    coord_log     = zeros(nSubj, Tmax, cfg.Dmax, 'int16');
    coord_nextlog = zeros(nSubj, Tmax, cfg.Dmax, 'int16');

    activeD_log   = zeros(nSubj, Tmax, 'int8');     % active dimensionality at each step
    action_log    = zeros(nSubj, Tmax, 'int16');    % action index (1..2*activeD)
    reward_log    = nan(nSubj, Tmax);
    delta_log     = nan(nSubj, Tmax);
    V_s_log       = nan(nSubj, Tmax);
    V_sp_log      = nan(nSubj, Tmax);
    d_s_log       = nan(nSubj, Tmax);
    d_sp_log      = nan(nSubj, Tmax);
    isGoal_log    = false(nSubj, Tmax);
    isReject_log  = false(nSubj, Tmax);
    done_log      = false(nSubj, Tmax);

    % Summary metrics (per subject)
    steps_taken   = zeros(nSubj,1);
    cumAbsDelta   = nan(nSubj,1);
    asymAbsDelta  = nan(nSubj,1);  

    % Run subject
    for i = 1:nSubj
        gamma = gamma_subj(i);
        beta  = cfg.beta_fixed;
        V = zeros(nStates, 1);         % Initialize value function 
        coord = ones(1, cfg.Dmax, 'int16');  % Initial state coordinates

        done = false;
        t = 1;

        while t <= Tmax && ~done
            % Active dimensionality schedule
            activeD = get_active_dim(cond, t, cfg.Dmax);

            % Check goal under active dims: goal is coord(d)=L for all active dims
            if is_goal(coord, activeD, cfg.L)
                done = true;
                % log terminal state as "done" at this timestep (no action taken)
                coord_log(i,t,:)   = coord;
                coord_nextlog(i,t,:) = coord;
                activeD_log(i,t)   = activeD;
                action_log(i,t)    = 0;
                reward_log(i,t)    = 0;
                delta_log(i,t)     = 0;
                V_s_log(i,t)       = V(state_to_index(coord, cfg.L, cfg.Dmax));
                V_sp_log(i,t)      = V_s_log(i,t);
                d_s_log(i,t)       = dist_to_goal(coord, activeD, cfg.L);
                d_sp_log(i,t)      = d_s_log(i,t);
                isGoal_log(i,t)    = true;
                done_log(i,t)      = true;
                steps_taken(i)     = t;
                break;
            end

            % Compute action set: 2*activeD actions = (+1 or -1) along each active dim
            nAct = 2 * activeD;

            % Compute Q(a) = r + gamma * V(s') for each action
            s_idx  = state_to_index(coord, cfg.L, cfg.Dmax);
            V_s    = V(s_idx);

            Q = -inf(1, nAct);
            s_next_coords = zeros(nAct, cfg.Dmax, 'int16');

            d_s = dist_to_goal(coord, activeD, cfg.L);

            for a = 1:nAct
                coord_next = transition_deterministic(coord, a, activeD, cfg.L);

                % Apply rejection manipulation
                [coord_next2, isReject, doneReject] = apply_rejection(cond.rejection, coord_next, coord, activeD, cfg.L);

                s_next_coords(a,:) = coord_next2;

                d_sp = dist_to_goal(coord_next2, activeD, cfg.L);

                % Progress reward
                r = (d_s - d_sp) - cfg.c_step;

                if is_goal(coord_next2, activeD, cfg.L)
                    r = r + cfg.R_goal;
                end

                % If rejection is terminal, define next value as 0
                if doneReject
                    V_sp = 0;
                else
                    s_next_idx = state_to_index(coord_next2, cfg.L, cfg.Dmax);
                    V_sp = V(s_next_idx);
                end

                Q(a) = r + gamma * V_sp;
            end

            % Choose action (softmax)
            a_choice = softmax_choice(Q, beta);

            coord_next = s_next_coords(a_choice,:);
            [coord_next, isReject, doneReject] = apply_rejection(cond.rejection, coord_next, coord, activeD, cfg.L);

            d_sp = dist_to_goal(coord_next, activeD, cfg.L);
            r    = (d_s - d_sp) - cfg.c_step;
            if is_goal(coord_next, activeD, cfg.L)
                r = r + cfg.R_goal;
            end

            % TD error δ = r + gamma V(s') - V(s)
            if doneReject
                V_sp = 0;
            else
                s_next_idx = state_to_index(coord_next, cfg.L, cfg.Dmax);
                V_sp = V(s_next_idx);
            end

            delta = r + gamma * V_sp - V_s;

            % Asymmetric TD(0) update
            if delta >= 0
                alpha = alpha_pos;
            else
                alpha = alpha_neg;
            end
            V(s_idx) = V_s + alpha * delta;

            done = doneReject || is_goal(coord_next, activeD, cfg.L);

            % ---- Log ----
            coord_log(i,t,:)     = coord;
            coord_nextlog(i,t,:) = coord_next;
            activeD_log(i,t)     = activeD;
            action_log(i,t)      = a_choice;
            reward_log(i,t)      = r;
            delta_log(i,t)       = delta;
            V_s_log(i,t)         = V_s;
            V_sp_log(i,t)        = V_sp;
            d_s_log(i,t)         = d_s;
            d_sp_log(i,t)        = d_sp;
            isGoal_log(i,t)      = is_goal(coord_next, activeD, cfg.L);
            isReject_log(i,t)    = isReject;
            done_log(i,t)        = done;

            % Advance
            coord = coord_next;
            t = t + 1;
        end

        if steps_taken(i) == 0
            steps_taken(i) = min(t-1, cfg.Tmax);
        end

        % subject-level summary metrics 
        deltas = delta_log(i, 1:steps_taken(i));
        deltas = deltas(isfinite(deltas));
        if ~isempty(deltas)
            cumAbsDelta(i) = sum(abs(deltas));
            K = min(20, numel(deltas));                 % asymptotic delta (last k steps)
            asymAbsDelta(i) = mean(abs(deltas(end-K+1:end)));
        end
    end

    % Pack outputs
    out = struct();
    out.cond = cond;
    out.A    = A;

    out.coord      = coord_log;       % nSubj x Tmax x Dmax
    out.coord_next = coord_nextlog;   % nSubj x Tmax x Dmax
    out.activeD    = activeD_log;     % nSubj x Tmax
    out.action     = action_log;      % nSubj x Tmax
    out.reward     = reward_log;      % nSubj x Tmax
    out.delta      = delta_log;       % nSubj x Tmax
    out.V_s        = V_s_log;         % nSubj x Tmax
    out.V_sp       = V_sp_log;        % nSubj x Tmax
    out.d_s        = d_s_log;         % nSubj x Tmax
    out.d_sp       = d_sp_log;        % nSubj x Tmax
    out.isGoal     = isGoal_log;      % nSubj x Tmax
    out.isReject   = isReject_log;    % nSubj x Tmax
    out.done       = done_log;        % nSubj x Tmax

    out.steps_taken = steps_taken;    % nSubj x 1
    out.cumAbsDelta = cumAbsDelta;    % nSubj x 1
    out.asymAbsDelta = asymAbsDelta;  % nSubj x 1
end




% function out = run_condition_hrl(cfg, cond, A, nSubj, gamma_subj)
% 
%     S = 0.7;                     % specified total learning-rate mass
%     % activate for negative asymmetry
%     alpha_pos = S / (1 + A);
%     alpha_neg = S * A / (1 + A);
% 
%     % % activate for positive asymmetry
%     % alpha_neg = S / (1 + A);
%     % alpha_pos = S * A / (1 + A);
% 
%     nStates = cfg.L ^ cfg.Dmax;
% 
%     % Preallocate trajectory logs (subject x time)
%     Tmax = cfg.Tmax;
%     % Size: nSubj x Tmax x Dmax
%     coord_log     = zeros(nSubj, Tmax, cfg.Dmax, 'int16');
%     coord_nextlog = zeros(nSubj, Tmax, cfg.Dmax, 'int16');
% 
%     activeD_log   = zeros(nSubj, Tmax, 'int8');     % active dimensionality at each step
%     action_log    = zeros(nSubj, Tmax, 'int16');    % action index (1..2*activeD)
%     reward_log    = nan(nSubj, Tmax);
%     delta_log     = nan(nSubj, Tmax);
%     V_s_log       = nan(nSubj, Tmax);
%     V_sp_log      = nan(nSubj, Tmax);
%     d_s_log       = nan(nSubj, Tmax);
%     d_sp_log      = nan(nSubj, Tmax);
%     isGoal_log    = false(nSubj, Tmax);
%     isReject_log  = false(nSubj, Tmax);
%     done_log      = false(nSubj, Tmax);
% 
%     % Summary metrics (per subject)
%     steps_taken   = zeros(nSubj,1);
%     cumAbsDelta   = nan(nSubj,1);
%     asymAbsDelta  = nan(nSubj,1);
% 
%     % ---------------- HRRL internal state settings ----------------
%     % Only reward function changes: we add an internal scalar H(t) per subject.
%     % Step cost c = +1; K(t)=c if not goal, -H(t) if goal -> restore to set point H*=0.
%     Hmax = cfg.Hmax;        % REQUIRED: add to cfg (e.g., cfg.Hmax = 1e6;)
%     c_int = 1;              % specified internal step cost (+1)
%     % --------------------------------------------------------------
% 
%     % Run subject
%     for i = 1:nSubj
%         gamma = gamma_subj(i);
%         beta  = cfg.beta_fixed;
%         V = zeros(nStates, 1);         % Initialize value function
%         coord = ones(1, cfg.Dmax, 'int16');  % Initial state coordinates
% 
%         % HRRL internal state (scalar) for this subject
%         H = 0;
% 
%         done = false;
%         t = 1;
% 
%         while t <= Tmax && ~done
%             % Active dimensionality schedule
%             activeD = get_active_dim(cond, t, cfg.Dmax);
% 
%             % Check goal under active dims: goal is coord(d)=L for all active dims
%             if is_goal(coord, activeD, cfg.L)
%                 done = true;
% 
%                 % ---- HRRL reward at terminal step (no action taken) ----
%                 % K = -H (restore to set point), H_next = 0 (bounded)
%                 dH  = H^2;          % d(H)=H^2 since H*=0
%                 Hn  = 0;
%                 dHn = Hn^2;
%                 r   = dH - dHn;     % reward = reduction in drive
%                 % --------------------------------------------------------
% 
%                 % log terminal state as "done" at this timestep (no action taken)
%                 coord_log(i,t,:)     = coord;
%                 coord_nextlog(i,t,:) = coord;
%                 activeD_log(i,t)     = activeD;
%                 action_log(i,t)      = 0;
%                 reward_log(i,t)      = r;
%                 delta_log(i,t)       = 0;
%                 V_s_log(i,t)         = V(state_to_index(coord, cfg.L, cfg.Dmax));
%                 V_sp_log(i,t)        = V_s_log(i,t);
%                 d_s_log(i,t)         = dist_to_goal(coord, activeD, cfg.L);
%                 d_sp_log(i,t)        = d_s_log(i,t);
%                 isGoal_log(i,t)      = true;
%                 done_log(i,t)        = true;
%                 steps_taken(i)       = t;
%                 break;
%             end
% 
%             % Compute action set: 2*activeD actions = (+1 or -1) along each active dim
%             nAct = 2 * activeD;
% 
%             % Compute Q(a) = r + gamma * V(s') for each action
%             s_idx  = state_to_index(coord, cfg.L, cfg.Dmax);
%             V_s    = V(s_idx);
% 
%             Q = -inf(1, nAct);
%             s_next_coords = zeros(nAct, cfg.Dmax, 'int16');
% 
%             d_s = dist_to_goal(coord, activeD, cfg.L);
% 
%             for a = 1:nAct
%                 coord_next = transition_deterministic(coord, a, activeD, cfg.L);
% 
%                 % Apply rejection manipulation
%                 [coord_next2, isReject, doneReject] = apply_rejection(cond.rejection, coord_next, coord, activeD, cfg.L);
% 
%                 s_next_coords(a,:) = coord_next2;
% 
%                 d_sp = dist_to_goal(coord_next2, activeD, cfg.L);
% 
%                 % ---------------- HRRL reward (ONLY CHANGE) ----------------
%                 % Define internal dynamics from your spec:
%                 % K(t) = +1 if not goal state, -H(t) if goal state
%                 % H(t+1) = H(t) + K(t), bounded [0, Hmax], set point H*=0
%                 % d(H)=H^2; r(t)=d(H(t)) - d(H(t+1))
%                 isGoalNext = is_goal(coord_next2, activeD, cfg.L);
% 
%                 if isGoalNext
%                     K = -H;           % restore fully to homeostasis
%                 else
%                     K = c_int;        % unit step cost (+1)
%                 end
% 
%                 H_next = H + K;
%                 % bound H_next to [0, Hmax]
%                 if H_next < 0
%                     H_next = 0;
%                 elseif H_next > Hmax
%                     H_next = Hmax;
%                 end
% 
%                 dH  = H^2;
%                 dHn = H_next^2;
% 
%                 r = - H^2;        % reward = drive reduction relative to set point
%                 % -----------------------------------------------------------
% 
%                 % If rejection is terminal, define next value as 0
%                 if doneReject
%                     V_sp = 0;
%                 else
%                     s_next_idx = state_to_index(coord_next2, cfg.L, cfg.Dmax);
%                     V_sp = V(s_next_idx);
%                 end
% 
%                 Q(a) = r + gamma * V_sp;
%             end
% 
%             % Choose action (softmax)
%             a_choice = softmax_choice(Q, beta);
% 
%             coord_next = s_next_coords(a_choice,:);
%             [coord_next, isReject, doneReject] = apply_rejection(cond.rejection, coord_next, coord, activeD, cfg.L);
% 
%             d_sp = dist_to_goal(coord_next, activeD, cfg.L);
% 
%             % ---------------- HRRL reward (ONLY CHANGE) ----------------
%             isGoalNext = is_goal(coord_next, activeD, cfg.L);
% 
%             if isGoalNext
%                 K = -H;              % restore fully to homeostasis
%             else
%                 K = c_int;           % unit step cost (+1)
%             end
% 
%             H_next = H + K;
%             % bound H_next to [0, Hmax]
%             if H_next < 0
%                 H_next = 0;
%             elseif H_next > Hmax
%                 H_next = Hmax;
%             end
% 
%             dH  = H^2;
%             dHn = H_next^2;
% 
%             r = dH - dHn;
%             % -----------------------------------------------------------
% 
%             % TD error δ = r + gamma V(s') - V(s)
%             if doneReject
%                 V_sp = 0;
%             else
%                 s_next_idx = state_to_index(coord_next, cfg.L, cfg.Dmax);
%                 V_sp = V(s_next_idx);
%             end
% 
%             delta = r + gamma * V_sp - V_s;
% 
%             % Asymmetric TD(0) update
%             if delta >= 0
%                 alpha = alpha_pos;
%             else
%                 alpha = alpha_neg;
%             end
%             V(s_idx) = V_s + alpha * delta;
% 
%             done = doneReject || is_goal(coord_next, activeD, cfg.L);
% 
%             % ---- Log ----
%             coord_log(i,t,:)     = coord;
%             coord_nextlog(i,t,:) = coord_next;
%             activeD_log(i,t)     = activeD;
%             action_log(i,t)      = a_choice;
%             reward_log(i,t)      = r;
%             delta_log(i,t)       = delta;
%             V_s_log(i,t)         = V_s;
%             V_sp_log(i,t)        = V_sp;
%             d_s_log(i,t)         = d_s;
%             d_sp_log(i,t)        = d_sp;
%             isGoal_log(i,t)      = is_goal(coord_next, activeD, cfg.L);
%             isReject_log(i,t)    = isReject;
%             done_log(i,t)        = done;
% 
%             % Advance
%             coord = coord_next;
%             H     = H_next;          % advance internal state (part of reward definition only)
%             t = t + 1;
%         end
% 
%         if steps_taken(i) == 0
%             steps_taken(i) = min(t-1, cfg.Tmax);
%         end
% 
%         % subject-level summary metrics
%         deltas = delta_log(i, 1:steps_taken(i));
%         deltas = deltas(isfinite(deltas));
%         if ~isempty(deltas)
%             cumAbsDelta(i) = sum(abs(deltas));
%             K = min(20, numel(deltas));                 % asymptotic delta (last k steps)
%             asymAbsDelta(i) = mean(abs(deltas(end-K+1:end)));
%         end
%     end
% 
%     % Pack outputs
%     out = struct();
%     out.cond = cond;
%     out.A    = A;
% 
%     out.coord      = coord_log;       % nSubj x Tmax x Dmax
%     out.coord_next = coord_nextlog;   % nSubj x Tmax x Dmax
%     out.activeD    = activeD_log;     % nSubj x Tmax
%     out.action     = action_log;      % nSubj x Tmax
%     out.reward     = reward_log;      % nSubj x Tmax
%     out.delta      = delta_log;       % nSubj x Tmax
%     out.V_s        = V_s_log;         % nSubj x Tmax
%     out.V_sp       = V_sp_log;        % nSubj x Tmax
%     out.d_s        = d_s_log;         % nSubj x Tmax
%     out.d_sp       = d_sp_log;        % nSubj x Tmax
%     out.isGoal     = isGoal_log;      % nSubj x Tmax
%     out.isReject   = isReject_log;    % nSubj x Tmax
%     out.done       = done_log;        % nSubj x Tmax
% 
%     out.steps_taken  = steps_taken;    % nSubj x 1
%     out.cumAbsDelta  = cumAbsDelta;    % nSubj x 1
%     out.asymAbsDelta = asymAbsDelta;   % nSubj x 1
% end






function activeD = get_active_dim(cond, t, Dmax)
    % Determine active dimensionality at timestep t (1-indexed)
    if cond.type == "dim_fixed"
        activeD = cond.D0;
        return;
    end
    if cond.type == "dim_add"
        inc = floor((t-1) / cond.addEvery);
        activeD = min(Dmax, cond.D0 + inc);
        return;
    end
    error("Unknown condition type: %s", cond.type);
end


function tf = is_goal(coord, activeD, L)
    tf = all(coord(1:activeD) == L);
end


function d = dist_to_goal(coord, activeD, L)
    d = sum(abs(int16(L) - coord(1:activeD)));
end


function coord_next = transition_deterministic(coord, a, activeD, L)
    
    coord_next = coord;

    k = ceil(double(a)/2);
    if k < 1 || k > activeD
        error("Invalid action index", a, activeD);
    end

    if mod(a,2)==1
        % +1
        coord_next(k) = min(int16(L), coord_next(k) + 1);
    else
        % -1
        coord_next(k) = max(int16(1), coord_next(k) - 1);
    end
end


function [coord_out, isReject, doneReject] = apply_rejection(rej, coord_next, coord_prev, activeD, L)
    
    % rejection.mode:
    %   "none"     => no rejection
    %   "terminal" => with prob 1 - p, terminate episode immediately
    %   "backward" => with prob p, push the agent away from goal by 3 steps in a random active dim
    %
    % Returns:
    %   coord_out   : resulting coord for the next state
    %   isReject    : whether rejection occurrs
    %   doneReject  : whether episode terminates due to rejection

    if ~isfield(rej, "mode") || rej.mode == "none" || rej.p <= 0
        coord_out = coord_next;
        isReject = false;
        doneReject = false;
        return;
    end

    isReject = (rand < rej.p);

    if ~isReject
        coord_out = coord_next;
        doneReject = false;
        return;
    end

    switch rej.mode
        case "terminal"
            coord_out = coord_next;  % state doesn't matter if terminal
            doneReject = true;

        case "backward"
            if ~isfield(rej, "backSteps")
                backSteps = 1;
            else
                backSteps = rej.backSteps;
            end

            coord_out = coord_next;

            % Choose a random active dimension and move away from goal (toward 1)
            k = randi(activeD);
            coord_out(k) = max(int16(1), coord_out(k) - int16(backSteps));

            doneReject = false;

        otherwise
            error("Unknown rejection mode: %s", rej.mode);
    end

end


function idx = state_to_index(coord, L, Dmax)
    % Convert Dmax-dimensional coordinate in {1..L} to linear index in [1..L^Dmax]
    %   idx = 1 + sum_{k=1..Dmax} (coord(k)-1) * L^(k-1)
    idx = 1;
    mult = 1;
    for k = 1:Dmax
        idx = idx + double(coord(k)-1) * mult;
        mult = mult * L;
    end
end


function a = softmax_choice(Q, beta)
    if all(~isfinite(Q))
        a = randi(numel(Q));
        return;
    end

    Qmax = max(Q(isfinite(Q)));
    z = beta * (Q - Qmax);
    z(~isfinite(z)) = -inf;

    w = exp(z);
    w(~isfinite(w)) = 0;

    s = sum(w);
    if s <= 0
        a = randi(numel(Q));
        return;
    end

    p = w / s;
    a = sample_discrete(p);
end


function k = sample_discrete(p)
    % Sample integer k from categorical distribution p (1..K)
    c = cumsum(p);
    r = rand;
    k = find(r <= c, 1, 'first');
    if isempty(k)
        k = numel(p);
    end
end


