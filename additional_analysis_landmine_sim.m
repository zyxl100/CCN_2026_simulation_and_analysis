%% landmine_grid_sim_3conditions.m
% 10x10 grid with landmines + belief warping variants
% - Control: 2D grid with landmines
% - Reject-warp: negative outcome triggers probabilistic teleport back 3 steps
% - Dim-add: portal into safe layer (no landmines)
%
% Learning:
% - TD(0) state-value learning with asymmetric feedback rates
% - Total learning mass S fixed; asymmetry A varies
%
% Outputs:
% - Saves .mat + figures into local directory

clear; clc; close all; rng(1);


%% Global config

cfg = struct();
cfg.L = 10;                 % grid size
cfg.Tmax = 1000;             % max steps per episode
cfg.nSubj = 2000;          % number of subjects per subcondition
cfg.nEpisodes = cfg.nSubj; % subjects
cfg.gamma_range = [0.5 0.99];

cfg.nMaps = 1;             % number of random landmine maps
cfg.stepCost = 0.00;        
cfg.R_landmine = -10;       % landmine reward
cfg.beta = 1;               % inverse temperature

A_list = [1, 1.5, 3, 10, 20, 50, 100];

% Reject-warp parameters
cfg.rejP_set = [0.2, 0.5, 0.8]; % sampled per-subject
cfg.backSteps = 3;

condNames = ["Control", "RejectWarp", "DimAdd"];
outDir = 'C:\Users\zyfxl\OneDrive - UC Irvine\UCI\CCNL Lab\Belief Warping Project\Paper and figures\landmine_sim_results';


%% Run simulation for multiple S values (per-subject summaries only)


S_list = [0.1, 0.7];

for iS = 1:numel(S_list)

    cfg.S = S_list(iS);
    fprintf("\n==============================\n");
    fprintf("Running S = %.1f\n", cfg.S);
    fprintf("==============================\n");

    nA    = numel(A_list);
    nCond = numel(condNames);

    Metrics = struct();
    Metrics.successRate = nan(cfg.nMaps, nCond, nA);
    Metrics.meanSteps   = nan(cfg.nMaps, nCond, nA);
    Metrics.meanReturn  = nan(cfg.nMaps, nCond, nA);
    Metrics.mineRate    = nan(cfg.nMaps, nCond, nA);

    Metrics.meanDelta    = nan(cfg.nMaps, nCond, nA);
    Metrics.meanDeltaPos = nan(cfg.nMaps, nCond, nA);
    Metrics.meanDeltaNeg = nan(cfg.nMaps, nCond, nA);

    Maps = cell(cfg.nMaps, 1);

    % Per-subject dataset: cell per (map, cond, A)
    Dataset = cell(cfg.nMaps, nCond, nA);

    for m = 1:cfg.nMaps
        [isMine2D, mineIdx] = make_landmine_map(cfg.L);
        Maps{m} = isMine2D;

        fprintf("Map %d/%d: %d landmines\n", m, cfg.nMaps, numel(mineIdx));

        for ia = 1:nA
            A = A_list(ia);

            % Asymmetric learning rates (sum = S, ratio = A)
            alpha_pos = cfg.S * (A / (1 + A));
            alpha_neg = cfg.S * (1 / (1 + A));

            for ic = 1:nCond
                cond = condNames(ic);

                [succ, steps, G, mineFrac, pe_mean_map, pe_pos_map, pe_neg_map, SubjData] = ...
                    run_many_episodes(cfg, cond, isMine2D, alpha_pos, alpha_neg);

                % Store per-subject summaries
                Dataset{m, ic, ia} = SubjData;

                % Map-level metrics
                Metrics.successRate(m, ic, ia) = mean(succ);
                Metrics.meanSteps(m, ic, ia)   = mean(steps, "omitnan");
                Metrics.meanReturn(m, ic, ia)  = mean(G, "omitnan");
                Metrics.mineRate(m, ic, ia)    = mean(mineFrac, "omitnan");

                Metrics.meanDelta(m, ic, ia)    = pe_mean_map;
                Metrics.meanDeltaPos(m, ic, ia) = pe_pos_map;
                Metrics.meanDeltaNeg(m, ic, ia) = pe_neg_map;
            end
        end
    end

    % Filename suffix: 0.1 -> 010, 0.3 -> 030, etc.
    sTag = sprintf("%03d", round(100*cfg.S));
    outFile = fullfile(outDir, ...
        sprintf("sim_landmine_3conditions_low_threat_WITH_SubjSummaries_s%s.mat", sTag));

    save(outFile, "cfg","A_list","condNames","Metrics","Maps","Dataset","-v7.3");
    fprintf("Saved: %s\n", outFile);
end


%%
% Plot subject-level distributions + mean±SEM bands (from Dataset)
% Evenly spaced x positions with labels = A_list

%clear all;

%load("sim_landmine_3conditions_low_threat_WITH_SubjSummaries_s010.mat")

%condition indices
condStr = string(condNames);
iCtrl = find(condStr=="Control",1);
iDim  = find(condStr=="DimAdd",1);
iRej  = find(condStr=="RejectWarp",1);
assert(~isempty(iCtrl)&&~isempty(iDim)&&~isempty(iRej), ...
    "condNames must include Control, DimAdd, RejectWarp");

plotOrder  = [iCtrl, iDim, iRej];
legendText = {'Control','DimAdd','RejectWarp'};

%colors
colCtrl = [0 0 0];
colDim  = [0 0.6 0];
colRej  = [0.85 0 0];
cols    = [colCtrl; colDim; colRej];

% style(text)
FS    = 18;
LW    = 3.0;
MS    = 7;
ALPHA = 0.18;     % band alpha
JIT   = 0.14;     % x-jitter half-range
PTSZ  = 18;       % scatter size
PTAL  = 0.08;     % scatter alpha

xPos = 1:numel(A_list);
xLab = string(A_list(:)');

fmtAxes = @(ax) set(ax, ...
    "FontSize", FS, ...
    "FontWeight","bold", ...
    "LineWidth", 1.4, ...
    "XTick", xPos, ...
    "XTickLabel", cellstr(xLab), ...
    "XLim", [0.5 numel(xPos)+0.5], ...
    "XMinorTick","off");

poolVals = @(fieldName, ic, ia) local_pool(Dataset, fieldName, ic, ia);

epsDen = 1e-6;  


allR = [];
for ic = 1:numel(condNames)
    for ia = 1:numel(A_list)
        v = poolVals("return", ic, ia);
        allR = [allR; v(:)];
    end
end
allR = allR(~isnan(allR));
if isempty(allR), error("No subject-level returns found in Dataset."); end

minR = min(allR);
if minR <= 0
    Rshift = 1 - minR;      % makes min(return+shift) = 1
    ylabReturn = "Cumulative Return";
else
    Rshift = 0;
    ylabReturn = "Mean return (log)";
end

% Precompute mean/SEM across subjects
nA    = numel(A_list);
nCond = numel(condNames);

R_mean  = nan(nCond,nA); R_sem  = nan(nCond,nA);
ST_mean = nan(nCond,nA); ST_sem = nan(nCond,nA);

Dnorm_mean  = nan(nCond,nA); Dnorm_sem  = nan(nCond,nA);
DPnorm_mean = nan(nCond,nA); DPnorm_sem = nan(nCond,nA);
DNnorm_mean = nan(nCond,nA); DNnorm_sem = nan(nCond,nA);

for ic = 1:nCond
    for ia = 1:nA
        % --- raw subject vectors ---
        vR = poolVals("return",  ic, ia);  % mean return per subject
        vS = poolVals("steps",   ic, ia);
        vD = poolVals("pe_mean", ic, ia);
        vP = poolVals("pe_pos",  ic, ia);
        vN = poolVals("pe_neg",  ic, ia);  % negative

        % --- panel 1: return (shift only for log plotting) ---
        vR_plot = vR(~isnan(vR)) + Rshift;
        if ~isempty(vR_plot)
            R_mean(ic,ia) = abs(mean(vR_plot));
            R_sem(ic,ia)  = std(vR_plot) / sqrt(numel(vR_plot));
        end

        % --- panel 2: steps ---
        vS = vS(~isnan(vS));
        if ~isempty(vS)
            ST_mean(ic,ia) = abs(mean(vS));
            ST_sem(ic,ia)  = std(vS) / sqrt(numel(vS));
        end

        % --- panels 3–4: normalize delta by mean return (per subject) ---
        % Use only subjects where all needed fields exist and |return| >= epsDen
        ok = ~isnan(vR) & (abs(vR) >= epsDen);

        % mean delta normalized
        okD = ok & ~isnan(vD);
        dnorm = vD(okD) ./ abs(vR(okD));
        if ~isempty(dnorm)
            Dnorm_mean(ic,ia) = abs(mean(dnorm));
            Dnorm_sem(ic,ia)  = std(dnorm) / sqrt(numel(dnorm));
        end

        % positive delta normalized
        okP = ok & ~isnan(vP);
        dpnorm = vP(okP) ./abs(vR(okP));
        if ~isempty(dpnorm)
            DPnorm_mean(ic,ia) = mean(dpnorm);
            DPnorm_sem(ic,ia)  = std(dpnorm) / sqrt(numel(dpnorm));
        end

        % |negative| delta normalized
        okN = ok & ~isnan(vN);
        dnnorm = abs(vN(okN)) ./ abs(vR(okN));
        if ~isempty(dnnorm)
            DNnorm_mean(ic,ia) = abs(mean(dnnorm));
            DNnorm_sem(ic,ia)  = std(dnnorm) / sqrt(numel(dnnorm));
        end
    end
end


% Plot

figure("Color","w","Position",[60 60 2250 800]);

tiledlayout(1,4,"Padding","loose","TileSpacing","compact");

legendHandles = gobjects(3,1);

% (1) Return (log Y) + subject distribution
ax1 = nexttile; hold on;
for k = 1:3
    ic = plotOrder(k); c = cols(k,:);

    m = R_mean(ic,:); s = R_sem(ic,:);
    lo = max(m - s, 1e-12);
    hi = max(m + s, 1e-12);
    mm = max(m,     1e-12);

    fill([xPos fliplr(xPos)], [lo fliplr(hi)], c, ...
        "FaceAlpha", ALPHA, "EdgeColor","none");

    h = plot(xPos, mm, "-o", "LineWidth", LW, "MarkerSize", MS, ...
        "Color", c, "MarkerFaceColor", c, "MarkerEdgeColor", c);

    legendHandles(k) = h;
end
set(ax1,"YScale","log");
xlabel("A","FontSize",FS,"FontWeight","bold");
ylabel(ylabReturn,"FontSize",FS,"FontWeight","bold");
fmtAxes(ax1); box off;

legend(ax1, legendHandles, legendText, "FontSize", FS, "Location","best");
legend boxoff;

% (2) Steps to goal + subject distribution
ax2 = nexttile; hold on;
for k = 1:3
    ic = plotOrder(k); c = cols(k,:);

    m = ST_mean(ic,:); s = ST_sem(ic,:);
    fill([xPos fliplr(xPos)], [(m-s) fliplr(m+s)], c, ...
        "FaceAlpha", ALPHA, "EdgeColor","none");
    plot(xPos, m, "-o", "LineWidth", LW, "MarkerSize", MS, ...
        "Color", c, "MarkerFaceColor", c, "MarkerEdgeColor", c);
end
xlabel("A","FontSize",FS,"FontWeight","bold");
ylabel("Steps to goal","FontSize",FS,"FontWeight","bold");
fmtAxes(ax2); box off;

% (3) Mean delta / mean return + subject distribution
ax3 = nexttile; hold on;
for k = 1:3
    ic = plotOrder(k); c = cols(k,:);

    m = Dnorm_mean(ic,:); s = Dnorm_sem(ic,:);
    fill([xPos fliplr(xPos)], [(m-s) fliplr(m+s)], c, ...
        "FaceAlpha", ALPHA, "EdgeColor","none");
    plot(xPos, m, "-o", "LineWidth", LW, "MarkerSize", MS, ...
        "Color", c, "MarkerFaceColor", c, "MarkerEdgeColor", c);
end
xlabel("A","FontSize",FS,"FontWeight","bold");
ylabel("\delta","FontSize",FS,"FontWeight","bold");
fmtAxes(ax3); box off;

% (4) Positive + |negative| normalized by mean return + subject distribution
ax4 = nexttile; hold on;
for k = 1:3
    ic = plotOrder(k); c = cols(k,:);


    mp = DPnorm_mean(ic,:); sp = DPnorm_sem(ic,:);
    mn = DNnorm_mean(ic,:); sn = DNnorm_sem(ic,:);

    fill([xPos fliplr(xPos)], [(mp-sp) fliplr(mp+sp)], c, ...
        "FaceAlpha", ALPHA, "EdgeColor","none");
    plot(xPos, mp, "-o", "LineWidth", LW, "MarkerSize", MS, ...
        "Color", c, "MarkerFaceColor", c, "MarkerEdgeColor", c);

    fill([xPos fliplr(xPos)], [(mn-sn) fliplr(mn+sn)], c, ...
        "FaceAlpha", ALPHA*0.85, "EdgeColor","none");
    plot(xPos, mn, "--o", "LineWidth", LW, "MarkerSize", MS, ...
        "Color", c, "MarkerFaceColor", c, "MarkerEdgeColor", c);
end
xlabel("A","FontSize",FS,"FontWeight","bold");
ylabel("|\delta|","FontSize",FS,"FontWeight","bold");
fmtAxes(ax4); box off;

text(0.02, 0.96, "Solid: +PE   Dashed: |−PE|", ...
    "Units","normalized","FontSize",FS-2,"FontWeight","bold");

annotation("textbox",[0.960 0.02 0.03 0.08], ...
    "String","B","EdgeColor","none", ...
    "FontSize",42,"FontWeight","bold", ...
    "HorizontalAlignment","right","VerticalAlignment","bottom");

%% ========= Bootstrapped reduction (Control -> Reject) pooled over ALL A (Panel 4 metrics) =========
% Uses the same definitions as your Panel 4:
%   +PE_norm   = pe_pos ./ return
%   |−PE|_norm = abs(pe_neg) ./ return
%
% Assumes already in workspace:
%   A_list, condNames, iCtrl, iRej, epsDen, poolVals(fieldName, ic, ia)

B = 1000;                 % bootstrap draws
alpha = 0.05;             % 95% CI
ciPct = [100*alpha/2, 100*(1-alpha/2)];
epsPct = 1e-12;

% -------- pool subject-level values across ALL A --------
xPos_ctrl = [];
xNeg_ctrl = [];
xPos_rej  = [];
xNeg_rej  = [];

for ia = 1:numel(A_list)
    % Control
    vR = poolVals("return", iCtrl, ia);
    vP = poolVals("pe_pos", iCtrl, ia);
    vN = poolVals("pe_neg", iCtrl, ia);

    ok = isfinite(vR) & abs(vR) >= epsDen;
    okP = ok & isfinite(vP);
    okN = ok & isfinite(vN);

    xp = vP(okP) ./ vR(okP);
    xn = abs(vN(okN)) ./ vR(okN);

    xPos_ctrl = [xPos_ctrl; xp(isfinite(xp))]; %#ok<AGROW>
    xNeg_ctrl = [xNeg_ctrl; xn(isfinite(xn))]; %#ok<AGROW>

    % Reject
    vR = poolVals("return", iRej, ia);
    vP = poolVals("pe_pos", iRej, ia);
    vN = poolVals("pe_neg", iRej, ia);

    ok = isfinite(vR) & abs(vR) >= epsDen;
    okP = ok & isfinite(vP);
    okN = ok & isfinite(vN);

    xp = vP(okP) ./ vR(okP);
    xn = abs(vN(okN)) ./ vR(okN);

    xPos_rej = [xPos_rej; xp(isfinite(xp))]; %#ok<AGROW>
    xNeg_rej = [xNeg_rej; xn(isfinite(xn))]; %#ok<AGROW>
end

if isempty(xPos_ctrl) || isempty(xPos_rej) || isempty(xNeg_ctrl) || isempty(xNeg_rej)
    error('Pooled vectors are empty. Check poolVals outputs and epsDen filtering.');
end

% -------- bootstrap condition means + differences --------
nCP = numel(xPos_ctrl); nRP = numel(xPos_rej);
nCN = numel(xNeg_ctrl); nRN = numel(xNeg_rej);

bCtrlPos = nan(B,1); bRejPos = nan(B,1);
bCtrlNeg = nan(B,1); bRejNeg = nan(B,1);

for b = 1:B
    bCtrlPos(b) = abs(mean(xPos_ctrl(randi(nCP,nCP,1)), 'omitnan'));
    bRejPos(b)  = abs(mean(xPos_rej (randi(nRP,nRP,1)), 'omitnan'));

    bCtrlNeg(b) = abs(mean(xNeg_ctrl(randi(nCN,nCN,1)), 'omitnan'));
    bRejNeg(b)  = abs(mean(xNeg_rej (randi(nRN,nRN,1)), 'omitnan'));
end

% point estimates
mCtrlPos = abs(mean(xPos_ctrl,'omitnan'));
mRejPos  = abs(mean(xPos_rej ,'omitnan'));
mCtrlNeg = abs(mean(xNeg_ctrl,'omitnan'));
mRejNeg  = abs(mean(xNeg_rej ,'omitnan'));

% CIs for condition means
ciCtrlPos = prctile(bCtrlPos, ciPct);
ciRejPos  = prctile(bRejPos,  ciPct);
ciCtrlNeg = prctile(bCtrlNeg, ciPct);
ciRejNeg  = prctile(bRejNeg,  ciPct);

% absolute reductions (Reject - Control)
dPos = bRejPos - bCtrlPos;
dNeg = bRejNeg - bCtrlNeg;

dPos_mean = abs(mRejPos) - abs(mCtrlPos);
dNeg_mean = abs(mRejNeg) - abs(mCtrlNeg);

dPos_ci = prctile(dPos, ciPct);
dNeg_ci = prctile(dNeg, ciPct);

% percent change relative to Control
pctPos = 100 * (dPos ./ max(abs(bCtrlPos), epsPct));
pctNeg = 100 * (dNeg ./ max(abs(bCtrlNeg), epsPct));

dPos_pct_mean = 100 * (dPos_mean / max(abs(mCtrlPos), epsPct));
dNeg_pct_mean = 100 * (dNeg_mean / max(abs(mCtrlNeg), epsPct));

dPos_pct_ci = prctile(pctPos, ciPct);
dNeg_pct_ci = prctile(pctNeg, ciPct);

% -------- print compact summary (APA-ish, no p-values) --------
fprintf('\n================ Panel 4 pooled over ALL A: Control vs Reject =================\n');

fprintf('(+PE_norm)  Control mean = %.4g, CI [%.4g, %.4g];  Reject mean = %.4g, CI [%.4g, %.4g]\n', ...
    mCtrlPos, ciCtrlPos(1), ciCtrlPos(2), mRejPos, ciRejPos(1), ciRejPos(2));
fprintf('           Reject − Control: Δ = %.4g, CI [%.4g, %.4g]  (%+.1f%%, CI [%+.1f%%, %+.1f%%])\n', ...
    dPos_mean, dPos_ci(1), dPos_ci(2), dPos_pct_mean, dPos_pct_ci(1), dPos_pct_ci(2));

fprintf('(|-PE|_norm) Control mean = %.4g, CI [%.4g, %.4g];  Reject mean = %.4g, CI [%.4g, %.4g]\n', ...
    mCtrlNeg, ciCtrlNeg(1), ciCtrlNeg(2), mRejNeg, ciRejNeg(1), ciRejNeg(2));
fprintf('            Reject − Control: Δ = %.4g, CI [%.4g, %.4g]  (%+.1f%%, CI [%+.1f%%, %+.1f%%])\n', ...
    dNeg_mean, dNeg_ci(1), dNeg_ci(2), dNeg_pct_mean, dNeg_pct_ci(1), dNeg_pct_ci(2));

fprintf('================================================================================\n');



%% STATS SECTION (ABSOLUTE METRICS, BOOTSTRAP)
% Requires in workspace: Dataset, A_list, condNames
% Dataset{m, ic, ia} contains subject-level fields:
%   return, steps, pe_mean, pe_pos, pe_neg   (each is [nSubj x 1])
% Outputs printed stats to Command Window and returns a Stats struct.

rng(1);

% -------------------------
% Options (edit here)
% -------------------------
opt = struct();
opt.nBoot  = 5000;
opt.CIlev  = [2.5 97.5];
opt.epsDen = 1e-6;      % exclude subjects with abs(return) < epsDen for normalized metrics
opt.splitMode = "median";  % "median" or "cutoff"
opt.A_cut = 10;            % used only if splitMode="cutoff"


% Condition indices + pair list
condStr = string(condNames);
iCtrl = find(condStr=="Control",1);
iDim  = find(condStr=="DimAdd",1);
iRej  = find(condStr=="RejectWarp",1);
assert(~isempty(iCtrl)&&~isempty(iDim)&&~isempty(iRej), ...
    "condNames must include Control, DimAdd, RejectWarp");

condPairs = {
    "Control",    "RejectWarp";
    "Control",    "DimAdd";
    "DimAdd",     "RejectWarp"
};

% -------------------------
% Metrics (absolute definitions)
% -------------------------
% All metrics are ABS at subject level.
metricList = {
    "return_abs",      "abs(mean return)";
    "steps_abs",       "abs(steps)";
    "d_abs_norm",      "abs(mean δ / abs(return))";
    "dp_abs_norm",     "abs(+δ / abs(return))";
    "dn_abs_norm",     "abs(−δ / abs(return))"
};

% pooling helper
poolVals = @(fieldName, ic, ia) local_pool(Dataset, fieldName, ic, ia);

% -------------------------
% 1) Pairwise stats at EACH A
% -------------------------
Stats = struct();
Stats.opt = opt;
Stats.A_list = A_list;
Stats.condNames = condNames;
Stats.metricList = metricList;
Stats.condPairs = condPairs;

fprintf("\n================ BOOTSTRAP PAIRWISE COMPARISONS (PER A) ================\n");
fprintf("Metrics are ABSOLUTE at subject-level; normalized metrics use denom=abs(return) with epsDen=%.1e.\n", opt.epsDen);

for ia = 1:numel(A_list)
    fprintf("\n================ A = %.3g ================\n", A_list(ia));

    for im = 1:size(metricList,1)
        mKey   = metricList{im,1};
        mLabel = metricList{im,2};
        fprintf("\n-- %s --\n", mLabel);

        % precompute values per condition for this A
        vals = struct();
        for ic = 1:numel(condNames)
            cname = string(condNames(ic));
            vals.(cname) = get_metric_abs(poolVals, ic, ia, mKey, opt.epsDen);
        end

        % compare all pairs
        for ip = 1:size(condPairs,1)
            c1 = string(condPairs{ip,1});
            c2 = string(condPairs{ip,2});
            x = vals.(c1);  % baseline
            y = vals.(c2);  % comparison

            if numel(x) < 10 || numel(y) < 10
                fprintf("  %s vs %s: insufficient data (n=%d,%d)\n", c1, c2, numel(x), numel(y));
                continue;
            end

            [dMean, CI, dEff] = boot_diff_and_d(x, y, opt.nBoot, opt.CIlev);

            fprintf("  %s − %s: Δ=%.4f  CI=[%.4f, %.4f]  d=%.3f  (n=%d,%d)\n", ...
                c2, c1, dMean, CI(1), CI(2), dEff, numel(x), numel(y));
        end
    end
end


% 2) Collapse across A
if opt.splitMode == "median"
    A_med = median(A_list);
    idxLow  = find(A_list <= A_med);
    idxHigh = find(A_list >  A_med);
    splitNote = sprintf("median split at A=%.3g", A_med);
elseif opt.splitMode == "cutoff"
    idxLow  = find(A_list <= opt.A_cut);
    idxHigh = find(A_list >  opt.A_cut);
    splitNote = sprintf("cutoff split at A=%g", opt.A_cut);
else
    error("opt.splitMode must be 'median' or 'cutoff'.");
end

fprintf("\n================ COLLAPSE ACROSS A: LOW vs HIGH (%s) ================\n", splitNote);
fprintf("Low A indices: %s | High A indices: %s\n", mat2str(idxLow), mat2str(idxHigh));

for band = 1:2
    if band == 1
        idxA = idxLow; bandName = "LOW A";
    else
        idxA = idxHigh; bandName = "HIGH A";
    end

    fprintf("\n================ %s =================\n", bandName);

    for im = 1:size(metricList,1)
        mKey   = metricList{im,1};
        mLabel = metricList{im,2};
        fprintf("\n-- %s --\n", mLabel);

        % pool across A within band
        vals = struct();
        for ic = 1:numel(condNames)
            cname = string(condNames(ic));
            vAll = [];
            for ia = idxA(:)'
                v = get_metric_abs(poolVals, ic, ia, mKey, opt.epsDen);
                vAll = [vAll; v(:)];
            end
            vals.(cname) = vAll(~isnan(vAll));
        end

        for ip = 1:size(condPairs,1)
            c1 = string(condPairs{ip,1});
            c2 = string(condPairs{ip,2});
            x = vals.(c1);
            y = vals.(c2);

            if numel(x) < 20 || numel(y) < 20
                fprintf("  %s vs %s: insufficient data (n=%d,%d)\n", c1, c2, numel(x), numel(y));
                continue;
            end

            [dMean, CI, dEff] = boot_diff_and_d(x, y, opt.nBoot, opt.CIlev);

            fprintf("  %s − %s: Δ=%.4f  CI=[%.4f, %.4f]  d=%.3f  (n=%d,%d)\n", ...
                c2, c1, dMean, CI(1), CI(2), dEff, numel(x), numel(y));
        end
    end
end

% 3) Trend over A: bootstrap slopes vs log(A)

fprintf("\n================ TREND OVER A: BOOTSTRAP SLOPES vs log(A) ================\n");
x = log(A_list(:)');

condIdxList = [iCtrl iDim iRej];
condLab = ["Control","DimAdd","RejectWarp"];

for im = 1:size(metricList,1)
    mKey   = metricList{im,1};
    mLabel = metricList{im,2};

    % Pre-pull subject vectors V{condK, ia}
    V = cell(numel(condIdxList), numel(A_list));
    for kc = 1:numel(condIdxList)
        ic = condIdxList(kc);
        for ia = 1:numel(A_list)
            V{kc, ia} = get_metric_abs(poolVals, ic, ia, mKey, opt.epsDen);
        end
    end

    % point-estimate slopes (fit on per-A means)
    slope0 = nan(1, numel(condIdxList));
    for kc = 1:numel(condIdxList)
        ybar0 = nan(1, numel(A_list));
        for ia = 1:numel(A_list)
            v = V{kc, ia};
            v = v(~isnan(v));
            if isempty(v), ybar0(ia) = NaN; else, ybar0(ia) = mean(v); end
        end
        ok = ~isnan(ybar0);
        if sum(ok) < 3
            slope0(kc) = NaN;
        else
            p = polyfit(x(ok), ybar0(ok), 1);
            slope0(kc) = p(1);
        end
    end

    % bootstrap slopes
    bootSlope = nan(opt.nBoot, numel(condIdxList));
    for b = 1:opt.nBoot
        for kc = 1:numel(condIdxList)
            ybar = nan(1, numel(A_list));
            for ia = 1:numel(A_list)
                v = V{kc, ia};
                v = v(~isnan(v));
                if isempty(v)
                    ybar(ia) = NaN;
                else
                    vb = v(randi(numel(v), numel(v), 1)); % resample subjects within A
                    ybar(ia) = mean(vb);
                end
            end
            ok = ~isnan(ybar);
            if sum(ok) < 3
                bootSlope(b,kc) = NaN;
            else
                p = polyfit(x(ok), ybar(ok), 1);
                bootSlope(b,kc) = p(1);
            end
        end
    end

    fprintf("\n-- %s --\n", mLabel);
    for kc = 1:numel(condIdxList)
        bs = bootSlope(:,kc);
        bs = bs(~isnan(bs));
        if isempty(bs)
            fprintf("  %s: slope=NaN (insufficient)\n", condLab(kc));
            continue;
        end
        CI = prctile(bs, opt.CIlev);
        fprintf("  %s: slope=%.6f  CI=[%.6f, %.6f]\n", condLab(kc), slope0(kc), CI(1), CI(2));
    end

    % slope differences (Δ slope = cond2 - cond1)
    pairs = [1 2; 1 3; 2 3]; % Ctrl-Dim, Ctrl-Rej, Dim-Rej
    pairLab = ["DimAdd-Control","RejectWarp-Control","RejectWarp-DimAdd"];
    for ip = 1:size(pairs,1)
        a = pairs(ip,1); c = pairs(ip,2);
        d0 = slope0(c) - slope0(a);
        bd = bootSlope(:,c) - bootSlope(:,a);
        bd = bd(~isnan(bd));
        if isempty(bd)
            fprintf("  Δ slope (%s): NaN (insufficient)\n", pairLab(ip));
            continue;
        end
        CI = prctile(bd, opt.CIlev);
        fprintf("  Δ slope (%s): %.6f  CI=[%.6f, %.6f]\n", pairLab(ip), d0, CI(1), CI(2));
    end
end

fprintf("\n================ DONE ================\n");



function v = get_metric_abs(poolVals, ic, ia, mKey, epsDen)
    % Returns a column vector of subject-level metric values (ABSOLUTE)
    vR = poolVals("return",  ic, ia);
    vS = poolVals("steps",   ic, ia);
    vD = poolVals("pe_mean", ic, ia);
    vP = poolVals("pe_pos",  ic, ia);
    vN = poolVals("pe_neg",  ic, ia);

    vR = vR(:); vS = vS(:); vD = vD(:); vP = vP(:); vN = vN(:);

    switch mKey
        case "return_abs"
            v = abs(vR);

        case "steps_abs"
            v = abs(vS);

        case "d_abs_norm"
            den = abs(vR);
            ok  = ~isnan(den) & den > epsDen & ~isnan(vD);
            v = nan(size(vD));
            v(ok) = abs(vD(ok) ./ den(ok));

        case "dp_abs_norm"
            den = abs(vR);
            ok  = ~isnan(den) & den > epsDen & ~isnan(vP);
            v = nan(size(vP));
            v(ok) = abs(vP(ok) ./ den(ok));

        case "dn_abs_norm"
            den = abs(vR);
            ok  = ~isnan(den) & den > epsDen & ~isnan(vN);
            v = nan(size(vN));
            v(ok) = abs(vN(ok) ./ den(ok));

        otherwise
            error("Unknown metric key: %s", mKey);
    end

    v = v(~isnan(v));
end

function [dMean, CI, dEff] = boot_diff_and_d(x, y, nBoot, CIlev)
   
    x = x(:); y = y(:);
    nx = numel(x); ny = numel(y);

    dMean = mean(y) - mean(x);

    bootDiff = nan(nBoot,1);
    for b = 1:nBoot
        xb = x(randi(nx,nx,1));
        yb = y(randi(ny,ny,1));
        bootDiff(b) = mean(yb) - mean(xb);
    end
    CI = prctile(bootDiff, CIlev);

    sp = sqrt((var(x,1) + var(y,1)) / 2);
    if sp < eps
        dEff = NaN;
    else
        dEff = dMean / sp;
    end
end

function v = local_pool(Dataset, fieldName, ic, ia)
    % Pool values across maps for a given condition and A
    nMaps = size(Dataset, 1);
    v = [];
    for m = 1:nMaps
        sd = Dataset{m, ic, ia};
        if isempty(sd) || ~isfield(sd, fieldName), continue; end
        tmp = sd.(fieldName);
        v = [v; tmp(:)];
    end
end



%% ========================================================================
% Functions

function [isMine2D, mineIdx] = make_landmine_map(L)
    % Randomly select x percentage of states as landmines, excluding start and goal
    x = 0.25; % high threat  x = 0.5
    N = L*L;
    isMine2D = false(L,L);

    start = sub2ind([L,L], L, 1);
    goal  = sub2ind([L,L], 1, L);

    candidates = setdiff(1:N, [start, goal]);
    nMines = floor(x * numel(candidates)) + 1;
    mineIdx = candidates(randperm(numel(candidates), nMines));
    isMine2D(mineIdx) = true;
end

function [succ, steps, G, mineFrac, pe_mean_map, pe_pos_map, pe_neg_map, SubjData] = ...
    run_many_episodes(cfg, cond, isMine2D, alpha_pos, alpha_neg)

    nE = cfg.nEpisodes;
    L  = cfg.L;

    % Per-subject outputs
    succ     = false(nE,1);
    steps    = nan(nE,1);
    G        = nan(nE,1);
    mineFrac = nan(nE,1);

    pe_mean_all = nan(nE,1);
    pe_pos_all  = nan(nE,1);
    pe_neg_all  = nan(nE,1);

    % Subject-specific discount factors
    gammas = cfg.gamma_range(1) + rand(nE,1) * diff(cfg.gamma_range);

    % Episode-specific settings to store (useful later)
    p_rej_all = nan(nE,1);
    portal_r  = nan(nE,1);
    portal_c  = nan(nE,1);

    % Portal candidates for DimAdd
    startRC = [L,1];
    goalRC  = [1,L];
    validPortal = true(L,L);
    validPortal(startRC(1), startRC(2)) = false;
    validPortal(goalRC(1), goalRC(2))   = false;
    portalList = find(validPortal);

    for ep = 1:nE
        % Reject prob per subject
        if cond == "RejectWarp"
            p_rej = cfg.rejP_set(randi(numel(cfg.rejP_set)));
        else
            p_rej = 0;
        end
        p_rej_all(ep) = p_rej;

        % Portal per subject (DimAdd)
        if cond == "DimAdd"
            portalIdx = portalList(randi(numel(portalList)));
            [pr, pc] = ind2sub([L,L], portalIdx);
            portalRC = [pr, pc];
            portal_r(ep) = pr; portal_c(ep) = pc;
        else
            portalRC = [-1,-1];
            portal_r(ep) = NaN; portal_c(ep) = NaN;
        end

        gamma_i = gammas(ep);

        [succ(ep), steps(ep), G(ep), mineFrac(ep), ...
            pe_mean_all(ep), pe_pos_all(ep), pe_neg_all(ep)] = ...
            run_one_episode(cfg, cond, isMine2D, alpha_pos, alpha_neg, p_rej, portalRC, gamma_i);
    end

    % Map-level means (what you already used for Metrics)
    pe_mean_map = mean(pe_mean_all, "omitnan");
    pe_pos_map  = mean(pe_pos_all,  "omitnan");
    pe_neg_map  = mean(pe_neg_all,  "omitnan");

    % Pack per-subject dataset
    SubjData = struct();
    SubjData.gamma    = gammas;
    SubjData.p_rej    = p_rej_all;
    SubjData.portal_r = portal_r;
    SubjData.portal_c = portal_c;

    SubjData.success  = succ;
    SubjData.steps    = steps;
    SubjData.return   = G;
    SubjData.mineFrac = mineFrac;

    SubjData.pe_mean = pe_mean_all;
    SubjData.pe_pos  = pe_pos_all;
    SubjData.pe_neg  = pe_neg_all;
end













function [success, T, G, mineFrac, pe_mean, pe_pos_mean, pe_neg_mean] = ...
    run_one_episode(cfg, cond, isMine2D, alpha_pos, alpha_neg, p_rej, portalRC, gamma_i)

    L = cfg.L;

    % Start
    s = [L, 1, 1];

    % PE accumulators
    pe_sum = 0; pe_count = 0;
    pe_pos_sum = 0; pe_pos_count = 0;
    pe_neg_sum = 0; pe_neg_count = 0;

    % Value function
    if cond == "DimAdd"
        V = zeros(2*L*L, 1);
    else
        V = zeros(L*L, 1);
    end

    traj = zeros(cfg.Tmax, 3);
    traj(1,:) = s;
    t = 1;

    G = 0;
    mineHits = 0;

    while t <= cfg.Tmax

        % Goal check
        if s(1) == 1 && s(2) == L
            success = true;
            T = t-1;
            mineFrac = mineHits / max(1, T);

            pe_mean     = pe_sum / max(1, pe_count);
            pe_pos_mean = pe_pos_sum / max(1, pe_pos_count);
            pe_neg_mean = pe_neg_sum / max(1, pe_neg_count);
            return;
        end

        % Portal transition
        if cond == "DimAdd" && s(3) == 1 && s(1)==portalRC(1) && s(2)==portalRC(2)
            s(3) = 2;
        end

        acts = [ -1 0;
                  1 0;
                  0 -1;
                  0  1 ];

        Q = -inf(4,1);
        sp_all = zeros(4,3);

        for a = 1:4
            sp = s;
            sp(1:2) = s(1:2) + acts(a,:);
            sp(1) = min(max(sp(1),1), L);
            sp(2) = min(max(sp(2),1), L);

            sp_all(a,:) = sp;

            r_a = reward_progress_with_landmines(cfg, cond, s, sp, isMine2D);
            Q(a) = r_a + gamma_i * V(get_idx(cond, sp, L));
        end

        a_choice = softmax_choice(Q, cfg.beta);
        sp = sp_all(a_choice,:);

        r = reward_progress_with_landmines(cfg, cond, s, sp, isMine2D);
        G = G + r;

        if is_landmine(cond, sp, isMine2D)
            mineHits = mineHits + 1;
        end

        s_idx  = get_idx(cond, s, L);
        sp_idx = get_idx(cond, sp, L);
        delta = r + gamma_i * V(sp_idx) - V(s_idx);

        pe_sum = pe_sum + delta; pe_count = pe_count + 1;
        if delta > 0
            pe_pos_sum = pe_pos_sum + delta; pe_pos_count = pe_pos_count + 1;
        elseif delta < 0
            pe_neg_sum = pe_neg_sum + delta; pe_neg_count = pe_neg_count + 1;
        end

        if delta >= 0
            V(s_idx) = V(s_idx) + alpha_pos * delta;
        else
            V(s_idx) = V(s_idx) + alpha_neg * delta;
        end

        if cond == "RejectWarp" && r < 0 && rand < p_rej
            t_back = max(1, t - cfg.backSteps);
            sp = traj(t_back,:);
        end

        t = t + 1;
        traj(t,:) = sp;
        s = sp;
    end

    % Timeout
    success = false;
    T = cfg.Tmax;
    mineFrac = mineHits / max(1, T);

    pe_mean     = pe_sum / max(1, pe_count);
    pe_pos_mean = pe_pos_sum / max(1, pe_pos_count);
    pe_neg_mean = pe_neg_sum / max(1, pe_neg_count);
end
















function r = reward_progress_with_landmines(cfg, cond, s, sp, isMine2D)
    % Landmine penalty if (row,col) is a landmine AND (not in safe layer)
    if is_landmine(cond, sp, isMine2D)
        r = cfg.R_landmine;
        return;
    end

    % Progress reward = reduction in Euclidean distance to goal (row=1,col=L)
    goalRC = [1, cfg.L];
    d_s  = norm(double(s(1:2))  - double(goalRC));
    d_sp = norm(double(sp(1:2)) - double(goalRC));
    r = (d_s - d_sp) - cfg.stepCost;
end

function tf = is_landmine(cond, s, isMine2D)
    % Landmines only exist in layer 1 (hazardous) in DimAdd; otherwise always apply
    if cond == "DimAdd"
        if s(3) == 2
            tf = false;
        else
            tf = isMine2D(s(1), s(2));
        end
    else
        tf = isMine2D(s(1), s(2));
    end
end

function idx = get_idx(cond, s, L)
    % Convert state (row,col,layer) to linear index in V
    rc_idx = sub2ind([L,L], s(1), s(2));
    if cond == "DimAdd"
        layer = s(3); % 1 or 2
        idx = rc_idx + (layer-1)*(L*L);
    else
        idx = rc_idx;
    end
end

function a = softmax_choice(Q, beta)
    z = beta * Q(:);
    z = z - max(z);
    p = exp(z) ./ sum(exp(z));
    u = rand;
    cdf = cumsum(p);
    a = find(u <= cdf, 1, "first");
end
