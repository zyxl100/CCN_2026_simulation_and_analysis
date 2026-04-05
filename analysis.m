%% Preprocessing + Loading Dataset (OPTIONAL positive-asym; keeps SAME SRes structure)

clear; clc;

% S SETTINGS
S_list = [0.1 0.3 0.5 0.7];
S_use  = S_list;
condNames = {'Control','Dim-Add', 'Rejection'};
matDir = 'C:\Users\zyfxl\OneDrive - UC Irvine\UCI\CCNL Lab\Belief Warping Project\CCN 2026\Paper and figures';
sTag = @(s) sprintf('%.1f', s);

% ---- toggle: include positive-asym files or not ----
INCLUDE_POS_ASYM = false;   % <- set false to skip loading *_positive_asym

SRes = struct([]);

for iS = 1:numel(S_use)
    S = S_use(iS);

    % --------- load negative-asym (default) ---------
    matName_neg = fullfile(matDir, sprintf('meta_all_runs_s%s.mat', sTag(S)));
    tmpN = load(matName_neg);

    A_list_neg    = tmpN.A_list(:);
    dimLevels_neg = tmpN.dimLevels;
    rejPs_neg     = tmpN.rejPs;
    metaRes_neg   = tmpN.metaRes;

    % --------- preprocess NEG ---------
    outNeg = preprocess_one_meta(metaRes_neg, A_list_neg);

    if INCLUDE_POS_ASYM
        % --------- load positive-asym ---------
        matName_pos = fullfile(matDir, sprintf('meta_all_runs_s%s_positive_asym.mat', sTag(S)));
        tmpP = load(matName_pos);

        A_list_pos    = tmpP.A_list(:);
        dimLevels_pos = tmpP.dimLevels;
        rejPs_pos     = tmpP.rejPs;
        metaRes_pos   = tmpP.metaRes;

        % --------- preprocess POS ---------
        outPos = preprocess_one_meta(metaRes_pos, A_list_pos);

        % --------- relabel positive-asym A's as negative, drop A=-1 ---------
        A_pos_signed = -A_list_pos;
        keepPos      = (A_list_pos ~= 1);
        A_pos_signed = A_pos_signed(keepPos);
        outPos = slice_out_by_keep(outPos, keepPos);

        % --------- merge A dimension (stack NEG then POS) ---------
        A_all  = [A_list_neg; A_pos_signed];
        outAll = concat_out(outNeg, outPos);

    else
        % --------- keep SAME structure, but only NEG data ---------
        A_all  = A_list_neg;
        outAll = outNeg;

        % still define these so SRes fields exist (downstream-safe)
        dimLevels_pos = [];
        rejPs_pos     = [];
    end

    % --------- store (SRes structure unchanged) ---------
    SRes(iS).S          = S;
    SRes(iS).A_list     = A_all;
    SRes(iS).condNames  = outAll.condNames;

    SRes(iS).muR        = outAll.muR;
    SRes(iS).muD        = outAll.muD;
    SRes(iS).muPosR     = outAll.muPosR;
    SRes(iS).muNegR     = outAll.muNegR;
    SRes(iS).muPosD     = outAll.muPosD;
    SRes(iS).muNegD     = outAll.muNegD;
    SRes(iS).asyR       = outAll.asyR;
    SRes(iS).asyD       = outAll.asyD;
    SRes(iS).tailR5     = outAll.tailR5;
    SRes(iS).tailD5     = outAll.tailD5;
    SRes(iS).tailR5n    = outAll.tailR5n;
    SRes(iS).tailD5n    = outAll.tailD5n;
    SRes(iS).metaCells  = outAll.metaCells;

    % Optional: keep meta grids
    SRes(iS).dimLevels_neg = dimLevels_neg;
    SRes(iS).rejPs_neg     = rejPs_neg;
    SRes(iS).dimLevels_pos = dimLevels_pos;
    SRes(iS).rejPs_pos     = rejPs_pos;
end



%% -------- local helpers for preprocessing data--------

function out = preprocess_one_meta(metaRes, A_list)
    % condition indices in traj{ic,ia}
    ic_ctrl = 1; ic_dim = 2; ic_rej = 3;
    condNames = {'Control','Dim-add','Reject'};
    condIdx   = [ic_ctrl, ic_dim, ic_rej];
    nCond = numel(condIdx);

    A  = A_list(:);
    nA = numel(A);

    % meta grid
    nDim = size(metaRes,1);
    nP   = size(metaRes,2);

    % pooling scheme
    ip_fixed = 1;
    id_fixed = 1;

    metaCells = cell(nCond,1);
    metaCells{1} = [ (1:nDim)' , repmat(ip_fixed,nDim,1) ];
    metaCells{2} = [ (1:nDim)' , repmat(ip_fixed,nDim,1) ];
    metaCells{3} = [ repmat(id_fixed,nP,1), (1:nP)' ];

    % containers
    muR      = cell(nA,nCond);
    muD      = cell(nA,nCond);
    muPosR   = cell(nA,nCond);
    muNegR   = cell(nA,nCond);
    muPosD   = cell(nA,nCond);
    muNegD   = cell(nA,nCond);
    asyR     = cell(nA,nCond);
    asyD     = cell(nA,nCond);
    tailR5   = cell(nA,nCond);
    tailD5   = cell(nA,nCond);
    tailR5n  = cell(nA,nCond);
    tailD5n  = cell(nA,nCond);

    for ia = 1:nA
        for ic = 1:nCond

            pooled_muR    = [];
            pooled_muD    = [];
            pooled_muPosR = [];
            pooled_muNegR = [];
            pooled_muPosD = [];
            pooled_muNegD = [];
            pooled_asyR   = [];
            pooled_asyD   = [];
            pooled_tailR5 = [];
            pooled_tailD5 = [];
            pooled_tailR5n= [];
            pooled_tailD5n= [];

            pairs = metaCells{ic};   % [id, ip] rows
            for kk = 1:size(pairs,1)
                id = pairs(kk,1);
                ip = pairs(kk,2);

                traj = metaRes{id, ip}.res.traj{condIdx(ic), ia};

                D = traj.delta;
                R = traj.reward;
                steps = traj.steps_taken(:);

                [nSubj, Tmax] = size(D);

                % mask by steps
                for s = 1:nSubj
                    Ti = steps(s);
                    if Ti < Tmax
                        D(s, Ti+1:end) = NaN;
                        R(s, Ti+1:end) = NaN;
                    end
                end

                muR_i = nansum(R,2) ./ steps;
                muD_i = nansum(D,2) ./ steps;

                % reward conditional means
                Rpos = R; Rpos(Rpos<=0) = NaN;
                Rneg = R; Rneg(Rneg>=0) = NaN;
                nPos = sum(isfinite(Rpos),2);
                nNeg = sum(isfinite(Rneg),2);
                muPosR_i = nansum(Rpos,2)./nPos;  muPosR_i(nPos==0)=NaN;
                muNegR_i = nansum(Rneg,2)./nNeg;  muNegR_i(nNeg==0)=NaN;

                % delta conditional means
                Dpos = D; Dpos(Dpos<=0) = NaN;
                Dneg = D; Dneg(Dneg>=0) = NaN;
                nPosD = sum(isfinite(Dpos),2);
                nNegD = sum(isfinite(Dneg),2);
                muPosD_i = nansum(Dpos,2)./nPosD;  muPosD_i(nPosD==0)=NaN;
                muNegD_i = nansum(Dneg,2)./nNegD;  muNegD_i(nNegD==0)=NaN;

                % asymmetry metrics (your existing definition)
                asyR_i = (muPosR_i - abs(muNegR_i)) ./ muR_i;
                asyD_i = (muPosD_i - abs(muNegD_i)) ./ muD_i;

                % tail risk (5th percentile across subjects)
                vR = muR_i(isfinite(muR_i));
                vD = muD_i(isfinite(muD_i));

                if ~isempty(vR)
                    tR5  = prctile(vR,5);
                    mR   = mean(vR,'omitnan');
                    tR5n = tR5 / mR;
                else
                    tR5 = NaN; tR5n = NaN;
                end

                if ~isempty(vD)
                    tD5  = prctile(vD,5);
                    mR   = mean(vR,'omitnan');      % normalize by mean reward
                    tD5n = tD5 / mR;
                else
                    tD5 = NaN; tD5n = NaN;
                end

                % append pooled
                pooled_muR     = [pooled_muR;     muR_i(isfinite(muR_i))];
                pooled_muD     = [pooled_muD;     muD_i(isfinite(muD_i))];
                pooled_muPosR  = [pooled_muPosR;  muPosR_i(isfinite(muPosR_i))];
                pooled_muNegR  = [pooled_muNegR;  muNegR_i(isfinite(muNegR_i))];
                pooled_muPosD  = [pooled_muPosD;  muPosD_i(isfinite(muPosD_i))];
                pooled_muNegD  = [pooled_muNegD;  muNegD_i(isfinite(muNegD_i))];
                pooled_asyR    = [pooled_asyR;    asyR_i(isfinite(asyR_i))];
                pooled_asyD    = [pooled_asyD;    asyD_i(isfinite(asyD_i))];

                pooled_tailR5  = [pooled_tailR5;  tR5];
                pooled_tailD5  = [pooled_tailD5;  tD5];
                pooled_tailR5n = [pooled_tailR5n; tR5n];
                pooled_tailD5n = [pooled_tailD5n; tD5n];
            end

            muR{ia,ic}     = pooled_muR;
            muD{ia,ic}     = pooled_muD;
            muPosR{ia,ic}  = pooled_muPosR;
            muNegR{ia,ic}  = pooled_muNegR;
            muPosD{ia,ic}  = pooled_muPosD;
            muNegD{ia,ic}  = pooled_muNegD;
            asyR{ia,ic}    = pooled_asyR;
            asyD{ia,ic}    = pooled_asyD;

            tailR5{ia,ic}  = pooled_tailR5;
            tailD5{ia,ic}  = pooled_tailD5;
            tailR5n{ia,ic} = pooled_tailR5n;
            tailD5n{ia,ic} = pooled_tailD5n;
        end
    end

    out = struct();
    out.condNames = condNames;

    out.muR     = muR;
    out.muD     = muD;
    out.muPosR  = muPosR;
    out.muNegR  = muNegR;
    out.muPosD  = muPosD;
    out.muNegD  = muNegD;
    out.asyR    = asyR;
    out.asyD    = asyD;
    out.tailR5  = tailR5;
    out.tailD5  = tailD5;
    out.tailR5n = tailR5n;
    out.tailD5n = tailD5n;
    out.metaCells = metaCells;
end


function out2 = slice_out_by_keep(out, keepA)
    % keepA is logical over A rows
    out2 = out;
    fns = fieldnames(out);
    for i = 1:numel(fns)
        fn = fns{i};
        if iscell(out.(fn)) && size(out.(fn),1) == numel(keepA)
            out2.(fn) = out.(fn)(keepA, :);
        end
    end
end

function outAll = concat_out(outA, outB)
%     % Concatenate along A dimension for all cell fields that are nA x nCond
%     outAll = outA;
%     fns = fieldnames(outA);
%     for i = 1:numel(fns)
%         fn = fns{i};
%         if iscell(outA.(fn)) && iscell(outB.(fn)) ...
%                 && size(outA.(fn),2) == size(outB.(fn),2)
%             % assume A dimension is rows
%             outAll.(fn) = [outA.(fn); outB.(fn)];
%         end
%     end
%     outAll.condNames = outA.condNames; % same
% end
% 
% %% Supplement 1 effect of Gamma (pooled A,S) =================
% % Update: last 2 subplots are RAW (muPos_pool and muNeg_pool), NOT abs and NOT z-scored.
% 
% 
% % ---- settings ----
% S_list = [0.1 0.3 0.5 0.7];
% matDir = pwd;   % or set explicitly
% NBINS  = 10;
% 
% condNames = {'Control','Dim-add','Reject'};
% condIdx   = [1 2 3];
% nCond     = numel(condNames);
% 
% ip_fixed = 1;   % fixed rejP for Control/Dim-add
% id_fixed = 1;   % fixed Dmax for Reject
% 
% % ---- load first file to get A_list, nSubj, gamma_subj ----
% tmp0 = load(fullfile(matDir, sprintf('meta_all_runs_s%.1f.mat', S_list(1))));
% A_list0 = tmp0.A_list(:);
% traj0   = tmp0.metaRes{1,1}.res.traj{condIdx(1), 1};
% nSubj   = numel(traj0.steps_taken);
% 
% if isfield(tmp0,'gamma_subj')
%     gamma_subj = tmp0.gamma_subj(:);
% else
%     error('gamma_subj not found in meta_all_runs_s*.mat. Save it during simulation.');
% end
% 
% % ---- subject-level accumulators (sum + count across A,S) ----
% sum_muD     = zeros(nSubj, nCond); cnt_muD     = zeros(nSubj, nCond);
% sum_asy     = zeros(nSubj, nCond); cnt_asy     = zeros(nSubj, nCond);
% sum_muPosD  = zeros(nSubj, nCond); cnt_muPosD  = zeros(nSubj, nCond);
% 
% % UPDATED: store RAW negative conditional mean (negative-valued)
% sum_muNegD  = zeros(nSubj, nCond); cnt_muNegD  = zeros(nSubj, nCond);
% 
% for iS = 1:numel(S_list)
%     S = S_list(iS);
%     tmp = load(fullfile(matDir, sprintf('meta_all_runs_s%.1f.mat', S)));
% 
%     A_list  = tmp.A_list(:);
%     metaRes = tmp.metaRes;
% 
%     nDim = size(metaRes,1);
%     nP   = size(metaRes,2);
% 
%     metaCells = cell(nCond,1);
%     metaCells{1} = [ (1:nDim)' , repmat(ip_fixed,nDim,1) ];   % Control
%     metaCells{2} = [ (1:nDim)' , repmat(ip_fixed,nDim,1) ];   % Dim-add
%     metaCells{3} = [ repmat(id_fixed,nP,1), (1:nP)' ];        % Reject
% 
%     for ic = 1:nCond
%         pairs = metaCells{ic};
% 
%         for ia = 1:numel(A_list)
%             muD_pair    = nan(nSubj, size(pairs,1));
%             asy_pair    = nan(nSubj, size(pairs,1));
%             muPos_pair  = nan(nSubj, size(pairs,1));
%             muNeg_pair  = nan(nSubj, size(pairs,1)); % UPDATED: raw negative mean
% 
%             for kk = 1:size(pairs,1)
%                 id = pairs(kk,1);
%                 ip = pairs(kk,2);
% 
%                 traj  = metaRes{id, ip}.res.traj{condIdx(ic), ia};
%                 D     = traj.delta;
%                 steps = traj.steps_taken(:);
% 
%                 % mask beyond steps
%                 [~,Tmax] = size(D);
%                 for s = 1:nSubj
%                     Ti = steps(s);
%                     if Ti < Tmax
%                         D(s, Ti+1:end) = NaN;
%                     end
%                 end
% 
%                 % mean delta
%                 muD_i = nansum(D,2) ./ steps;
% 
%                 % conditional means
%                 Dpos = D; Dpos(Dpos<=0) = NaN;
%                 Dneg = D; Dneg(Dneg>=0) = NaN;
% 
%                 nPos = sum(isfinite(Dpos),2);
%                 nNeg = sum(isfinite(Dneg),2);
% 
%                 muPos = nansum(Dpos,2)./nPos; muPos(nPos==0)=NaN;    % E[δ|δ>0]
%                 muNeg = nansum(Dneg,2)./nNeg; muNeg(nNeg==0)=NaN;    % E[δ|δ<0] (RAW, <=0)
% 
%                 % asymmetry index (keep your definition, but use |muNeg| for asymmetry)
%                 asy_i = muPos - abs(muNeg);
% 
%                 muD_pair(:,kk)   = muD_i;
%                 muPos_pair(:,kk) = muPos;
%                 muNeg_pair(:,kk) = muNeg;  % UPDATED
%                 asy_pair(:,kk)   = asy_i;
%             end
% 
%             muD_sub   = nanmean(muD_pair, 2);
%             muPos_sub = nanmean(muPos_pair, 2);
%             muNeg_sub = nanmean(muNeg_pair, 2);  % UPDATED
%             asy_sub   = nanmean(asy_pair, 2);
% 
%             ok = isfinite(muD_sub);
%             sum_muD(ok,ic) = sum_muD(ok,ic) + muD_sub(ok);
%             cnt_muD(ok,ic) = cnt_muD(ok,ic) + 1;
% 
%             ok = isfinite(muPos_sub);
%             sum_muPosD(ok,ic) = sum_muPosD(ok,ic) + muPos_sub(ok);
%             cnt_muPosD(ok,ic) = cnt_muPosD(ok,ic) + 1;
% 
%             ok = isfinite(muNeg_sub);
%             sum_muNegD(ok,ic) = sum_muNegD(ok,ic) + muNeg_sub(ok);
%             cnt_muNegD(ok,ic) = cnt_muNegD(ok,ic) + 1;
% 
%             ok = isfinite(asy_sub);
%             sum_asy(ok,ic) = sum_asy(ok,ic) + asy_sub(ok);
%             cnt_asy(ok,ic) = cnt_asy(ok,ic) + 1;
%         end
%     end
% end
% 
% %% ---- finalize pooled metrics (A,S-averaged) ----
% muD_pool   = sum_muD    ./ max(cnt_muD,1);
% muPos_pool = sum_muPosD ./ max(cnt_muPosD,1);
% muNeg_pool = sum_muNegD ./ max(cnt_muNegD,1);  % UPDATED (raw negative)
% asy_pool   = sum_asy    ./ max(cnt_asy,1);
% 
% % ---- normalize within condition (z-score) for first two panels only ----
% muD_z = nan(size(muD_pool));
% asy_z = nan(size(asy_pool));
% for ic = 1:nCond
%     muD_z(:,ic) = muD_pool(:,ic);
%     asy_z(:,ic) = asy_pool(:,ic);
% end
% 
% % ---- bin by gamma ----
% gammaEdges   = linspace(min(gamma_subj), max(gamma_subj), NBINS+1);
% gammaCenters = (gammaEdges(1:end-1) + gammaEdges(2:end))/2;
% cols = lines(nCond);
% 
% % ===================== Figure: 2x2 panels =====================
% figure('Color','w','Units','inches','Position',[1 1 10.2 7.0]);
% tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
% 
% % (1) mean delta (z)
% nexttile; hold on;
% for ic = 1:nCond
%     [m,ci] = binned_mean_ci(gamma_subj, muD_z(:,ic), gammaEdges);
%     errorbar(gammaCenters, m, ci, '-o', 'LineWidth',2, 'MarkerSize',6, 'Color', cols(ic,:));
% end
% xlabel('\gamma'); ylabel('mean delta'); title('Mean \delta vs \gamma (pooled A,S)');
% yline(0,':'); axis tight; legend(condNames,'Box','off','Location','best');
% 
% % (2) asymmetry (z)
% nexttile; hold on;
% for ic = 1:nCond
%     [m,ci] = binned_mean_ci(gamma_subj, asy_z(:,ic), gammaEdges);
%     errorbar(gammaCenters, m, ci, '-o', 'LineWidth',2, 'MarkerSize',6, 'Color', cols(ic,:));
% end
% xlabel('\gamma'); ylabel('asymmetry index'); title('Asymmetry vs \gamma (pooled A,S)');
% yline(0,':'); axis tight;
% 
% % (3) positive conditional mean (RAW)
% nexttile; hold on;
% for ic = 1:nCond
%     [m,ci] = binned_mean_ci(gamma_subj, muPos_pool(:,ic), gammaEdges);
%     errorbar(gammaCenters, m, ci, '-o', 'LineWidth',2, 'MarkerSize',6, 'Color', cols(ic,:));
% end
% xlabel('\gamma'); ylabel('E[\delta\mid\delta>0]'); title('Positive PE vs \gamma (pooled A,S)');
% yline(0,':'); axis tight;
% 
% % (4) negative conditional mean (RAW; negative-valued)
% nexttile; hold on;
% for ic = 1:nCond
%     [m,ci] = binned_mean_ci(gamma_subj, muNeg_pool(:,ic), gammaEdges);
%     errorbar(gammaCenters, m, ci, '-o', 'LineWidth',2, 'MarkerSize',6, 'Color', cols(ic,:));
% end
% xlabel('\gamma'); ylabel('E[\delta\mid\delta<0]'); title('Negative PE vs \gamma (pooled A,S)');
% yline(0,':'); axis tight;
% 
% % ---------- helper ----------
% function [mu, ci] = binned_mean_ci(x, y, edges)
%     nb = numel(edges)-1;
%     mu = nan(nb,1);
%     ci = nan(nb,1);
%     for b = 1:nb
%         inb = x >= edges(b) & x < edges(b+1);
%         yy  = y(inb);
%         mu(b) = mean(yy,'omitnan');
%         if sum(isfinite(yy)) > 1
%             ci(b) = 1.96 * std(yy,'omitnan') / sqrt(sum(isfinite(yy)));
%         end
%     end
% end




% %% Figure1B Steps-to-goal summary (mean ± SEM)
% % 2 panels: by S and by A
% % 3 conditions overlaid, same y-axis
% load('ready to run workspace.mat');
% condNames = {'Control','Dim-Add', 'Rejection'};
% A  = SRes(1).A_list(:);
% 
% 
% Svals   = arrayfun(@(z) z.S, SRes);
% nS      = numel(Svals);
% Slabels = arrayfun(@(s) sprintf('%.1f',s), Svals, 'uni', 0);
% 
% A       = SRes(1).A_list(:);
% nA      = numel(A);
% Alabels = compose('%.3g',A);
% 
% nCond = numel(condNames);
% 
% condColors = [ ...
%     0.20 0.20 0.20;   % Control
%     0.20 0.55 0.20;   % Dim-add
%     0.80 0.20 0.20];  % Reject
% if size(condColors,1) < nCond
%     condColors = lines(nCond);
% end
% 
% fontAxis  = 18;
% fontTick  = 18;
% fontTitle = 18;
% LW = 3.0;
% MS = 8.5;
% 
% fSteps = figure('Units','inches','Position',[1 1 9.8 4.2], ...
%     'Color','w','Renderer','painters');
% tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
% 
% % ==================== Panel 1: Steps vs S ====================
% axS = nexttile; hold(axS,'on');
% 
% for ic = 1:nCond
%     stepsSA = cell(nS,nA);
% 
%     for iS = 1:nS
%         matName = sprintf('meta_all_runs_s%.1f.mat', Svals(iS));
%         tmp = load(matName,'metaRes');
%         metaRes = tmp.metaRes;
% 
%         pairs = metaCells{ic};
%         for ia = 1:nA
%             v = [];
%             for kk = 1:size(pairs,1)
%                 traj = metaRes{pairs(kk,1),pairs(kk,2)}.res.traj{condIdx(ic),ia};
%                 v = [v; traj.steps_taken(:)];
%             end
%             stepsSA{iS,ia} = v(isfinite(v));
%         end
%     end
% 
%     mS  = nan(nS,1);
%     seS = nan(nS,1);
%     for iS = 1:nS
%         v = vertcat(stepsSA{iS,:});
%         mS(iS)  = mean(v,'omitnan');
%         seS(iS) = std(v,'omitnan') / sqrt(numel(v));
%     end
% 
%     errorbar(axS,1:nS,mS,seS,'-o', ...
%         'Color',condColors(ic,:), ...
%         'LineWidth',LW,'MarkerSize',MS, ...
%         'MarkerFaceColor','w','CapSize',0);
% end
% 
% axS.Box='off'; axS.TickDir='out'; axS.LineWidth=2;
% axS.FontName='Arial'; axS.FontSize=fontTick;
% axS.XTick=1:nS; axS.XTickLabel=Slabels;
% xlabel(axS,'S','FontSize',fontAxis,'FontWeight','bold');
% ylabel(axS,'Steps to goal','FontSize',fontAxis,'FontWeight','bold');
% %title(axS,'Steps to goal vs S','FontSize',fontTitle,'FontWeight','bold');
% 
% % ==================== Panel 2: Steps vs A ====================
% axA = nexttile; hold(axA,'on');
% 
% for ic = 1:nCond
%     stepsSA = cell(nS,nA);
% 
%     for iS = 1:nS
%         matName = sprintf('meta_all_runs_s%.1f.mat', Svals(iS));
%         tmp = load(matName,'metaRes');
%         metaRes = tmp.metaRes;
% 
%         pairs = metaCells{ic};
%         for ia = 1:nA
%             v = [];
%             for kk = 1:size(pairs,1)
%                 traj = metaRes{pairs(kk,1),pairs(kk,2)}.res.traj{condIdx(ic),ia};
%                 v = [v; traj.steps_taken(:)];
%             end
%             stepsSA{iS,ia} = v(isfinite(v));
%         end
%     end
% 
%     mA  = nan(nA,1);
%     seA = nan(nA,1);
%     for ia = 1:nA
%         v = vertcat(stepsSA{:,ia});
%         mA(ia)  = mean(v,'omitnan');
%         seA(ia) = std(v,'omitnan') / sqrt(numel(v));
%     end
% 
%     errorbar(axA,1:nA,mA,seA,'-o', ...
%         'Color',condColors(ic,:), ...
%         'LineWidth',LW,'MarkerSize',MS, ...
%         'MarkerFaceColor','w','CapSize',0);
% end
% 
% axA.Box='off'; axA.TickDir='out'; axA.LineWidth=1.2;
% axA.FontName='Arial'; axA.FontSize=fontTick;
% axA.XTick=1:nA; axA.XTickLabel=Alabels;
% xlabel(axA,'A','FontSize',fontAxis);
% title(axA,'Steps to goal vs A','FontSize',fontTitle,'FontWeight','bold');
% 
% % ==================== harmonize y-axis + legend ====================
% yl = [min([ylim(axS) ylim(axA)]) max([ylim(axS) ylim(axA)])];
% ylim(axS,yl); ylim(axA,yl);
% 
% legend(axS,condNames,'Box','off','Location','northwest','FontSize',14);
% 
% % exportgraphics(fSteps, ...
% %     fullfile(outdir,'StepsToGoal_MeanSEM_CombinedConditions.pdf'), ...
% %     'ContentType','vector');
% 
% %Stats for Figure1B 
% 
% Svals   = arrayfun(@(z) z.S, SRes);
% nS      = numel(Svals);
% A       = SRes(1).A_list(:);
% nA      = numel(A);
% nCond   = numel(condNames);
% 
% % Storage for plotted means/SEMs (so trends match your plots)
% m_byS  = nan(nCond,nS);  se_byS  = nan(nCond,nS);   n_byS  = nan(nCond,nS);
% m_byA  = nan(nCond,nA);  se_byA  = nan(nCond,nA);   n_byA  = nan(nCond,nA);
% 
% % Also keep raw pooled vectors for pairwise tests
% raw_byS = cell(nCond,nS);   % pooled over A
% raw_byA = cell(nCond,nA);   % pooled over S
% 
% % -------- collect raw steps per (cond, S, A) once ----------
% stepsSA_cond = cell(nCond, nS, nA);
% 
% for ic = 1:nCond
%     for iS = 1:nS
%         matName = sprintf('meta_all_runs_s%.1f.mat', Svals(iS));
%         tmp = load(matName,'metaRes');
%         metaRes = tmp.metaRes;
% 
%         pairs = metaCells{ic};
%         for ia = 1:nA
%             v = [];
%             for kk = 1:size(pairs,1)
%                 traj = metaRes{pairs(kk,1),pairs(kk,2)}.res.traj{condIdx(ic),ia};
%                 v = [v; traj.steps_taken(:)];
%             end
%             v = v(isfinite(v));
%             stepsSA_cond{ic,iS,ia} = v;
%         end
%     end
% end
% 
% % -------- reduce to (cond x S) pooled over A, and (cond x A) pooled over S ----------
% for ic = 1:nCond
%     % by S (pool A)
%     for iS = 1:nS
%         v = vertcat(stepsSA_cond{ic,iS,:});
%         v = v(isfinite(v));
%         raw_byS{ic,iS} = v;
%         n_byS(ic,iS)  = numel(v);
%         m_byS(ic,iS)  = mean(v,'omitnan');
%         se_byS(ic,iS) = std(v,'omitnan')/sqrt(max(numel(v),1));
%     end
% 
%     % by A (pool S)
%     for ia = 1:nA
%         v = vertcat(stepsSA_cond{ic,:,ia});
%         v = v(isfinite(v));
%         raw_byA{ic,ia} = v;
%         n_byA(ic,ia)  = numel(v);
%         m_byA(ic,ia)  = mean(v,'omitnan');
%         se_byA(ic,ia) = std(v,'omitnan')/sqrt(max(numel(v),1));
%     end
% end
% 
% % -------------------- print summary tables --------------------
% fprintf('\n================ Figure1B summary: Steps-to-goal (mean ± SEM) ================\n');
% 
% fprintf('\n--- By S (pooled over A) ---\n');
% for ic = 1:nCond
%     fprintf('%s:\n', condNames{ic});
%     for iS = 1:nS
%         fprintf('  S=%.1f:  %.2f ± %.2f  (n=%d)\n', Svals(iS), m_byS(ic,iS), se_byS(ic,iS), n_byS(ic,iS));
%     end
% end
% 
% fprintf('\n--- By A (pooled over S) ---\n');
% for ic = 1:nCond
%     fprintf('%s:\n', condNames{ic});
%     for ia = 1:nA
%         fprintf('  A=%g:  %.2f ± %.2f ', A(ia), m_byA(ic,ia), se_byA(ic,ia));
%     end
% end
% 
% % -------------------- trend tests (linear slope + p) --------------------
% fprintf('\n================ Trend tests (linear) ================\n');
% for ic = 1:nCond
%     % steps vs S
%     Xs = Svals(:);
%     Ys = m_byS(ic,:).';
%     [bS,~,~,~,statsS] = regress(Ys, [ones(size(Xs)) Xs]);
%     slopeS = bS(2);
%     pS = statsS(3);
% 
%     % steps vs A
%     Xa = A(:);
%     Ya = m_byA(ic,:).';
%     [bA,~,~,~,statsA] = regress(Ya, [ones(size(Xa)) Xa]);
%     slopeA = bA(2);
%     pA = statsA(3);
% 
%     fprintf('%s:\n', condNames{ic});
%     fprintf('  Steps vs S: slope = %.3f steps per S-unit, p = %.3g\n', slopeS, pS);
%     fprintf('  Steps vs A: slope = %.3f steps per A-unit, p = %.3g\n', slopeA, pA);
% end
% 
% % -------------------- pairwise contrasts (Welch t-test + Cohen d) --------------------
% fprintf('\n================ Pairwise condition contrasts (Welch t-test) ================\n');
% pairsCond = nchoosek(1:nCond,2);
% 
% fprintf('\n--- At each S (pooled over A) ---\n');
% for iS = 1:nS
%     fprintf('S=%.1f:\n', Svals(iS));
%     for pp = 1:size(pairsCond,1)
%         c1 = pairsCond(pp,1); c2 = pairsCond(pp,2);
%         v1 = raw_byS{c1,iS};  v2 = raw_byS{c2,iS};
%         [~,p,~,st] = ttest2(v1,v2,'Vartype','unequal');
%         d = cohend_welch(v1,v2);
%         fprintf('  %s vs %s: Δmean = %.2f, t(%0.1f)=%.2f, p=%.3g, d=%.2f\n', ...
%             condNames{c1}, condNames{c2}, mean(v1)-mean(v2), st.df, st.tstat, p, d);
%     end
% end
% 
% fprintf('\n--- At each A (pooled over S) ---\n');
% for ia = 1:nA
%     fprintf('A=%g:\n', A(ia));
%     for pp = 1:size(pairsCond,1)
%         c1 = pairsCond(pp,1); c2 = pairsCond(pp,2);
%         v1 = raw_byA{c1,ia};  v2 = raw_byA{c2,ia};
%         [~,p,~,st] = ttest2(v1,v2,'Vartype','unequal');
%         d = cohend_welch(v1,v2);
%         fprintf('  %s vs %s: Δmean = %.2f, t(%0.1f)=%.2f, p=%.3g, d=%.2f\n', ...
%             condNames{c1}, condNames{c2}, mean(v1)-mean(v2), st.df, st.tstat, p, d);
%     end
% end
% 
% fprintf('\n=============================================================================\n');
% 
% % -------- helper: Cohen''s d with Welch pooled SD (robust to unequal n/var) --------
% function d = cohend_welch(x,y)
%     x = x(isfinite(x)); y = y(isfinite(y));
%     nx = numel(x); ny = numel(y);
%     vx = var(x,1);  vy = var(y,1);   % population variance estimate (N), stable
%     s = sqrt( (vx + vy)/2 );
%     if s == 0
%         d = NaN;
%     else
%         d = (mean(x)-mean(y))/s;
%     end
% end

%% ===================== Pooled steps-to-goal distribution (ALL S, ALL A) =====================
% Requires: SRes, metaCells, condIdx, condNames
% Loads metaRes from meta_all_runs_sX.mat for each S in SRes and pools steps_taken across all A and metaCells pairs.
condNames = {'Control','Dim-Add', 'Rejection'};
if ~exist('matDir','var') || isempty(matDir), matDir = pwd; end

Svals = arrayfun(@(z) z.S, SRes);
nS    = numel(Svals);

A  = SRes(1).A_list(:);
nA = numel(A);

nCond = numel(condNames);

% ---------- collect pooled steps per condition ----------
stepsAll = cell(nCond,1);

for ic = 1:nCond
    X = [];

    pairs = metaCells{ic};  % [id ip] rows

    for iS = 1:nS
        matName = fullfile(matDir, sprintf('meta_all_runs_s%.1f.mat', Svals(iS)));
        tmp = load(matName,'metaRes');
        metaRes = tmp.metaRes;

        for ia = 1:nA
            for kk = 1:size(pairs,1)
                id = pairs(kk,1);
                ip = pairs(kk,2);

                % guard against meta-grid mismatch
                if id > size(metaRes,1) || ip > size(metaRes,2)
                    continue;
                end

                traj = metaRes{id,ip}.res.traj{condIdx(ic), ia};
                if ~isfield(traj,'steps_taken') || isempty(traj.steps_taken), continue; end

                v = traj.steps_taken(:);
                v = v(isfinite(v));
                X = [X; v]; %#ok<AGROW>
            end
        end
    end

    stepsAll{ic} = X;
    fprintf('Pooled %-10s: n=%d\n', condNames{ic}, numel(X));
end

% ---------- visualize distribution (KDE + light histogram) ----------
condColors = [ ...
    0.20 0.20 0.20;   % Control
    0.20 0.55 0.20;   % Dim-add
    0.80 0.20 0.20];  % Reject
if size(condColors,1) < nCond
    condColors = lines(nCond);
end

fontAxis = 18;
fontTick = 16;
LW = 3.0;

% robust x-grid
xAll = vertcat(stepsAll{:});
xAll = xAll(isfinite(xAll));
if isempty(xAll), error('No steps data found after pooling.'); end
xMin = max(0, floor(prctile(xAll,0.5)));
xMax = ceil(prctile(xAll,99.5));
xGrid = linspace(xMin, xMax, 400);

fDist = figure('Units','inches','Position',[1 1 8.8 4.8], 'Color','w','Renderer','painters');
ax = axes(fDist); hold(ax,'on');


for ic = 1:nCond
    v = stepsAll{ic};
    v = v(isfinite(v));
    if numel(v) < 10, continue; end

    % KDE
    [f, xk] = ksdensity(v, xGrid);
    plot(ax, xk, f, '-', 'Color',condColors(ic,:), 'LineWidth',LW);
end

ax.Box='off'; ax.TickDir='out'; ax.LineWidth=1.5;
ax.FontName='Arial'; ax.FontSize=fontTick;
xlabel(ax,'Steps to goal','FontSize',fontAxis,'FontWeight','bold');
ylabel(ax,'Probability density','FontSize',fontAxis,'FontWeight','bold');

leg = legend(ax, condNames, 'Box','off', 'Location','northeast');
leg.FontSize = 14;

% ---------- print stats (similar spirit to your original code) ----------
fprintf('\n================ Pooled steps-to-goal stats (ALL S, ALL A) ================\n');

m  = nan(nCond,1);
se = nan(nCond,1);
med = nan(nCond,1);
iqrV = nan(nCond,1);
p95 = nan(nCond,1);
p99 = nan(nCond,1);
nobs = nan(nCond,1);

for ic = 1:nCond
    v = stepsAll{ic};
    v = v(isfinite(v));
    nobs(ic) = numel(v);

    m(ic)   = mean(v,'omitnan');
    se(ic)  = std(v,'omitnan') / sqrt(max(numel(v),1));
    med(ic) = median(v,'omitnan');
    iqrV(ic)= iqr(v);
    p95(ic) = prctile(v,95);
    p99(ic) = prctile(v,99);

    fprintf('%-10s: mean ± SEM = %.2f ± %.2f (n=%d), median [IQR] = %.2f [%.2f], p95=%.2f, p99=%.2f\n', ...
        condNames{ic}, m(ic), se(ic), nobs(ic), med(ic), iqrV(ic), p95(ic), p99(ic));
end

% ---------- pairwise contrasts (Welch t-test + Cohen d; optional KS) ----------
fprintf('\n================ Pairwise contrasts (pooled) ================\n');
pairsCond = nchoosek(1:nCond,2);

for pp = 1:size(pairsCond,1)
    c1 = pairsCond(pp,1);
    c2 = pairsCond(pp,2);

    v1 = stepsAll{c1}; v1 = v1(isfinite(v1));
    v2 = stepsAll{c2}; v2 = v2(isfinite(v2));

    % Welch t-test
    [~,p,~,st] = ttest2(v1, v2, 'Vartype','unequal');

    % Cohen d (Welch-style pooled SD)
    vx = var(v1,1); vy = var(v2,1);
    s  = sqrt((vx+vy)/2);
    if s==0, d = NaN; else, d = (mean(v1)-mean(v2))/s; end

    % Optional KS test (distribution difference)
    try
        [hks, pks] = kstest2(v1, v2);
    catch
        hks = NaN; pks = NaN;
    end

    fprintf('%s vs %s:\n', condNames{c1}, condNames{c2});
    fprintf('  Δmean = %.2f, Welch t(%0.1f)=%.2f, p=%.3g, d=%.2f\n', ...
        mean(v1)-mean(v2), st.df, st.tstat, p, d);
    fprintf('  KS test: h=%g, p=%.3g\n', hks, pks);
end

fprintf('=============================================================================\n');



%% figure 2A1.2: Outcomes + asymmetry vs A for ALL 3 conditions (Control / Dim-add / Reject), overlay S
% This version FIXES the A-label sorting bug by:
%   (1) sorting A once -> A_sorted and index ix
%   (2) reordering all plotted vectors by ix
%   (3) setting XTickLabel ONCE per axis to A_sorted (NOT inside the iS loop)
condNames = {'Control','Dim-Add', 'Rejection'};
nS = numel(SRes);
A  = SRes(1).A_list(:);
nA = numel(A);

% ---- sort A so x-axis goes from negative A to positive A ----
[A_sorted, ix] = sort(A, 'ascend');   % ix maps original -> sorted order
x  = 1:nA;                            % x positions stay 1..nA, labels show A_sorted

nCond = size(SRes(1).muR, 2);


S_colors = lines(nS);
S_labels = arrayfun(@(z) sprintf('S=%.1f', z.S), SRes, 'uni', 0);

mkR = 'o'; lsR = '-';     % reward
mkD = 's'; lsD = '--';    % delta

LW = 2.0;
MS = 8.0;
ALPHA_BAND = 0.12;

drawBand = @(ax,x,y,ci,c) patch(ax,[x fliplr(x)],[(y-ci)' fliplr((y+ci)')],c, ...
    'FaceAlpha',ALPHA_BAND,'EdgeColor','none','HandleVisibility','off');

fA12b = figure('Units','inches','Position',[1 1 10.8 6.2], ...
    'Color','w','Renderer','painters');
tiledlayout(nCond,2,'TileSpacing','compact','Padding','compact');

for ic = 1:nCond

    
    % ==================== left col: mean outcomes ====================
    ax1 = nexttile; hold(ax1,'on');

    if ic == 1
    lgHandles = gobjects(nS,1);
        for iS = 1:nS
            lgHandles(iS) = plot(ax1, nan, nan, ...
                'Color', S_colors(iS,:), ...
                'LineWidth', LW, ...
                'Marker', mkD, ...
                'MarkerSize', MS, ...
                'MarkerFaceColor','w');
        end
    end
    


    for iS = 1:nS
        muR = SRes(iS).muR;
        muD = SRes(iS).muD;

        m_muR = nan(nA,1); ci_muR = nan(nA,1);
        m_muD = nan(nA,1); ci_muD = nan(nA,1);

        for ia = 1:nA
            vR = muR{ia,ic}; vR = vR(isfinite(vR));
            vD = muD{ia,ic}; vD = vD(isfinite(vD));

            m_muR(ia)  = mean(vR,'omitnan');
            ci_muR(ia) = 1.96*std(vR,'omitnan')/sqrt(max(numel(vR),1));

            m_muD(ia)  = mean(vD,'omitnan');
            ci_muD(ia) = 1.96*std(vD,'omitnan')/sqrt(max(numel(vD),1));
        end

        % --- PLOT SORTED BY A (ix) ---
        drawBand(ax1, x, m_muD(ix), ci_muD(ix), S_colors(iS,:));
        plot(ax1, x, m_muD(ix), [lsD mkD], ...
            'Color',S_colors(iS,:), 'LineWidth',LW, ...
            'MarkerSize',MS,'MarkerFaceColor','w','MarkerEdgeColor',S_colors(iS,:), ...
            'HandleVisibility','off');
        
        if ic == 1
            legend(ax1, lgHandles, S_labels, ...
                'Location','northwest', ...
                'Box','off', ...
                'FontSize',14);
        end

    end

    ax1.Box='off'; ax1.TickDir='out'; ax1.LineWidth=1.0;
    ax1.FontName='Arial'; ax1.FontSize=12.5;

    ax1.XTick = x;
    ax1.XTickLabel = compose('%.3g', A_sorted);   % ✅ sorted labels (set ONCE)
    xlabel(ax1,'A','FontSize',18,'FontWeight','bold');
    ylabel(ax1,'Mean \delta','FontSize',18,'FontWeight','bold');
    xlim(ax1,[0.7 nA+0.3]);
    yl = ylim(ax1); ylim(ax1,[yl(1) yl(2)*1.08]);
    title(ax1, sprintf('%s', condNames{ic}), 'FontWeight','bold','FontSize',18);

    % ==================== right col: BOUNDED asymmetry indices =================
    ax2 = nexttile; hold(ax2,'on');

    for iS = 1:nS
        muPosR = SRes(iS).muPosR;  muNegR = SRes(iS).muNegR;
        muPosD = SRes(iS).muPosD;  muNegD = SRes(iS).muNegD;

        m_asyRb = nan(nA,1); ci_asyRb = nan(nA,1);
        m_asyDb = nan(nA,1); ci_asyDb = nan(nA,1);

        for ia = 1:nA
            pR = muPosR{ia,ic}; nR = muNegR{ia,ic};
            pD = muPosD{ia,ic}; nD = muNegD{ia,ic};

            pR = pR(isfinite(pR)); nR = nR(isfinite(nR));
            pD = pD(isfinite(pD)); nD = nD(isfinite(nD));

            numR = mean(pR,'omitnan') - mean(abs(nR),'omitnan');
            denR = mean(pR,'omitnan') + mean(abs(nR),'omitnan');
            asyRb = numR / denR;

            numD = mean(pD,'omitnan') - mean(abs(nD),'omitnan');
            denD = mean(pD,'omitnan') + mean(abs(nD),'omitnan');
            asyDb = numD / denD;

            % bootstrap CI (distribution-level)
            B = 500;
            asyRb_bs = nan(B,1);
            asyDb_bs = nan(B,1);

            if ~isempty(pR) && ~isempty(nR)
                for b=1:B
                    pr = pR(randi(numel(pR),numel(pR),1));
                    nr = nR(randi(numel(nR),numel(nR),1));
                    asyRb_bs(b) = (mean(pr)-mean(abs(nr))) / (mean(pr)+mean(abs(nr)));
                end
            end
            if ~isempty(pD) && ~isempty(nD)
                for b=1:B
                    pd = pD(randi(numel(pD),numel(pD),1));
                    nd = nD(randi(numel(nD),numel(nD),1));
                    asyDb_bs(b) = (mean(pd)-mean(abs(nd))) / (mean(pd)+mean(abs(nd)));
                end
            end

            m_asyRb(ia)  = asyRb;
            m_asyDb(ia)  = asyDb;
            ci_asyRb(ia) = 1.96*std(asyRb_bs,'omitnan');
            ci_asyDb(ia) = 1.96*std(asyDb_bs,'omitnan');
        end

        % --- PLOT SORTED BY A (ix) ---
        drawBand(ax2, x, m_asyDb(ix), ci_asyDb(ix), S_colors(iS,:));
        plot(ax2, x, m_asyDb(ix), [lsD mkD], ...
            'Color',S_colors(iS,:), 'LineWidth',LW, ...
            'MarkerSize',MS,'MarkerFaceColor','w','MarkerEdgeColor',S_colors(iS,:), ...
            'HandleVisibility','off');
    end
end

    yline(ax2,0,':','LineWidth',2,'Color',[0 0 0],'Alpha',0.28,'HandleVisibility','off');

    ax2.Box='off'; ax2.TickDir='out'; ax2.LineWidth=1.0;
    ax2.FontName='Arial'; ax2.FontSize=14;

    ax2.XTick = x;
    ax2.XTickLabel = compose('%.3g', A_sorted);   % ✅ sorted labels (set ONCE)
    xlabel(ax2,'A','FontSize',18,'FontWeight','bold');
    ylabel(ax2,'Asymmetry index','FontSize',18,'FontWeight','bold');
    xlim(ax2,[0.7 nA+0.3]);
    ylim(ax2,[-0.2 0.7]);
    title(ax2, sprintf('%s', condNames{ic}), 'FontWeight','bold','FontSize',18);

end




















%% figure 2A1.2: Outcomes + asymmetry vs A for ALL 3 conditions (Control / Dim-add / Reject), overlay S
% Updated:
%   (1) left column uses one shared nonlinear compressed scale
%       via log1p-style transform
%   (2) all left-column subplots share identical y-limits
%   (3) A-F labels moved higher above top-right corner
%   (4) x-axis sorted by A as before

condNames = {'Control','Dim-Add','Rejection'};
nS = numel(SRes);
A  = SRes(1).A_list(:);
nA = numel(A);

% ---- sort A so x-axis goes from negative A to positive A ----
[A_sorted, ix] = sort(A, 'ascend');
x = 1:nA;

nCond = size(SRes(1).muR, 2);

S_colors = lines(nS);
S_labels = arrayfun(@(z) sprintf('S=%.1f', z.S), SRes, 'uni', 0);

mkD = 's';
lsD = '--';

LW = 2.0;
MS = 8.0;
ALPHA_BAND = 0.12;

subplotLabels = {'A','B','C','D','E','F'};


% Collect all left-column values to determine a global compression constant
allLeftVals = [];

for ic = 1:nCond
    for iS = 1:nS
        muD = SRes(iS).muD;
        for ia = 1:nA
            vD = muD{ia,ic};
            vD = vD(isfinite(vD));
            if ~isempty(vD)
                allLeftVals = [allLeftVals; vD(:)]; %#ok<AGROW>
            end
        end
    end
end

% Choose compression constant c
% A smaller c gives stronger compression.
% Here we use the 20th percentile of positive magnitudes as a stable default.
absVals = abs(allLeftVals(isfinite(allLeftVals)));
absVals = absVals(absVals > 0);

if isempty(absVals)
    c = 0.01;
else
    absVals = sort(absVals);
    idx20 = max(1, round(0.20 * numel(absVals)));
    c = absVals(idx20);
end

% Safety fallback
if ~isfinite(c) || c <= 0
    c = 0.01;
end

% Transform and inverse-transform
tf  = @(y) sign(y) .* log1p(abs(y) ./ c);
itf = @(z) sign(z) .* c .* (exp(abs(z)) - 1);

% Helper for transformed CI bands
drawBand = @(ax, x, ylo, yhi, col) ...
    patch(ax, [x fliplr(x)], [ylo(:)' fliplr(yhi(:)')], col, ...
    'FaceAlpha', ALPHA_BAND, ...
    'EdgeColor', 'none', ...
    'HandleVisibility', 'off');


% PRECOMPUTE GLOBAL Y LIMITS FOR ALL LEFT-COLUMN PANELS

allLeftLowT  = [];
allLeftHighT = [];

for ic = 1:nCond
    for iS = 1:nS
        muD = SRes(iS).muD;

        m_muD  = nan(nA,1);
        ci_muD = nan(nA,1);

        for ia = 1:nA
            vD = muD{ia,ic};
            vD = vD(isfinite(vD));

            if isempty(vD)
                m_muD(ia)  = NaN;
                ci_muD(ia) = NaN;
            else
                m_muD(ia)  = mean(vD,'omitnan');
                ci_muD(ia) = 1.96 * std(vD,'omitnan') / sqrt(numel(vD));
            end
        end

        y  = m_muD(ix);
        ci = ci_muD(ix);

        yloT = tf(y - ci);
        yhiT = tf(y + ci);

        allLeftLowT  = [allLeftLowT;  yloT(:)]; %#ok<AGROW>
        allLeftHighT = [allLeftHighT; yhiT(:)]; %#ok<AGROW>
    end
end

leftYMin = min(allLeftLowT, [], 'omitnan');
leftYMax = max(allLeftHighT, [], 'omitnan');

ypad = 0.06 * (leftYMax - leftYMin);
if ~isfinite(ypad) || ypad <= 0
    ypad = 0.1;
end
leftYLims = [leftYMin - ypad, leftYMax + ypad];

% Build nice shared y ticks in ORIGINAL units
% Manually choose some useful original-unit tick candidates spanning the data.
origMin = min(allLeftVals, [], 'omitnan');
origMax = max(allLeftVals, [], 'omitnan');

if ~isfinite(origMin), origMin = 0; end
if ~isfinite(origMax), origMax = 1; end

% Candidate ticks; MATLAB will keep the ones within range
tickCandidates = unique([ ...
    0, ...
    0.01, 0.02, 0.05, ...
    0.1, 0.2, 0.3, 0.5, ...
    0.7, 0.9, 1, 1.2, 1.5, 2, 3, 5]);

% Keep those that fall within a padded data range
tickCandidates = tickCandidates(tickCandidates >= max(0, origMin*0.9) & ...
                                tickCandidates <= origMax*1.1);

% If the data cross zero, add symmetric negatives
if origMin < 0
    tickCandidates = unique([-fliplr(tickCandidates(tickCandidates>0)), 0, tickCandidates]);
    tickCandidates = tickCandidates(tickCandidates >= origMin*1.1 & tickCandidates <= origMax*1.1);
end

leftTickPos = tf(tickCandidates);
keepTicks = leftTickPos >= leftYLims(1) & leftTickPos <= leftYLims(2);
leftTickPos = leftTickPos(keepTicks);
leftTickLab = compose('%.2g', tickCandidates(keepTicks));

% PLOT

fA12b = figure('Units','inches','Position',[1 1 10.8 6.2], ...
    'Color','w','Renderer','painters');

tiledlayout(nCond, 2, 'TileSpacing','compact', 'Padding','compact');

tileCounter = 0;

for ic = 1:nCond

    % ==================== LEFT COLUMN: Mean delta ====================
    ax1 = nexttile;
    hold(ax1, 'on');
    tileCounter = tileCounter + 1;

    if ic == 1
        lgHandles = gobjects(nS,1);
        for iS = 1:nS
            lgHandles(iS) = plot(ax1, nan, nan, ...
                'Color', S_colors(iS,:), ...
                'LineWidth', LW, ...
                'Marker', mkD, ...
                'MarkerSize', MS, ...
                'MarkerFaceColor', 'w');
        end
    end

    for iS = 1:nS
        muD = SRes(iS).muD;

        m_muD  = nan(nA,1);
        ci_muD = nan(nA,1);

        for ia = 1:nA
            vD = muD{ia,ic};
            vD = vD(isfinite(vD));

            if isempty(vD)
                m_muD(ia)  = NaN;
                ci_muD(ia) = NaN;
            else
                m_muD(ia)  = mean(vD, 'omitnan');
                ci_muD(ia) = 1.96 * std(vD, 'omitnan') / sqrt(numel(vD));
            end
        end

        y  = m_muD(ix);
        ci = ci_muD(ix);

        yT   = tf(y);
        yLoT = tf(y - ci);
        yHiT = tf(y + ci);

        drawBand(ax1, x, yLoT, yHiT, S_colors(iS,:));

        plot(ax1, x, yT, [lsD mkD], ...
            'Color', S_colors(iS,:), ...
            'LineWidth', LW, ...
            'MarkerSize', MS, ...
            'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor', S_colors(iS,:), ...
            'HandleVisibility', 'off');
    end

    if ic == 1
        legend(ax1, lgHandles, S_labels, ...
            'Location', 'northwest', ...
            'Box', 'off', ...
            'FontSize', 15);
    end

    ax1.Box = 'off';
    ax1.TickDir = 'out';
    ax1.LineWidth = 1.0;
    ax1.FontName = 'Arial';
    ax1.FontSize = 12.5;

    ax1.XTick = x;
    ax1.XTickLabel = compose('%.3g', A_sorted);
    xlabel(ax1, 'A', 'FontSize',18, 'FontWeight','bold');
    ylabel(ax1, 'Mean \delta', 'FontSize',18, 'FontWeight','bold');
    xlim(ax1, [0.7 nA+0.3]);

    % SHARED transformed limits and shared ticks for all left-column panels
    ylim(ax1, leftYLims);
    ax1.YTick = leftTickPos;
    ax1.YTickLabel = leftTickLab;

    title(ax1, condNames{ic}, 'FontWeight','bold', 'FontSize',18);

    % Higher subplot label
    text(ax1, 0.98, 1.04, subplotLabels{tileCounter}, ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'bottom', ...
        'FontWeight', 'bold', ...
        'FontSize', 28, ...
        'Clipping', 'off');

    % ==================== RIGHT COLUMN: Asymmetry index ====================
    ax2 = nexttile;
    hold(ax2, 'on');
    tileCounter = tileCounter + 1;

    for iS = 1:nS
        muPosR = SRes(iS).muPosR; %#ok<NASGU>
        muNegR = SRes(iS).muNegR; %#ok<NASGU>
        muPosD = SRes(iS).muPosD;
        muNegD = SRes(iS).muNegD;

        m_asyDb  = nan(nA,1);
        ci_asyDb = nan(nA,1);

        for ia = 1:nA
            pD = muPosD{ia,ic};
            nD = muNegD{ia,ic};

            pD = pD(isfinite(pD));
            nD = nD(isfinite(nD));

            numD = mean(pD,'omitnan') - mean(abs(nD),'omitnan');
            denD = mean(pD,'omitnan') + mean(abs(nD),'omitnan');
            asyDb = numD / denD;

            B = 500;
            asyDb_bs = nan(B,1);

            if ~isempty(pD) && ~isempty(nD)
                for b = 1:B
                    pd = pD(randi(numel(pD), numel(pD), 1));
                    nd = nD(randi(numel(nD), numel(nD), 1));
                    asyDb_bs(b) = (mean(pd) - mean(abs(nd))) / (mean(pd) + mean(abs(nd)));
                end
            end

            m_asyDb(ia)  = asyDb;
            ci_asyDb(ia) = 1.96 * std(asyDb_bs, 'omitnan');
        end

        y = m_asyDb(ix);
        ci = ci_asyDb(ix);

        drawBand(ax2, x, y-ci, y+ci, S_colors(iS,:));

        plot(ax2, x, y, [lsD mkD], ...
            'Color', S_colors(iS,:), ...
            'LineWidth', LW, ...
            'MarkerSize', MS, ...
            'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor', S_colors(iS,:), ...
            'HandleVisibility', 'off');
    end

    yline(ax2, 0, ':', 'LineWidth', 2, 'Color', [0 0 0], 'HandleVisibility','off');

    ax2.Box = 'off';
    ax2.TickDir = 'out';
    ax2.LineWidth = 1.0;
    ax2.FontName = 'Arial';
    ax2.FontSize = 14;

    ax2.XTick = x;
    ax2.XTickLabel = compose('%.3g', A_sorted);
    xlabel(ax2, 'A', 'FontSize',18, 'FontWeight','bold');
    ylabel(ax2, 'Asymmetry index', 'FontSize',18, 'FontWeight','bold');
    xlim(ax2, [0.7 nA+0.3]);
    ylim(ax2, [-0.2 0.7]);

    title(ax2, condNames{ic}, 'FontWeight','bold', 'FontSize',18);

    % Higher subplot label
    text(ax2, 0.98, 1.04, subplotLabels{tileCounter}, ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'bottom', ...
        'FontWeight', 'bold', ...
        'FontSize', 28, ...
        'Clipping', 'off');
end





%% Figure 3 (condensed): Ratio of negative/positive hedonic experience pooled over S (single panel, x=A)
% Plot ratio per subject:  |E[h | h<0]| / E[h | h>0|
% pooled across ALL S, for each A and condition.
%
% Assumes:
%   SRes(iS).muPosD and SRes(iS).muNegD are cell(nA,nCond) with subject-level values
%   condNames exists
%   outdir exists (optional for export)

condNames = {'Control','Dim-Add', 'Rejection'};
Svals   = arrayfun(@(z) z.S, SRes);
nS      = numel(Svals);

A       = SRes(1).A_list(:);
nA      = numel(A);
Alabels = compose('%.3g',A);

% ---- sort A for plotting: negative -> positive ----
[A_sorted, ix] = sort(A, 'ascend');          % ix maps sorted -> original
Alabels_sorted = compose('%.3g', A_sorted);
x = 1:nA;                                   % plotting positions stay 1..nA

nCond = numel(condNames);

condColors = [ ...
    0.20 0.20 0.20;   % Control
    0.20 0.55 0.20;   % Dim-add
    0.80 0.20 0.20];  % Reject
if size(condColors,1) < nCond
    condColors = lines(nCond);
end

% --- typography (bigger everywhere) ---
fontTitle = 18;
fontAxis  = 18;
fontTick  = 18;
fontLeg   = 18;

% --- line widths ---
LW_axis = 1.5;
LW_box  = 1.5;
LW_med  = 2.5;
LW_edge = 1.1;

% --- alpha settings ---
violinAlpha  = 0.22;
boxFaceAlpha = 0.12;

% --- spacing for 3 conditions around each A ---
offsets = linspace(-0.25, 0.25, nCond);
boxW = 0.08;        % half-width of box overlay
violinW = 0.12;     % max half-width of violin

% -------------------- pooled container: ratio --------------------
% ratio per observation: abs(neg) ./ pos  (drop invalid, drop pos<=0, drop neg>=0, drop div-by-0)
ratio_pool = cell(nA,nCond);

for ia = 1:nA
    ia0 = ix(ia);   % original A index in SRes (because ix sorts A)
    for ic = 1:nCond
        vr = [];
        for iS = 1:nS
            p = SRes(iS).muPosD{ia0,ic}; p = p(isfinite(p));
            n = SRes(iS).muNegD{ia0,ic}; n = n(isfinite(n));

            if isempty(p) || isempty(n)
                continue;
            end

            % pair up to preserve within-subject ratio
            m = min(numel(p), numel(n));
            if m < 1
                continue;
            end
            p = p(1:m);
            n = n(1:m);

            ok = isfinite(p) & isfinite(n) & (p > 0) & (n < 0);
            if ~any(ok), continue; end

            r = abs(n(ok)) ./ p(ok);
            r = r(isfinite(r));

            vr = [vr; r(:)]; %#ok<AGROW>
        end
        ratio_pool{ia,ic} = vr;
    end
end

% -------------------- Figure: Ratio (pooled over S) --------------------
fRatioPool = figure('Units','inches','Position',[1 1 9.2 4.6], 'Color','w','Renderer','painters');
ax = axes(fRatioPool); hold(ax,'on');

for ia = 1:nA
    for ic = 1:nCond
        v = ratio_pool{ia,ic};
        v = v(isfinite(v));
        if numel(v) < 5, continue; end

        x0 = ia + offsets(ic);

        %violin11(ax, v, x0, condColors(ic,:), violinAlpha, violinW);

        [q1,q3,med,wlo,whi,mu] = boxStats(v);

        patch(ax, [x0-boxW x0+boxW x0+boxW x0-boxW], [q1 q1 q3 q3], condColors(ic,:), ...
            'FaceAlpha',boxFaceAlpha,'EdgeColor',condColors(ic,:),'LineWidth',LW_box, ...
            'HandleVisibility','off');

        plot(ax,[x0 x0],[wlo q1],'-','Color',condColors(ic,:),'LineWidth',LW_box,'HandleVisibility','off');
        plot(ax,[x0 x0],[q3 whi],'-','Color',condColors(ic,:),'LineWidth',LW_box,'HandleVisibility','off');
        plot(ax,[x0-0.04 x0+0.04],[wlo wlo],'-','Color',condColors(ic,:),'LineWidth',LW_box,'HandleVisibility','off');
        plot(ax,[x0-0.04 x0+0.04],[whi whi],'-','Color',condColors(ic,:),'LineWidth',LW_box,'HandleVisibility','off');

        plot(ax,[x0-boxW x0+boxW],[med med],'-','Color',condColors(ic,:), ...
            'LineWidth',LW_med,'HandleVisibility','off');

        plot(ax,x0,mu,'o','MarkerSize',6.5,'MarkerFaceColor','w', ...
            'MarkerEdgeColor',condColors(ic,:),'LineWidth',LW_edge,'HandleVisibility','off');
    end
end

styleAxes(ax, fontTick, LW_axis);
ax.XLim = [0.5 nA+0.5];
ax.XTick = 1:nA;
ax.XTickLabel = Alabels_sorted;
xlabel(ax,'A','FontSize',fontAxis,'FontWeight','bold');
ylabel(ax,'|E[h \mid h<0]| / E[h \mid h>0]','FontSize',fontAxis,'FontWeight','bold');
ax.YGrid = 'on';
ax.GridAlpha = 0.06;

% legend (same style as before)
lgHandles = gobjects(nCond,1);
for ic=1:nCond
    lgHandles(ic) = plot(ax,nan,nan,'-','Color',condColors(ic,:),'LineWidth',3);
end
leg = legend(ax,lgHandles,condNames,'Box','off','Location','southoutside','NumColumns',nCond);
leg.FontSize = fontLeg;

% optional export:
% exportgraphics(fRatioPool, fullfile(outdir,'Figure3_RatioNegPos_pooledOverS.pdf'), 'ContentType','vector');

%% POOL OVER S (given the ratio_pool{ia,ic} already computed by the previous code)
% This pools across ALL A and ALL S (since ratio_pool already pooled over S per-A),
% then makes ONE distribution per condition.

condNames = {'Control','Dim-Add', 'Rejection'};
nCond = numel(condNames);
nA    = size(ratio_pool,1);

% ---------- pool across A (and thus across S) ----------
ratio_pool_all = cell(nCond,1);
for ic = 1:nCond
    v = [];
    for ia = 1:nA
        tmp = ratio_pool{ia,ic};
        tmp = tmp(isfinite(tmp));
        v = [v; tmp(:)]; %#ok<AGROW>
    end
    ratio_pool_all{ic} = v;
end

% ---------- simple summary plot WITHOUT violins (box + mean dot) ----------
fRatioByCond = figure('Units','inches','Position',[1 1 6.6 4.4], ...
    'Color','w','Renderer','painters');
ax = axes(fRatioByCond); hold(ax,'on');

LW_axis = 1.5; LW_box = 1.6; LW_med = 2.6; LW_edge = 1.2;
fontAxis = 18; fontTick = 16; fontTitle = 18;

boxFaceAlpha = 0.12;
boxW = 0.22;

condColors = [ ...
    0.20 0.20 0.20;   % Control
    0.20 0.55 0.20;   % Dim-add
    0.80 0.20 0.20];  % Reject
if size(condColors,1) < nCond
    condColors = lines(nCond);
end

x = 1:nCond;

for ic = 1:nCond
    v = ratio_pool_all{ic};
    v = v(isfinite(v));
    if numel(v) < 5, continue; end

    [q1,q3,med,wlo,whi,mu] = boxStats(v);

    patch(ax, [x(ic)-boxW x(ic)+boxW x(ic)+boxW x(ic)-boxW], [q1 q1 q3 q3], condColors(ic,:), ...
        'FaceAlpha',boxFaceAlpha,'EdgeColor',condColors(ic,:), ...
        'LineWidth',LW_box,'HandleVisibility','off');

    plot(ax,[x(ic) x(ic)],[wlo q1],'-','Color',condColors(ic,:),'LineWidth',LW_box,'HandleVisibility','off');
    plot(ax,[x(ic) x(ic)],[q3 whi],'-','Color',condColors(ic,:),'LineWidth',LW_box,'HandleVisibility','off');
    plot(ax,[x(ic)-0.10 x(ic)+0.10],[wlo wlo],'-','Color',condColors(ic,:),'LineWidth',LW_box,'HandleVisibility','off');
    plot(ax,[x(ic)-0.10 x(ic)+0.10],[whi whi],'-','Color',condColors(ic,:),'LineWidth',LW_box,'HandleVisibility','off');

    plot(ax,[x(ic)-boxW x(ic)+boxW],[med med],'-','Color',condColors(ic,:), ...
        'LineWidth',LW_med,'HandleVisibility','off');

    plot(ax,x(ic),mu,'o','MarkerSize',7.5,'MarkerFaceColor','w', ...
        'MarkerEdgeColor',condColors(ic,:),'LineWidth',LW_edge,'HandleVisibility','off');
end

styleAxes(ax, fontTick, LW_axis);
ax.XLim = [0.5 nCond+0.5];
ax.XTick = x;
ax.XTickLabel = condNames;
xtickangle(ax,0);

ylabel(ax,'|E[h \mid h<0]| / E[h \mid h>0]','FontSize',fontAxis,'FontWeight','bold');
%title(ax,'Ratio pooled over S (and A)','FontSize',fontTitle,'FontWeight','bold');
ax.YGrid = 'on';
ax.GridAlpha = 0.06;

% optional: clip y to avoid showing extreme tail (if you want)
% ylim(ax,[0 5]);

% optional export:
% exportgraphics(fRatioByCond, fullfile(outdir,'Figure3_RatioNegPos_pooledOverS_andA_BoxOnly.pdf'), 'ContentType','vector');





%% Figure 3 Supplement Figure - Time Point Plot

% Window-section method using the shortest common usable steps across subjects
% and computing the SAME quantity in Early / Middle / Late stages:
%
%   ratio = |E[h | h<0]| / E[h | h>0]
%
% Assumptions:
%   For each SRes(iS).muPosD{ia,ic} and muNegD{ia,ic}, the stored data are either:
%     (1) cell array, one vector per subject
%     (2) numeric matrix, rows = subjects, cols = steps
%
% If instead they are already pooled flat vectors with no subject identity,
% this stage-wise subject-based method is not recoverable from those fields.

condNames = {'Control','Dim-Add','Rejection'};
stageNames = {'Early','Middle','Late'};

Svals   = arrayfun(@(z) z.S, SRes);
nS      = numel(Svals);

A       = SRes(1).A_list(:);
nA      = numel(A);
[A_sorted, ix] = sort(A, 'ascend');
Alabels_sorted = compose('%.3g', A_sorted);
xA = 1:nA;

nCond = numel(condNames);
nStage = numel(stageNames);

condColors = [ ...
    0.20 0.20 0.20;   % Control
    0.20 0.55 0.20;   % Dim-add
    0.80 0.20 0.20];  % Reject
if size(condColors,1) < nCond
    condColors = lines(nCond);
end

% --- typography ---
fontTitle = 18;
fontAxis  = 18;
fontTick  = 15;
fontLeg   = 15;

% --- line widths ---
LW_axis = 1.4;
LW_box  = 1.2;
LW_med  = 2.3;
LW_edge = 1.1;

% --- alpha settings ---
boxFaceAlpha = 0.12;

% --- spacing ---
offsets = linspace(-0.25, 0.25, nCond);
boxW = 0.08;

% PART 1: Build stage-wise ratio distributions pooled over S
% For each A and condition, compute one distribution per stage.
%
% ratio_stage_pool{ia, ic, istage} contains subject-level window ratios
% pooled over S, after truncating subjects to the shortest common usable length
% within that (A, condition, S) cell.

ratio_stage_pool = cell(nA, nCond, nStage);

for ia_sorted = 1:nA
    ia0 = ix(ia_sorted);   % original A index
    for ic = 1:nCond

        % Collect stage-wise subject ratios pooled over all S
        tmpStage = cell(1, nStage);
        for ist = 1:nStage
            tmpStage{ist} = [];
        end

        for iS = 1:nS
            posRaw = SRes(iS).muPosD{ia0, ic};
            negRaw = SRes(iS).muNegD{ia0, ic};

            % Convert raw storage into subject-wise vectors
            posSubj = toSubjectVectors(posRaw);
            negSubj = toSubjectVectors(negRaw);

            nSub = min(numel(posSubj), numel(negSubj));
            if nSub < 1
                continue;
            end

            % usable paired length for each subject
            usableLen = nan(nSub,1);
            for isub = 1:nSub
                p = posSubj{isub};
                n = negSubj{isub};

                if isempty(p) || isempty(n)
                    usableLen(isub) = NaN;
                    continue;
                end

                m = min(numel(p), numel(n));
                p = p(1:m);
                n = n(1:m);

                ok = isfinite(p) & isfinite(n) & (p > 0) & (n < 0);
                usableLen(isub) = sum(ok);
            end

            commonLen = min(usableLen, [], 'omitnan');
            if ~isfinite(commonLen) || commonLen < 3
                continue;
            end
            commonLen = floor(commonLen);

            % define early/middle/late windows on the common length
            edges = round(linspace(0, commonLen, 4));
            % windows: (1:edges(2)), (edges(2)+1:edges(3)), (edges(3)+1:edges(4))
            if any(diff(edges) < 1)
                continue;
            end

            % compute one ratio per subject per stage
            for isub = 1:nSub
                p = posSubj{isub};
                n = negSubj{isub};

                m = min(numel(p), numel(n));
                p = p(1:m);
                n = n(1:m);

                ok = isfinite(p) & isfinite(n) & (p > 0) & (n < 0);
                p = p(ok);
                n = n(ok);

                if numel(p) < commonLen || numel(n) < commonLen
                    continue;
                end

                p = p(1:commonLen);
                n = n(1:commonLen);

                for ist = 1:nStage
                    idxWin = (edges(ist)+1):edges(ist+1);

                    pW = p(idxWin);
                    nW = n(idxWin);

                    if isempty(pW) || isempty(nW)
                        continue;
                    end

                    muPos = mean(pW, 'omitnan');
                    muNeg = mean(abs(nW), 'omitnan');

                    if ~isfinite(muPos) || ~isfinite(muNeg) || muPos <= 0
                        continue;
                    end

                    r = muNeg / muPos;
                    if isfinite(r)
                        tmpStage{ist} = [tmpStage{ist}; r]; %#ok<AGROW>
                    end
                end
            end
        end

        for ist = 1:nStage
            ratio_stage_pool{ia_sorted, ic, ist} = tmpStage{ist};
        end
    end
end

% ============================================================
% FIGURE 1:
% 3 panels (Early / Middle / Late), x-axis = A
% distributions pooled over S
% ============================================================
fRatioByStageA = figure('Units','inches','Position',[1 1 14.5 4.8], ...
    'Color','w','Renderer','painters');

tiledlayout(1, nStage, 'TileSpacing','compact', 'Padding','compact');

for ist = 1:nStage
    ax = nexttile; hold(ax,'on');

    for ia = 1:nA
        for ic = 1:nCond
            v = ratio_stage_pool{ia, ic, ist};
            v = v(isfinite(v));
            if numel(v) < 3
                continue;
            end

            x0 = ia + offsets(ic);
            [q1,q3,med,wlo,whi,mu] = boxStats(v);

            patch(ax, [x0-boxW x0+boxW x0+boxW x0-boxW], [q1 q1 q3 q3], condColors(ic,:), ...
                'FaceAlpha', boxFaceAlpha, ...
                'EdgeColor', condColors(ic,:), ...
                'LineWidth', LW_box, ...
                'HandleVisibility', 'off');

            plot(ax, [x0 x0], [wlo q1], '-', 'Color', condColors(ic,:), ...
                'LineWidth', LW_box, 'HandleVisibility', 'off');
            plot(ax, [x0 x0], [q3 whi], '-', 'Color', condColors(ic,:), ...
                'LineWidth', LW_box, 'HandleVisibility', 'off');
            plot(ax, [x0-0.04 x0+0.04], [wlo wlo], '-', 'Color', condColors(ic,:), ...
                'LineWidth', LW_box, 'HandleVisibility', 'off');
            plot(ax, [x0-0.04 x0+0.04], [whi whi], '-', 'Color', condColors(ic,:), ...
                'LineWidth', LW_box, 'HandleVisibility', 'off');

            plot(ax, [x0-boxW x0+boxW], [med med], '-', ...
                'Color', condColors(ic,:), ...
                'LineWidth', LW_med, ...
                'HandleVisibility', 'off');

            plot(ax, x0, mu, 'o', ...
                'MarkerSize', 6.5, ...
                'MarkerFaceColor', 'w', ...
                'MarkerEdgeColor', condColors(ic,:), ...
                'LineWidth', LW_edge, ...
                'HandleVisibility', 'off');
        end
    end

    styleAxes(ax, fontTick, LW_axis);
    ax.XLim = [0.5 nA+0.5];
    ax.XTick = 1:nA;
    ax.XTickLabel = Alabels_sorted;
    ylim([0,1.1]);
    xlabel(ax, 'A', 'FontSize', fontAxis, 'FontWeight', 'bold');
    ylabel(ax, 'h^- / h^+', 'FontSize', fontAxis, 'FontWeight', 'bold');
    title(ax, stageNames{ist}, 'FontSize', fontTitle, 'FontWeight', 'bold');
    ax.YGrid = 'on';
    ax.GridAlpha = 0.06;
end

lgHandles = gobjects(nCond,1);
for ic = 1:nCond
    lgHandles(ic) = plot(nan,nan,'-','Color',condColors(ic,:),'LineWidth',3);
end
leg = legend(lgHandles, condNames, 'Box','off', 'Location','southoutside', 'NumColumns',nCond);
leg.FontSize = fontLeg;

% optional export:
% exportgraphics(fRatioByStageA, fullfile(outdir,'Figure3_Ratio_EarlyMidLate_byA.pdf'), 'ContentType','vector');

% ============================================================
% PART 2:
% Pool across A and S, but keep Early / Middle / Late separate
% ratio_stage_all{ic, istage} = all subject-level window ratios
% ============================================================
ratio_stage_all = cell(nCond, nStage);

for ic = 1:nCond
    for ist = 1:nStage
        v = [];
        for ia = 1:nA
            tmp = ratio_stage_pool{ia, ic, ist};
            tmp = tmp(isfinite(tmp));
            v = [v; tmp(:)]; %#ok<AGROW>
        end
        ratio_stage_all{ic, ist} = v;
    end
end

% ============================================================
% FIGURE 2:
% 3 panels (Early / Middle / Late), each panel shows 1 box per condition
% pooled across both A and S
% ============================================================
fRatioByStageCond = figure('Units','inches','Position',[1 1 12.5 4.4], ...
    'Color','w','Renderer','painters');

tiledlayout(1, nStage, 'TileSpacing','compact', 'Padding','compact');

boxW2 = 0.22;
xC = 1:nCond;

for ist = 1:nStage
    ax = nexttile; hold(ax,'on');

    for ic = 1:nCond
        v = ratio_stage_all{ic, ist};
        v = v(isfinite(v));
        if numel(v) < 3
            continue;
        end

        [q1,q3,med,wlo,whi,mu] = boxStats(v);

        patch(ax, [xC(ic)-boxW2 xC(ic)+boxW2 xC(ic)+boxW2 xC(ic)-boxW2], [q1 q1 q3 q3], condColors(ic,:), ...
            'FaceAlpha', boxFaceAlpha, ...
            'EdgeColor', condColors(ic,:), ...
            'LineWidth', LW_box, ...
            'HandleVisibility', 'off');

        plot(ax, [xC(ic) xC(ic)], [wlo q1], '-', 'Color', condColors(ic,:), ...
            'LineWidth', LW_box, 'HandleVisibility', 'off');
        plot(ax, [xC(ic) xC(ic)], [q3 whi], '-', 'Color', condColors(ic,:), ...
            'LineWidth', LW_box, 'HandleVisibility', 'off');
        plot(ax, [xC(ic)-0.10 xC(ic)+0.10], [wlo wlo], '-', 'Color', condColors(ic,:), ...
            'LineWidth', LW_box, 'HandleVisibility', 'off');
        plot(ax, [xC(ic)-0.10 xC(ic)+0.10], [whi whi], '-', 'Color', condColors(ic,:), ...
            'LineWidth', LW_box, 'HandleVisibility', 'off');

        plot(ax, [xC(ic)-boxW2 xC(ic)+boxW2], [med med], '-', ...
            'Color', condColors(ic,:), ...
            'LineWidth', LW_med, ...
            'HandleVisibility', 'off');

        plot(ax, xC(ic), mu, 'o', ...
            'MarkerSize', 7.5, ...
            'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor', condColors(ic,:), ...
            'LineWidth', LW_edge, ...
            'HandleVisibility', 'off');
    end

    styleAxes(ax, fontTick, LW_axis);
    ax.XLim = [0.5 nCond+0.5];
    ax.XTick = xC;
    ax.XTickLabel = condNames;
    ylim([0,1.1]);
    ylabel(ax, 'h^- / h^+', 'FontSize', fontAxis, 'FontWeight', 'bold');
    title(ax, stageNames{ist}, 'FontSize', fontTitle, 'FontWeight', 'bold');
    ax.YGrid = 'on';
    ax.GridAlpha = 0.06;
end


%% ========================= local helpers =========================

function subjCell = toSubjectVectors(raw)
% Convert storage into a cell array of subject vectors.
%
% Supported input:
%   1) raw is a cell array, one element per subject
%   2) raw is a numeric matrix, rows = subjects, cols = steps
%   3) raw is a numeric vector -> treated as one subject
%
% Output:
%   subjCell{isub} = row vector of steps for subject isub

    if iscell(raw)
        subjCell = raw(:);
        for i = 1:numel(subjCell)
            v = subjCell{i};
            if isempty(v)
                subjCell{i} = [];
            else
                subjCell{i} = v(:)';
            end
        end

    elseif isnumeric(raw)
        if isvector(raw)
            subjCell = {raw(:)'};
        else
            nSub = size(raw,1);
            subjCell = cell(nSub,1);
            for i = 1:nSub
                v = raw(i,:);
                v = v(isfinite(v) | isnan(v)); %#ok<NASGU>
                subjCell{i} = raw(i,:);
            end
        end

    else
        error('Unsupported data type in muPosD/muNegD. Need cell or numeric array.');
    end
end

function styleAxes(ax, fontTick, LW_axis)
    ax.Box='off';
    ax.TickDir='out';
    ax.LineWidth=LW_axis;
    ax.FontName='Arial';
    ax.FontSize=fontTick;
end

function [q1,q3,med,wlo,whi,mu] = boxStats(v)
    v = v(isfinite(v));
    q1  = prctile(v,25);
    q3  = prctile(v,75);
    med = median(v,'omitnan');
    mu  = mean(v,'omitnan');
    iqrV = q3 - q1;
    wlo = max(min(v), q1 - 1.5*iqrV);
    whi = min(max(v), q3 + 1.5*iqrV);
end

function violin11(ax, v, x0, faceColor, faceAlpha, maxHalfWidth)
    v = v(isfinite(v));
    if numel(v) < 5, return; end
    y = linspace(min(v), max(v), 250);
    [f, y] = ksdensity(v, y);
    f = f ./ max(f);
    w = maxHalfWidth * f;
    X = [x0 - w, fliplr(x0 + w)];
    Y = [y,      fliplr(y)];
    patch(ax, X, Y, faceColor, ...
        'FaceAlpha',faceAlpha, ...
        'EdgeColor',faceColor, ...
        'LineWidth',1.0, ...
        'HandleVisibility','off');
end
%% Stats for figure4
% ===================== Ratio pooled over ALL S and A =====================
% Computes:
%   |E[h|h<0]| / E[h|h>0]
% Pooled across S and A
% Prints stats in manuscript style

condNames = {'Control','Dim-Add','Rejection'};
nCond = numel(condNames);

% -------- collect pooled ratios ----------
ratio_all = cell(nCond,1);

for ic = 1:nCond
    v_all = [];

    for iS = 1:numel(SRes)
        nA = numel(SRes(iS).A_list);

        for ia = 1:nA
            pos = SRes(iS).muPosD{ia,ic};
            neg = SRes(iS).muNegD{ia,ic};

            pos = pos(isfinite(pos));
            neg = neg(isfinite(neg));

            n = min(numel(pos), numel(neg));
            if n < 5, continue; end

            r = abs(neg(1:n)) ./ pos(1:n);
            r = r(isfinite(r));

            v_all = [v_all; r(:)]; %#ok<AGROW>
        end
    end

    ratio_all{ic} = v_all(isfinite(v_all));
end

% -------- bootstrap CI --------
NBOOT = 1000;

mean_val = zeros(nCond,1);
ci_low   = zeros(nCond,1);
ci_high  = zeros(nCond,1);

for ic = 1:nCond
    v = ratio_all{ic};
    mean_val(ic) = mean(v,'omitnan');

    bootM = zeros(NBOOT,1);
    n = numel(v);
    for b = 1:NBOOT
        idx = randi(n,n,1);
        bootM(b) = mean(v(idx),'omitnan');
    end

    ci = prctile(bootM,[2.5 97.5]);
    ci_low(ic)  = ci(1);
    ci_high(ic) = ci(2);
end

% -------- Cohen's d (Welch pooled SD) --------
cohend_d = @(x,y) (mean(x)-mean(y)) / sqrt((var(x,1)+var(y,1))/2);

% ===================== PRINT RESULTS =====================
fprintf('\n================ Ratio: |E[h|h<0]| / E[h|h>0] (pooled S,A) ================\n\n');

for ic = 1:nCond
    fprintf('%-10s mean = %.4f, CI [%.4f, %.4f]\n', ...
        condNames{ic}, mean_val(ic), ci_low(ic), ci_high(ic));
end

fprintf('\n--- Relative to Control ---\n');

for ic = 2:nCond
    delta  = mean_val(ic) - mean_val(1);
    pct    = 100 * delta / mean_val(1);
    dval   = cohend_d(ratio_all{ic}, ratio_all{1});

    fprintf('%s: Δ = %.4f (%+.2f%%), d = %.2f\n', ...
        condNames{ic}, delta, pct, dval);
end

fprintf('\n==========================================================================\n');


% ================= Ratio pooled over S (function of A) =================
% Computes ratio = |E[h|h<0]| / E[h|h>0] pooled across S
% Prints stats per A in manuscript style

condNames = {'Control','Dim-Add','Rejection'};
nCond = numel(condNames);

Avals = SRes(1).A_list(:);
nA    = numel(Avals);

NBOOT = 1000;


fprintf('\n================ Ratio pooled over S (by A) ================\n');

for ia = 1:nA
    
    ratioA = cell(nCond,1);
    
    % ---------- pool across S ----------
    for ic = 1:nCond
        v_all = [];
        
        for iS = 1:numel(SRes)
            pos = SRes(iS).muPosD{ia,ic};
            neg = SRes(iS).muNegD{ia,ic};
            
            pos = pos(isfinite(pos));
            neg = neg(isfinite(neg));
            
            n = min(numel(pos), numel(neg));
            if n < 5, continue; end
            
            r = abs(neg(1:n)) ./ pos(1:n);
            r = r(isfinite(r));
            
            v_all = [v_all; r(:)]; %#ok<AGROW>
        end
        
        ratioA{ic} = v_all(isfinite(v_all));
    end
    
    % ---------- compute mean + CI ----------
    mean_val = zeros(nCond,1);
    ci_low   = zeros(nCond,1);
    ci_high  = zeros(nCond,1);
    
    for ic = 1:nCond
        v = ratioA{ic};
        mean_val(ic) = mean(v,'omitnan');
        
        bootM = zeros(NBOOT,1);
        n = numel(v);
        for b = 1:NBOOT
            idx = randi(n,n,1);
            bootM(b) = mean(v(idx),'omitnan');
        end
        
        ci = prctile(bootM,[2.5 97.5]);
        ci_low(ic)  = ci(1);
        ci_high(ic) = ci(2);
    end
    
    % ---------- print ----------
    fprintf('\nA = %g\n', Avals(ia));
    
    for ic = 1:nCond
        fprintf('  %-10s mean = %.4f, CI [%.4f, %.4f]\n', ...
            condNames{ic}, mean_val(ic), ci_low(ic), ci_high(ic));
    end
    
    fprintf('  Relative to Control:\n');
    for ic = 2:nCond
        delta  = mean_val(ic) - mean_val(1);
        pct    = 100 * delta / mean_val(1);
        dval   = cohend_d(ratioA{ic}, ratioA{1});
        
        fprintf('    %s: Δ = %.4f (%+.2f%%), d = %.2f\n', ...
            condNames{ic}, delta, pct, dval);
    end
end

fprintf('\n============================================================\n');


%% Tail-risk analysis (FAST): use SUBJECT-level conditional means instead of all timepoints
% Goal: quantify how Reject changes tails of:
%   X = muPosD = E[δ | δ>0] per subject (within-episode)
%   Y = muNegD = E[δ | δ<0] per subject (negative; we analyze magnitude |Y|)
% Pool across ALL A and S, and compute tail quantiles + expected shortfall with bootstrap CI.
%
% This is much faster + avoids autocorrelation/timepoint inflation.

% ---------------- settings ----------------
if ~exist('NBOOT','var') || isempty(NBOOT), NBOOT = 1000; end
if ~exist('ALPHA','var') || isempty(ALPHA), ALPHA = 0.05; end

pTail = 0.05;               % ES95 (top/bottom 5%)
qPos  = [0.95 0.99];        % upper-tail quantiles for muPosD
qNeg  = [0.95 0.99];        % upper-tail quantiles for |muNegD| (magnitude)

condNamesUse = condNames;
nCond = numel(condNamesUse);

icReject = find(strcmpi(condNamesUse,'Reject') | contains(lower(condNamesUse),'reject'), 1);
if isempty(icReject), icReject = 3; end
icBase = [1 2]; icBase = icBase(icBase <= nCond & icBase ~= icReject);

Svals = arrayfun(@(z) z.S, SRes);
nS    = numel(Svals);
A     = SRes(1).A_list(:);
nA    = numel(A);

if ~exist('matDir','var') || isempty(matDir), matDir = pwd; end

fprintf('\n==================== Tail-risk (subject-level means) ====================\n');
fprintf('Stats on muPosD=E[δ|δ>0] and |muNegD|=|E[δ|δ<0]| pooled across all A,S\n');
fprintf('NBOOT=%d (bootstrap over subjects), CI=%d%%\n\n', NBOOT, round(100*(1-ALPHA)));

% ---------------- preload metaRes per S ----------------
metaResS = cell(nS,1);
for iS = 1:nS
    matName = fullfile(matDir, sprintf('meta_all_runs_s%.1f.mat', Svals(iS)));
    tmp = load(matName,'metaRes');
    metaResS{iS} = tmp.metaRes;
end

% ---------------- collect subject-level (muPosD, muNegD) per condition ----------------
Xall = cell(nCond,1);   % muPosD per subject
Yall = cell(nCond,1);   % |muNegD| per subject

for ic = 1:nCond
    pairs = metaCells{ic};
    X = [];
    Y = [];

    for iS = 1:nS
        metaRes = metaResS{iS};

        for ia = 1:nA
            for kk = 1:size(pairs,1)
                id = pairs(kk,1); ip = pairs(kk,2);

                traj  = metaRes{id,ip}.res.traj{condIdx(ic), ia};
                D     = traj.delta;          % [nSubj x Tmax]
                steps = traj.steps_taken(:); % [nSubj x 1]
                if isempty(D) || isempty(steps), continue; end

                [nSubj, Tmax] = size(D);

                % mask beyond goal
                for s = 1:nSubj
                    Ti = steps(s);
                    if isfinite(Ti) && Ti < Tmax
                        D(s, Ti+1:end) = NaN;
                    end
                end

                % conditional means per subject
                Dpos = D; Dpos(Dpos <= 0) = NaN;
                Dneg = D; Dneg(Dneg >= 0) = NaN;

                nPos = sum(isfinite(Dpos),2);
                nNeg = sum(isfinite(Dneg),2);

                muPos = nansum(Dpos,2) ./ nPos;   % E[δ|δ>0]
                muNeg = nansum(Dneg,2) ./ nNeg;   % E[δ|δ<0] (negative)

                muPos(nPos==0) = NaN;
                muNeg(nNeg==0) = NaN;

                ok = isfinite(muPos) & isfinite(muNeg);
                X = [X; muPos(ok)];
                Y = [Y; abs(muNeg(ok))];
            end
        end
    end

    Xall{ic} = X;
    Yall{ic} = Y;

    fprintf('Cond %-10s: nSubj=%d\n', condNamesUse{ic}, numel(X));
end

% ---------------- tail stats + bootstrap CI ----------------
ciLo = 100*(ALPHA/2);
ciHi = 100*(1-ALPHA/2);

stats0 = struct();
CI = struct();

for ic = 1:nCond
    X = Xall{ic}; X = X(isfinite(X));
    Y = Yall{ic}; Y = Y(isfinite(Y));

    stats0(ic).q95_pos = quantile(X, qPos(1));
    stats0(ic).q99_pos = quantile(X, qPos(2));
    thrX = quantile(X, 1-pTail);
    stats0(ic).es95_pos = mean(X(X >= thrX), 'omitnan');

    stats0(ic).q95_negmag = quantile(Y, qNeg(1));
    stats0(ic).q99_negmag = quantile(Y, qNeg(2));
    thrY = quantile(Y, 1-pTail);
    stats0(ic).es95_negmag = mean(Y(Y >= thrY), 'omitnan');

    stats0(ic).tci = stats0(ic).es95_negmag / stats0(ic).es95_pos; % tail-compression index

    nObs = min(numel(X), numel(Y));
    % bootstrap over subject samples (independent-ish units)
    boot = nan(NBOOT, 7); % [q95+ q99+ ES95+ q95|neg| q99|neg| ES95|neg| TCI]
    for b = 1:NBOOT
        idxX = randi(numel(X), numel(X), 1);
        idxY = randi(numel(Y), numel(Y), 1);
        xb = X(idxX);
        yb = Y(idxY);

        q95p = quantile(xb, qPos(1));
        q99p = quantile(xb, qPos(2));
        thr  = quantile(xb, 1-pTail);
        esP  = mean(xb(xb >= thr), 'omitnan');

        q95n = quantile(yb, qNeg(1));
        q99n = quantile(yb, qNeg(2));
        thr  = quantile(yb, 1-pTail);
        esN  = mean(yb(yb >= thr), 'omitnan');

        boot(b,:) = [q95p q99p esP q95n q99n esN (esN/esP)];
    end
    CI(ic).mat = prctile(boot, [ciLo ciHi], 1);
end

% ---------------- print concise report ----------------
fprintf('\n--- Tail summaries (pooled A,S) ---\n');
for ic = 1:nCond
    ci = CI(ic).mat;
    fprintf('%s:\n', condNamesUse{ic});
    fprintf('  muPosD tail: q95=%.4g [%.4g,%.4g], q99=%.4g [%.4g,%.4g], ES95+=%.4g [%.4g,%.4g]\n', ...
        stats0(ic).q95_pos, ci(1,1),ci(2,1), stats0(ic).q99_pos, ci(1,2),ci(2,2), stats0(ic).es95_pos, ci(1,3),ci(2,3));
    fprintf('  |muNegD| tail: q95=%.4g [%.4g,%.4g], q99=%.4g [%.4g,%.4g], ES95|neg|=%.4g [%.4g,%.4g]\n', ...
        stats0(ic).q95_negmag, ci(1,4),ci(2,4), stats0(ic).q99_negmag, ci(1,5),ci(2,5), stats0(ic).es95_negmag, ci(1,6),ci(2,6));
    fprintf('  TCI = ES95|neg| / ES95+ : %.4g [%.4g,%.4g]\n\n', ...
        stats0(ic).tci, ci(1,7), ci(2,7));
end

% ---------------- Reject vs baselines: percent change in tail severity ----------------
icR = icReject;
fprintf('--- Reject vs baselines (percent change; negative means reduction) ---\n');
for ic = icBase
    % positive tail: reduction means smaller
    pc_q99_pos = 100*(stats0(icR).q99_pos - stats0(ic).q99_pos) / stats0(ic).q99_pos;
    pc_es_pos  = 100*(stats0(icR).es95_pos - stats0(ic).es95_pos) / stats0(ic).es95_pos;

    % negative magnitude tail: reduction means smaller
    pc_q99_neg = 100*(stats0(icR).q99_negmag - stats0(ic).q99_negmag) / stats0(ic).q99_negmag;
    pc_es_neg  = 100*(stats0(icR).es95_negmag - stats0(ic).es95_negmag) / stats0(ic).es95_negmag;

    % TCI: reduction means smaller (relatively less negative tail per positive tail)
    pc_tci     = 100*(stats0(icR).tci - stats0(ic).tci) / stats0(ic).tci;

    fprintf('%s vs %s:\n', condNamesUse{icR}, condNamesUse{ic});
    fprintf('  muPosD:   q99 %+6.2f%%,  ES95+ %+6.2f%%\n', pc_q99_pos, pc_es_pos);
    fprintf('  |muNegD|: q99 %+6.2f%%,  ES95  %+6.2f%%\n', pc_q99_neg, pc_es_neg);
    fprintf('  TCI:      %+6.2f%%\n\n', pc_tci);
end



%% ===================== Figure 4A + 4B: Bootstrapped tradeoff slope =====================
% 4A: slope vs S (pooled over A) — 3 panels (one per condition)
% 4B: slope vs A (pooled over S) — 3 panels (one per condition)
%
% IMPORTANT: loads BOTH datasets per S:
%   - meta_all_runs_sX.mat                      (negative-asym)
%   - meta_all_runs_sX_positive_asym.mat        (positive-asym; A relabeled as negative)
%   Drop A_pos==1 (would map to A=-1, redundant).
%
% Assumes:
%   - SRes exists (for Svals) and metaCells exists in workspace (do NOT rename)
%   - violin1(ax,...) is defined globally (do NOT redefine here)
%   - matDir exists (else uses pwd)
USE_POS_ASYM = false;   % <-- set false to use ONLY negative-asym data
% -------------------- condition defs (keep your names) --------------------
ic_ctrl = 1; ic_dim = 2; ic_rej = 3;
condNames = {'Control','Dim-add','Reject'};
condIdx   = [ic_ctrl, ic_dim, ic_rej];
nCond = numel(condIdx);

% -------------------- settings / aesthetics --------------------
NBOOT = 10;
USE_SPEARMAN = false;

if ~exist('matDir','var') || isempty(matDir)
    matDir = pwd;
end

Svals   = arrayfun(@(z) z.S, SRes);
nS      = numel(Svals);
Slabels = arrayfun(@(s) sprintf('%.1f',s), Svals, 'uni', 0);

S_colors = lines(nS);

% ----- Y axis control -----
useCustomYLim = false;
customYLim    = [-0.25 0.25];

% ----- Figure size -----
figWidth  = 10.5;
figHeight = 3.6;

% ----- Font sizes -----
fontAxis  = 15;
fontTick  = 14;
fontTitle = 15;

% ----- Line widths -----
LW_axis = 1.2;
LW_box  = 1.4;
LW_med  = 2.0;
LW_edge = 1.1;

% ----- Transparency -----
violinAlpha  = 0.28;
boxFaceAlpha = 0.18;

% -------------------- metaCells source (do NOT rename metaCells) --------------------
% Use existing metaCells in workspace. If missing, try pulling from SRes(1) or SRes(1).out meta.
if ~exist('metaCells','var') || isempty(metaCells)
    if isfield(SRes(1),'metaCells')
        metaCells = SRes(1).metaCells;
    else
        error('metaCells not found in workspace or SRes. Please define metaCells first.');
    end
end

% ===================== Helper: load BOTH datasets + A mapping =====================
% Returns: metaRes_neg, metaRes_pos, A_all, A_src, ia_src
%   A_src(ia)=0 => use metaRes_neg;  A_src(ia)=1 => use metaRes_pos
%   ia_src(ia) gives the A-index inside that metaRes_* (pos uses keepPos indices)
%
% NOTE: uses Svals(iS) and matDir
load_both_for_S = @(Sval) local_loadBoth(matDir, Sval);

% slopeBootS{ic, iS} = [NBOOT x 1]
slopeBootS = cell(nCond, nS);

for ic = 1:nCond
    pairs = metaCells{ic}; % [nPairs x 2] of (id, ip)

    for iS = 1:nS
        % ----- load BOTH datasets for this S -----
        %[metaRes_neg, metaRes_pos, A_all, A_src, ia_src] = load_both_for_S(Svals(iS));
        if USE_POS_ASYM
            [metaRes_neg, metaRes_pos, A_all, A_src, ia_src] = local_loadBoth(matDir, Svals(iS));
        else
            tmpN = load(fullfile(matDir, sprintf('meta_all_runs_s%.1f.mat', Svals(iS))), 'metaRes','A_list');
            metaRes_neg = tmpN.metaRes;
            metaRes_pos = []; %#ok<NASGU>
            A_all     = tmpN.A_list(:);
            A_src     = zeros(numel(A_all),1);      % all from neg file
            ia_src    = (1:numel(A_all))';          % direct indices
        end


        nA_all = numel(A_all);

        % pool (X,Y) across ALL A (within this S)
        X = [];
        Y = [];

        for kk = 1:size(pairs,1)
            id = pairs(kk,1);
            ip = pairs(kk,2);

            for ia = 1:nA_all
                src = A_src(ia);
                ia0 = ia_src(ia);

                if src == 0
                    metaRes_use = metaRes_neg;
                else
                    metaRes_use = metaRes_pos;
                end

                % --- MINIMAL GUARD: skip invalid meta-grid indices for this file ---
                if id > size(metaRes_use,1) || ip > size(metaRes_use,2)
                    continue;
                end


                traj  = metaRes_use{id, ip}.res.traj{condIdx(ic), ia0};
                D     = traj.delta;
                steps = traj.steps_taken(:);

                [nSubj, Tmax] = size(D);

                % mask after each subject's termination
                for s = 1:nSubj
                    Ti = steps(s);
                    if Ti < Tmax
                        D(s, Ti+1:end) = NaN;
                    end
                end

                Dpos = D; Dpos(Dpos <= 0) = NaN;
                Dneg = D; Dneg(Dneg >= 0) = NaN;

                nPos = sum(isfinite(Dpos),2);
                nNeg = sum(isfinite(Dneg),2);

                muPos = nansum(Dpos,2) ./ nPos;
                muNeg = nansum(Dneg,2) ./ nNeg;

                muPos(nPos==0) = NaN;
                muNeg(nNeg==0) = NaN;

                ok = isfinite(muPos) & isfinite(muNeg);
                X = [X; muPos(ok)];
                Y = [Y; abs(muNeg(ok))];
            end
        end

        nObs = numel(X);
        slopes = nan(NBOOT,1);

        if nObs < 2
            warning('No valid obs for ic=%d, S=%.1f (nObs=%d). slopeBootS cell will be NaN.', ic, Svals(iS), nObs);
        else
            for b = 1:NBOOT
                idx = randi(nObs, nObs, 1);
                xb = X(idx);
                yb = Y(idx);

                if USE_SPEARMAN
                    slopes(b) = corr(xb, yb, 'Type','Spearman', 'Rows','complete');
                else
                    p = polyfit(xb, yb, 1);
                    slopes(b) = p(1);
                end
            end
        end

        slopeBootS{ic,iS} = slopes;
    end
end

% Plotting -------------------- harmonize y-lims across panels (ROBUST) ----------------

fontMain = 18;
fontAxis = 18;
fontTick = 14;

LW_axis  = 2;
LW_box   = 2;
LW_med   = 2.0;
LW_edge  = 2;


% -------------------- sort S ascending and reorder --------------------
Svals    = arrayfun(@(z) z.S, SRes);
nS       = numel(Svals);
[S_sorted, ixS] = sort(Svals,'ascend');               % ixS: sorted-position -> original S index
Slabels_sorted  = arrayfun(@(s) sprintf('%.1f',s), S_sorted, 'uni', 0);

S_colors_sorted = S_colors(ixS,:);

% -------------------- harmonize y-lims across panels (ROBUST) ----------------
yAll = [];
for ic = 1:nCond
    for jS = 1:nS
        iS0 = ixS(jS);
        v = slopeBootS{ic, iS0};
        if isempty(v), continue; end
        v = v(isfinite(v));
        yAll = [yAll; v(:)];
    end
end
yAll = yAll(isfinite(yAll));

if isempty(yAll)
    autoYLim = [-1 1];
else
    autoYLim = prctile(yAll,[1 99]);
    if numel(autoYLim)~=2 || any(~isfinite(autoYLim))
        autoYLim = [min(yAll) max(yAll)];
    end
    if autoYLim(2) <= autoYLim(1)
        m = autoYLim(1);
        autoYLim = [m-1e-3, m+1e-3];
    else
        pad = 0.08 * (autoYLim(2)-autoYLim(1));
        autoYLim = [autoYLim(1)-pad, autoYLim(2)+pad];
    end
end

if useCustomYLim
    ylims = double(customYLim(:)');
else
    ylims = autoYLim;
end
if numel(ylims)~=2 || any(~isfinite(ylims))
    ylims = autoYLim;
end
if ylims(2) <= ylims(1)
    ylims = fliplr(ylims);
end
if ylims(2) <= ylims(1)
    m = ylims(1);
    ylims = [m-1e-3, m+1e-3];
end

% -------------------- plot: 3 panels, x = S (ASCENDING) ----------------------
fA2 = figure('Units','inches','Position',[1 1 figWidth figHeight], ...
    'Color','w','Renderer','painters');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

for ic = 1:nCond
    ax = nexttile; hold(ax,'on');

    for jS = 1:nS
        iS0 = ixS(jS);                 % original index into slopeBootS / S_colors
        v = slopeBootS{ic, iS0};
        if isempty(v), continue; end
        v = v(isfinite(v));
        if isempty(v), continue; end

        % violin (color by sorted S)
        violin1(ax, v, jS, S_colors_sorted(jS,:), violinAlpha);

        % box overlay
        q1  = prctile(v,25);
        q3  = prctile(v,75);
        med = median(v,'omitnan');
        iqrV = q3 - q1;

        vmin = min(v);
        vmax = max(v);
        wlo = max(vmin, q1 - 1.5*iqrV);
        whi = min(vmax, q3 + 1.5*iqrV);

        boxW = 0.22;
        patch(ax, [jS-boxW jS+boxW jS+boxW jS-boxW], [q1 q1 q3 q3], S_colors_sorted(jS,:), ...
            'FaceAlpha',boxFaceAlpha,'EdgeColor',S_colors_sorted(jS,:), 'LineWidth',LW_box, ...
            'HandleVisibility','off');

        plot(ax, [jS jS], [wlo q1], '-', 'Color',S_colors_sorted(jS,:), 'LineWidth',LW_box, 'HandleVisibility','off');
        plot(ax, [jS jS], [q3 whi], '-', 'Color',S_colors_sorted(jS,:), 'LineWidth',LW_box, 'HandleVisibility','off');
        plot(ax, [jS-0.10 jS+0.10], [wlo wlo], '-', 'Color',S_colors_sorted(jS,:), 'LineWidth',LW_box, 'HandleVisibility','off');
        plot(ax, [jS-0.10 jS+0.10], [whi whi], '-', 'Color',S_colors_sorted(jS,:), 'LineWidth',LW_box, 'HandleVisibility','off');

        % median (bold)
        plot(ax, [jS-boxW jS+boxW], [med med], '-', 'Color',S_colors_sorted(jS,:), ...
            'LineWidth',LW_med, 'HandleVisibility','off');

        % mean (dot)
        mu = mean(v,'omitnan');
        plot(ax, jS, mu, 'o', 'MarkerSize',6.5, ...
            'MarkerFaceColor','w','MarkerEdgeColor',S_colors_sorted(jS,:), ...
            'LineWidth',LW_edge, 'HandleVisibility','off');
    end

    yline(ax,0,':','LineWidth',1.5,'Color',[0 0 0],'Alpha',0.25,'HandleVisibility','off');

    ax.Box='off';
    ax.TickDir='out';
    ax.LineWidth=LW_axis;
    ax.FontName='Arial';
    ax.FontSize=fontTick;

    ax.XLim = [0.5 nS+0.5];
    ax.XTick = 1:nS;
    ax.XTickLabel = Slabels_sorted;
    ax.YLim = ylims;

    xlabel(ax,'S','FontName','Arial','FontSize',fontAxis,'FontWeight','bold');
    if ic == 1
        ylabel(ax,'Tradeoff slope', 'FontName','Arial','FontSize',fontAxis,'FontWeight','bold');
    end
    title(ax, sprintf('%s', condNames{ic}), 'FontWeight','bold','FontSize',fontMain);

    ax.YGrid = 'on';
    ax.GridAlpha = 0.08;
    ax.GridLineStyle = '-';
end



%% ===================== Figure 4B: slope vs A (pooled over S) =====================
% Computes slopeBootA{ic,ia} by pooling across ALL S.
% Supports BOTH datasets per S:
%   meta_all_runs_sX.mat                 (negative-asym default; A > 0)
%   meta_all_runs_sX_positive_asym.mat   (positive-asym; we relabel as A < 0)


NBOOT = 10;
USE_SPEARMAN = false;

if ~exist('matDir','var') || isempty(matDir)
    matDir = pwd;
end

% Build a GLOBAL A axis from SRes(1).A_list (already includes +/- A from preprocessing)
A       = SRes(1).A_list(:);
nA      = numel(A);
Alabels = compose('%.3g',A);


violinAlpha  = 0.28;
boxFaceAlpha = 0.18;

A_colors = lines(nA);

% slopeBootA{ic, ia} = [NBOOT x 1]
slopeBootA = cell(nCond, nA);

for ic = 1:nCond
    pairs = metaCells{ic};

    for ia = 1:nA
        X = [];
        Y = [];

        aTarget = A(ia);

        for iS = 1:nS
            % ----- load BOTH datasets for this S -----
            %[metaRes_neg, metaRes_pos, A_all_S, A_src_S, ia_src_S] = local_loadBoth(matDir, Svals(iS));

            if USE_POS_ASYM
                [metaRes_neg, metaRes_pos, A_all, A_src, ia_src] = local_loadBoth(matDir, Svals(iS));
            else
                tmpN = load(fullfile(matDir, sprintf('meta_all_runs_s%.1f.mat', Svals(iS))), 'metaRes','A_list');
                metaRes_neg = tmpN.metaRes;
                metaRes_pos = []; %#ok<NASGU>
                A_all     = tmpN.A_list(:);
                A_src     = zeros(numel(A_all),1);      % all from neg file
                ia_src    = (1:numel(A_all))';          % direct indices
            end

            % find this A on this S's axis
            idxMatch = find(abs(A_all - aTarget) < 1e-12, 1);
            if isempty(idxMatch)
                continue;
            end

            src = A_src_S(idxMatch);      % 0=neg file, 1=pos file
            ia0 = ia_src_S(idxMatch);     % index within that file's A_list

            if src == 0
                metaRes_use = metaRes_neg;
            else
                metaRes_use = metaRes_pos;
            end

            for kk = 1:size(pairs,1)
                id = pairs(kk,1);
                ip = pairs(kk,2);

                % guard (MUST be after id/ip are defined)
                if id > size(metaRes_use,1) || ip > size(metaRes_use,2)
                    continue;
                end

                traj  = metaRes_use{id, ip}.res.traj{condIdx(ic), ia0};
                D     = traj.delta;
                steps = traj.steps_taken(:);

                if isempty(D) || isempty(steps)
                    continue;
                end

                [nSubj, Tmax] = size(D);

                % mask after each subject's termination
                for s = 1:nSubj
                    Ti = steps(s);
                    if isfinite(Ti) && Ti < Tmax
                        D(s, Ti+1:end) = NaN;
                    end
                end

                % subject-level conditional means
                Dpos = D; Dpos(Dpos <= 0) = NaN;
                Dneg = D; Dneg(Dneg >= 0) = NaN;

                nPos = sum(isfinite(Dpos),2);
                nNeg = sum(isfinite(Dneg),2);

                muPos = nansum(Dpos,2) ./ nPos;   % E[δ|δ>0]
                muNeg = nansum(Dneg,2) ./ nNeg;   % E[δ|δ<0] (negative)

                muPos(nPos==0) = NaN;
                muNeg(nNeg==0) = NaN;

                ok = isfinite(muPos) & isfinite(muNeg);
                X = [X; muPos(ok)];
                Y = [Y; abs(muNeg(ok))];
            end
        end

        nObs = numel(X);
        slopes = nan(NBOOT,1);

        if nObs < 2
            warning('No valid obs for ic=%d, A=%g (nObs=%d). slopeBootA cell will be NaN.', ic, A(ia), nObs);
        else
            for b = 1:NBOOT
                idx = randi(nObs, nObs, 1);
                xb = X(idx);
                yb = Y(idx);

                if USE_SPEARMAN
                    slopes(b) = corr(xb, yb, 'Type','Spearman', 'Rows','complete');
                else
                    p = polyfit(xb, yb, 1);
                    slopes(b) = p(1);
                end
            end
        end

        slopeBootA{ic, ia} = slopes;
    end
end

% Plotting
fontMain = 18;
fontAxis = 18;
fontTick = 14;

LW_axis  = 2;
LW_box   = 2;
LW_med   = 2.0;
LW_edge  = 2;
% -------------------- sort A ascending and reorder mapping --------------------
A       = SRes(1).A_list(:);
nA      = numel(A);
[A_sorted, ix] = sort(A,'ascend');                 % ix: sorted-position -> original A index
Alabels_sorted = compose('%.3g', A_sorted);

% make sure A_colors exists (defined earlier as lines(nA))
A_colors_sorted = A_colors(ix,:);

% -------------------- harmonize y-lims across panels ------------------------
yAll = [];
for ic = 1:nCond
    for ia = 1:nA
        v = slopeBootA{ic, ia};
        if isempty(v), continue; end
        v = v(isfinite(v));
        yAll = [yAll; v(:)];
    end
end
yAll = yAll(isfinite(yAll));

if isempty(yAll)
    ylimsB = [-1 1];
else
    ylimsB = prctile(yAll,[1 99]);
    pad = 0.08 * (ylimsB(2)-ylimsB(1));
    ylimsB = [ylimsB(1)-pad, ylimsB(2)+pad];
end

% -------------------- plot: 3 panels, x = A (ASCENDING) ---------------------
fA2A = figure('Units','inches','Position',[1 1 11.8 3.6], ...
    'Color','w','Renderer','painters');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

for ic = 1:nCond
    ax = nexttile; hold(ax,'on');

    for ia = 1:nA
        ia0 = ix(ia);                       % original index in slopeBootA / unsorted A
        v = slopeBootA{ic, ia0};
        if isempty(v), continue; end
        v = v(isfinite(v));
        if isempty(v), continue; end

        % violin (color by sorted A)
        violin1(ax, v, ia, A_colors_sorted(ia,:), violinAlpha);

        % box overlay
        q1  = prctile(v,25);
        q3  = prctile(v,75);
        med = median(v,'omitnan');
        iqrV = q3 - q1;

        wlo = max(min(v), q1 - 1.5*iqrV);
        whi = min(max(v), q3 + 1.5*iqrV);

        boxW = 0.22;
        patch(ax, [ia-boxW ia+boxW ia+boxW ia-boxW], [q1 q1 q3 q3], A_colors_sorted(ia,:), ...
            'FaceAlpha',boxFaceAlpha,'EdgeColor',A_colors_sorted(ia,:),'LineWidth',LW_box, ...
            'HandleVisibility','off');

        plot(ax, [ia ia], [wlo q1], '-', 'Color',A_colors_sorted(ia,:), 'LineWidth',LW_box, 'HandleVisibility','off');
        plot(ax, [ia ia], [q3 whi], '-', 'Color',A_colors_sorted(ia,:), 'LineWidth',LW_box, 'HandleVisibility','off');
        plot(ax, [ia-0.10 ia+0.10], [wlo wlo], '-', 'Color',A_colors_sorted(ia,:), 'LineWidth',LW_box, 'HandleVisibility','off');
        plot(ax, [ia-0.10 ia+0.10], [whi whi], '-', 'Color',A_colors_sorted(ia,:), 'LineWidth',LW_box, 'HandleVisibility','off');

        % median (bold)
        plot(ax, [ia-boxW ia+boxW], [med med], '-', 'Color',A_colors_sorted(ia,:), ...
            'LineWidth',LW_med, 'HandleVisibility','off');

        % mean (dot)
        mu = mean(v,'omitnan');
        plot(ax, ia, mu, 'o', 'MarkerSize',6.5, ...
            'MarkerFaceColor','w','MarkerEdgeColor',A_colors_sorted(ia,:), ...
            'LineWidth',LW_edge, 'HandleVisibility','off');
    end

    yline(ax,0,':','LineWidth',1.5,'Color',[0 0 0],'Alpha',0.25,'HandleVisibility','off');

    ax.Box='off';
    ax.TickDir='out';
    ax.LineWidth=LW_axis;
    ax.FontName='Arial';
    ax.FontSize=fontTick;

    ax.XLim = [0.5 nA+0.5];
    ax.XTick = 1:nA;
    ax.XTickLabel = Alabels_sorted;
    ax.YLim = ylimsB;

    xlabel(ax,'A','FontName','Arial','FontSize',fontAxis);
    if ic == 1
        ylabel(ax,'Tradeoff slope', ...
            'FontName','Arial','FontSize',fontAxis);
    end
    title(ax, sprintf('%s (pooled over S)', condNames{ic}), 'FontWeight','normal','FontSize',fontMain);

    ax.YGrid = 'on';
    ax.GridAlpha = 0.08;
    ax.GridLineStyle = '-';
end

% ========================= local helper =========================
function [metaRes_neg, metaRes_pos, A_all, A_src, ia_src] = local_loadBoth(matDir, Sval)

    matName_neg = fullfile(matDir, sprintf('meta_all_runs_s%.1f.mat', Sval));
    matName_pos = fullfile(matDir, sprintf('meta_all_runs_s%.1f_positive_asym.mat', Sval));

    tmpN = load(matName_neg, 'metaRes', 'A_list');
    tmpP = load(matName_pos, 'metaRes', 'A_list');

    metaRes_neg = tmpN.metaRes;
    A_neg       = tmpN.A_list(:);

    metaRes_pos = tmpP.metaRes;
    A_pos       = tmpP.A_list(:);

    % relabel pos-asym A as negative A; drop A_pos==1 (would become -1, redundant)
    keepPos   = (A_pos ~= 1);
    A_pos_sgn = -A_pos(keepPos);

    A_all  = [A_neg; A_pos_sgn];
    nA_neg = numel(A_neg);

    % mapping: for each entry of A_all, where to read it from
    A_src  = [zeros(nA_neg,1); ones(numel(A_pos_sgn),1)]; % 0=neg file, 1=pos file

    % ia_src: index into A_list of the corresponding file
    % for pos-file, we need the ORIGINAL indices (pre-keepPos filtering)
    ia_src = [ (1:nA_neg)'; find(keepPos) ];
end


% -------------------- local function: simple violin -------------------------
function violin1(ax, v, x0, faceColor, faceAlpha)
    v = v(isfinite(v));
    if numel(v) < 5, return; end

    nGrid = 250;
    y = linspace(min(v), max(v), nGrid);
    [f, y] = ksdensity(v, y);

    f = f / max(f);
    w = 0.33 * f;

    X = [x0 - w, fliplr(x0 + w)];
    Y = [y,      fliplr(y)];

    patch(ax, X, Y, faceColor, ...
        'FaceAlpha',faceAlpha, ...
        'EdgeColor',faceColor, ...
        'LineWidth',1.0, ...
        'HandleVisibility','off');
end










% %% Stats pack for paper-ready figure descriptions (Δ means, % change, Cohen d, bootstrap CI, regression)
% 
% % ---------- user knobs ----------
% NBOOT = 1000;                 % bootstrap draws for CIs
% ALPHA = 0.05;                 % 95% CI
% USE_LME = false;              % set true if you want fitlme (needs Statistics Toolbox)
% 
% % ---------- factors ----------
% Svals = arrayfun(@(z) z.S, SRes);
% nS = numel(Svals);
% A = SRes(1).A_list(:);
% nA = numel(A);
% nCond = size(SRes(1).muR,2);
% 
% if ~exist('condNames','var') || numel(condNames) ~= nCond
%     condNames = arrayfun(@(k) sprintf('Cond %d',k), 1:nCond, 'uni', 0);
% end
% 
% % ========== helper: build long table of subject-level values ==========
% % field is e.g., 'muR','muD','asyR','asyD','muPosD','muNegD'
% % transform controls e.g. abs for negative magnitude
% makeLong = @(field, transformFn) local_makeLong(SRes, field, transformFn, condNames);
% 
% % ========== helper: summarize by condition (pooled over S and A) ==========
% % prints mean ± bootCI, and Δ vs Control with CI, % change, d
% summarizeByCond = @(tbl, metricName) local_summarizeByCond(tbl, metricName, condNames, NBOOT, ALPHA);
% 
% % ========== helper: regression QoI ~ cond + S + log(A) + interactions (boot CI) ==========
% runReg = @(tbl, metricName) local_regress(tbl, metricName, NBOOT, ALPHA, USE_LME);
% 
% fprintf('\n==================== PAPER STATS PACK ====================\n');
% fprintf('NBOOT=%d, CI=%d%%\n', NBOOT, round(100*(1-ALPHA)));
% fprintf('Model: QoI ~ condition + S + log(A) + interactions\n');
% 
% % ------------------------- FIGURE 2A1.2 -------------------------
% fprintf('\n==================== Figure 2A1.2: Outcomes + bounded asymmetry ====================\n');
% 
% tbl_muR = makeLong('muR', @(v) v);
% tbl_muD = makeLong('muD', @(v) v);
% 
% % If you still have old asyR/asyD in SRes: summarize them too (but they can blow up)
% tbl_asyR = makeLong('asyR', @(v) v);
% tbl_asyD = makeLong('asyD', @(v) v);
% 
% % If you want bounded indices (recommended): compute bounded per subject from stored muPos/muNeg fields if available
% % This uses per-subject paired pos/neg means already stored in cells.
% tbl_asyRb = local_makeBounded(SRes,'muPosR','muNegR',condNames,'AsyR_bounded');
% tbl_asyDb = local_makeBounded(SRes,'muPosD','muNegD',condNames,'AsyD_bounded');
% 
% summarizeByCond(tbl_muR,  'Mean reward per-step (muR)');
% summarizeByCond(tbl_muD,  'Mean delta per-step (muD)');
% 
% if ~isempty(tbl_asyRb)
%     summarizeByCond(tbl_asyRb,'Bounded asymmetry (Reward)');
%     runReg(tbl_asyRb,'AsyR_bounded');
% end
% if ~isempty(tbl_asyDb)
%     summarizeByCond(tbl_asyDb,'Bounded asymmetry (Delta)');
%     runReg(tbl_asyDb,'AsyD_bounded');
% end
% 
% runReg(tbl_muR,'muR');
% runReg(tbl_muD,'muD');
% 
% % ------------------------- FIGURE 3 -------------------------
% fprintf('\n==================== Figure 3: Positive vs Negative delta ====================\n');
% 
% tbl_posD = makeLong('muPosD', @(v) v);          % E[delta | delta>0]
% tbl_negDmag = makeLong('muNegD', @(v) abs(v)); % E[|delta| | delta<0]
% 
% summarizeByCond(tbl_posD,   'Positive delta mean (E[δ|δ>0])');
% summarizeByCond(tbl_negDmag,'Negative delta magnitude (E[|δ||δ<0])');
% 
% runReg(tbl_posD,'muPosD');
% runReg(tbl_negDmag,'muNegD_abs');
% 
% % ------------------------- FIGURE 4A: slope vs S -------------------------
% fprintf('\n==================== Figure 4A: Bootstrapped tradeoff slope vs S ====================\n');
% % slopeBootS{ic,iS} are bootstrap draws. Treat each draw as a sample for CI + regression.
% 
% if exist('slopeBootS','var') && ~isempty(slopeBootS)
%     tbl_slopeS = local_makeSlopeTable_S(slopeBootS, Svals, condNames);
%     local_summarizeSlope(tbl_slopeS, 'Slope (vs S)', condNames, NBOOT, ALPHA);
%     runReg(tbl_slopeS,'slope');  % uses cond + S only; logA missing => handled
% else
%     fprintf('slopeBootS not found. Skipping Figure 4A stats.\n');
% end
% 
% % ------------------------- FIGURE 4B: slope vs A (pooled over S) -------------------------
% fprintf('\n==================== Figure 4B: Bootstrapped tradeoff slope vs A (pooled S) ====================\n');
% 
% if exist('slopeBootA','var') && ~isempty(slopeBootA)
%     tbl_slopeA = local_makeSlopeTable_A(slopeBootA, A, condNames);
%     local_summarizeSlope(tbl_slopeA, 'Slope (vs A)', condNames, NBOOT, ALPHA);
%     runReg(tbl_slopeA,'slope');
% else
%     fprintf('slopeBootA not found. Skipping Figure 4B stats.\n');
% end
% 
% fprintf('\n==================== END STATS PACK ====================\n');

% %% bootstrapped for A given S
% % Goal: create slopeBootA{ic,ia} (NBOOT x 1) for each condition and A,
% % pooling ALL S datasets (meta_all_runs_s*.mat), then print summary stats
% 
% 
% % -------------------- settings --------------------
% if ~exist('NBOOT','var') || isempty(NBOOT), NBOOT = 10; end
% if ~exist('ALPHA','var') || isempty(ALPHA), ALPHA = 0.05; end
% if ~exist('USE_SPEARMAN','var') || isempty(USE_SPEARMAN), USE_SPEARMAN = false; end
% 
% Svals = arrayfun(@(z) z.S, SRes);
% nS    = numel(Svals);
% 
% A     = SRes(1).A_list(:);
% nA    = numel(A);
% 
% nCond = numel(condNames);
% 
% % If your .mat files are not in pwd, set this:
% if ~exist('matDir','var') || isempty(matDir)
%     matDir = pwd;
% end
% 
% fprintf('\n==================== Figure 4B: Bootstrapped tradeoff slope vs A (pooled over S) ====================\n');
% fprintf('Pooling over %d S levels: %s\n', nS, strjoin(arrayfun(@(s) sprintf('%.1f',s), Svals,'uni',0), ', '));
% fprintf('NBOOT=%d, CI=%d%%\n', NBOOT, round(100*(1-ALPHA)));
% 
% % -------------------- load all metaRes once (speed + reproducibility) --------------------
% metaResS = cell(nS,1);
% for iS = 1:nS
%     matName = fullfile(matDir, sprintf('meta_all_runs_s%.1f.mat', Svals(iS)));
%     tmp = load(matName, 'metaRes');
%     metaResS{iS} = tmp.metaRes;
% end
% 
% % -------------------- compute slopes per (cond, A) pooling across S --------------------
% slopeBootA = cell(nCond, nA);
% 
% for ic = 1:nCond
%     pairs = metaCells{ic};
% 
%     for ia = 1:nA
%         % collect aligned per-subject (X,Y) across ALL S and all pooling pairs
%         X = [];
%         Y = [];
% 
%         for iS = 1:nS
%             metaRes = metaResS{iS};
% 
%             for kk = 1:size(pairs,1)
%                 id = pairs(kk,1);
%                 ip = pairs(kk,2);
% 
%                 traj  = metaRes{id, ip}.res.traj{condIdx(ic), ia};
%                 D     = traj.delta;               % [nSubj x Tmax]
%                 steps = traj.steps_taken(:);      % [nSubj x 1]
% 
%                 [nSubj, Tmax] = size(D);
% 
%                 % mask steps beyond goal
%                 for s = 1:nSubj
%                     Ti = steps(s);
%                     if Ti < Tmax
%                         D(s, Ti+1:end) = NaN;
%                     end
%                 end
% 
%                 % per-subject conditional means
%                 Dpos = D; Dpos(Dpos <= 0) = NaN;
%                 Dneg = D; Dneg(Dneg >= 0) = NaN;
% 
%                 nPos = sum(isfinite(Dpos),2);
%                 nNeg = sum(isfinite(Dneg),2);
% 
%                 muPos = nansum(Dpos,2) ./ nPos;     % E[δ|δ>0]
%                 muNeg = nansum(Dneg,2) ./ nNeg;     % E[δ|δ<0] (negative)
% 
%                 muPos(nPos==0) = NaN;
%                 muNeg(nNeg==0) = NaN;
% 
%                 ok = isfinite(muPos) & isfinite(muNeg);
%                 X = [X; muPos(ok)];
%                 Y = [Y; abs(muNeg(ok))];            % |E[δ|δ<0]|
%             end
%         end
% 
%         nObs = numel(X);
%         if nObs < 10
%             slopeBootA{ic,ia} = nan(NBOOT,1);
%             continue;
%         end
% 
%         slopes = nan(NBOOT,1);
% 
%         for b = 1:NBOOT
%             idx = randi(nObs, nObs, 1);   % bootstrap resample observations
%             xb = X(idx);
%             yb = Y(idx);
% 
%             if USE_SPEARMAN
%                 slopes(b) = corr(xb, yb, 'Type','Spearman', 'Rows','complete');
%             else
%                 p = polyfit(xb, yb, 1);
%                 slopes(b) = p(1);
%             end
%         end
% 
%         slopeBootA{ic,ia} = slopes;
%     end
% end
% 
% % -------------------- pretty summary stats for reporting --------------------
% ciLo = 100*(ALPHA/2);
% ciHi = 100*(1-ALPHA/2);
% 
% fprintf('\n--- Per-condition slope vs A (pooled over S): mean [CI] across bootstrap ---\n');
% for ic = 1:nCond
%     fprintf('\n%s:\n', condNames{ic});
%     for ia = 1:nA
%         v = slopeBootA{ic,ia};
%         v = v(isfinite(v));
%         if isempty(v)
%             fprintf('  A=%g:  (empty)\n', A(ia));
%             continue;
%         end
%         m  = mean(v,'omitnan');
%         ci = prctile(v, [ciLo ciHi]);
%         fprintf('  A=%g:  %.4g  [%.4g, %.4g]  (nboot=%d)\n', A(ia), m, ci(1), ci(2), numel(v));
%     end
% end
% 
% % -------------------- create table + run regression (fixes previous failure) --------------------
% tbl_slopeA = local_makeSlopeTable_A(slopeBootA, A, condNames);
% 
% % quick global summary
% local_summarizeSlope(tbl_slopeA, 'Slope (vs A, pooled S)', condNames, NBOOT, ALPHA);
% 
% % IMPORTANT: runReg expects response column named "value"
% % Convert tbl_slopeA.slope -> tbl_slopeA.value and add S dummy column if needed
% tbl_slopeA.value = tbl_slopeA.slope;   % for compatibility with your runReg()
% 
% % recommended model for Fig 4B: slope ~ cond * logA
% % (S is already pooled; interactions with S are not identifiable here)
% try
%     fprintf('\n--- Regression for slope (Fig 4B): slope ~ cond * logA ---\n');
%     mdl = fitlm(tbl_slopeA, 'value ~ cond * logA');
%     disp(mdl.Coefficients);
% 
%     % bootstrap CI for coefficients (simple resampling over A bins within condition)
%     coefNames = mdl.CoefficientNames;
%     B = NBOOT;
%     bootCoef = nan(B, numel(coefNames));
% 
%     cellID = strcat(string(tbl_slopeA.cond), "_A", string(tbl_slopeA.A));
%     uID = unique(cellID);
% 
%     for b = 1:B
%         idxAll = [];
%         for k = 1:numel(uID)
%             idx = find(cellID == uID(k));
%             idxResamp = idx(randi(numel(idx), numel(idx), 1));
%             idxAll = [idxAll; idxResamp]; %#ok<AGROW>
%         end
%         tbb = tbl_slopeA(idxAll,:);
%         try
%             mb = fitlm(tbb, 'value ~ cond * logA');
%             bootCoef(b,:) = mb.Coefficients.Estimate(:)';
%         catch
%         end
%     end
% 
%     fprintf('\nBootstrap %d%% CI for coefficients:\n', round(100*(1-ALPHA)));
%     for j = 1:numel(coefNames)
%         v = bootCoef(:,j);
%         v = v(isfinite(v));
%         if isempty(v), continue; end
%         ci = prctile(v, [ciLo ciHi]);
%         fprintf('  %s: %.4g  CI[%.4g, %.4g]\n', coefNames{j}, mdl.Coefficients.Estimate(j), ci(1), ci(2));
%     end
% 
% catch ME
%     fprintf('Regression failed: %s\n', ME.message);
% end
% 
% % Figure 4A-by-A: Bootstrapped tradeoff slope vs S (WITHIN each A)
% % What you asked: “bootstrap the slope for S given A”
% % => For each condition ic AND each A index ia:
% %       compute slopeBootS_givenA{ic, ia, iS} (NBOOT x 1)
% %       where each slope distribution is computed using ONLY data from that (A, S),
% %       pooled across meta pairs (metaCells{ic}).
% %
% % This is the exact analogue of your previous “pooled over A” Fig 4A,
% % but now stratified by A. You can later (a) plot slope vs S for each A, or
% % (b) regress slopes on S within each A.
% 
% if ~exist('NBOOT','var') || isempty(NBOOT), NBOOT = 10; end
% if ~exist('ALPHA','var') || isempty(ALPHA), ALPHA = 0.05; end
% if ~exist('USE_SPEARMAN','var') || isempty(USE_SPEARMAN), USE_SPEARMAN = false; end
% 
% Svals = arrayfun(@(z) z.S, SRes);
% nS    = numel(Svals);
% 
% A     = SRes(1).A_list(:);
% nA    = numel(A);
% 
% nCond = numel(condNames);
% 
% if ~exist('matDir','var') || isempty(matDir)
%     matDir = pwd;
% end
% 
% fprintf('\n==================== Figure 4A-by-A: Bootstrapped tradeoff slope vs S (given A) ====================\n');
% fprintf('NBOOT=%d, CI=%d%%\n', NBOOT, round(100*(1-ALPHA)));
% fprintf('S levels: %s\n', strjoin(arrayfun(@(s) sprintf('%.1f',s), Svals,'uni',0), ', '));
% fprintf('A levels: %s\n', strjoin(arrayfun(@(a) sprintf('%.3g',a), A,'uni',0), ', '));
% 
% % -------------------- load all metaRes once --------------------
% metaResS = cell(nS,1);
% for iS = 1:nS
%     matName = fullfile(matDir, sprintf('meta_all_runs_s%.1f.mat', Svals(iS)));
%     tmp = load(matName, 'metaRes');
%     metaResS{iS} = tmp.metaRes;
% end
% 
% % -------------------- compute slopes: (cond, A, S) --------------------
% % slopeBootS_givenA{ic, ia, iS} = [NBOOT x 1]
% slopeBootS_givenA = cell(nCond, nA, nS);
% 
% for ic = 1:nCond
%     pairs = metaCells{ic};
% 
%     for ia = 1:nA
%         for iS = 1:nS
% 
%             metaRes = metaResS{iS};
% 
%             X = [];
%             Y = [];
% 
%             for kk = 1:size(pairs,1)
%                 id = pairs(kk,1);
%                 ip = pairs(kk,2);
% 
%                 traj  = metaRes{id, ip}.res.traj{condIdx(ic), ia};
%                 D     = traj.delta;               % [nSubj x Tmax]
%                 steps = traj.steps_taken(:);      % [nSubj x 1]
%                 [nSubj, Tmax] = size(D);
% 
%                 % mask post-goal steps
%                 for s = 1:nSubj
%                     Ti = steps(s);
%                     if Ti < Tmax
%                         D(s, Ti+1:end) = NaN;
%                     end
%                 end
% 
%                 Dpos = D; Dpos(Dpos <= 0) = NaN;
%                 Dneg = D; Dneg(Dneg >= 0) = NaN;
% 
%                 nPos = sum(isfinite(Dpos),2);
%                 nNeg = sum(isfinite(Dneg),2);
% 
%                 muPos = nansum(Dpos,2) ./ nPos;  % E[δ|δ>0]
%                 muNeg = nansum(Dneg,2) ./ nNeg;  % E[δ|δ<0] (negative)
% 
%                 muPos(nPos==0) = NaN;
%                 muNeg(nNeg==0) = NaN;
% 
%                 ok = isfinite(muPos) & isfinite(muNeg);
%                 X = [X; muPos(ok)];
%                 Y = [Y; abs(muNeg(ok))];
%             end
% 
%             nObs = numel(X);
%             if nObs < 10
%                 slopeBootS_givenA{ic,ia,iS} = nan(NBOOT,1);
%                 continue;
%             end
% 
%             slopes = nan(NBOOT,1);
%             for b = 1:NBOOT
%                 idx = randi(nObs, nObs, 1);
%                 xb = X(idx);
%                 yb = Y(idx);
% 
%                 if USE_SPEARMAN
%                     slopes(b) = corr(xb, yb, 'Type','Spearman', 'Rows','complete');
%                 else
%                     p = polyfit(xb, yb, 1);
%                     slopes(b) = p(1);
%                 end
%             end
% 
%             slopeBootS_givenA{ic,ia,iS} = slopes;
%         end
%     end
% end
% 
% % -------------------- print summary: mean [CI] per (cond, A) across S --------------------
% ciLo = 100*(ALPHA/2);
% ciHi = 100*(1-ALPHA/2);
% 
% fprintf('\n--- Summary: slope vs S (given A) ---\n');
% for ic = 1:nCond
%     fprintf('\n%s:\n', condNames{ic});
%     for ia = 1:nA
%         fprintf('  A=%g:\n', A(ia));
%         for iS = 1:nS
%             v = slopeBootS_givenA{ic,ia,iS};
%             v = v(isfinite(v));
%             if isempty(v)
%                 fprintf('    S=%.1f: (empty)\n', Svals(iS));
%                 continue;
%             end
%             m  = mean(v,'omitnan');
%             ci = prctile(v, [ciLo ciHi]);
%             fprintf('    S=%.1f:  %.4g  [%.4g, %.4g]  (nboot=%d)\n', Svals(iS), m, ci(1), ci(2), numel(v));
%         end
%     end
% end
% 
% % -------------------- OPTIONAL: build a table for regression of slope on S within each A --------------------
% % This creates one row per bootstrap draw, per (cond, A, S), so you can fit:
% %   slope ~ cond * S + cond * logA + S * logA   (or within each A: slope ~ cond * S)
% tbl_slopeS_givenA = local_makeSlopeTable_S_givenA(slopeBootS_givenA, Svals, A, condNames);
% tbl_slopeS_givenA.value = tbl_slopeS_givenA.slope;  % for your existing runReg()
% 
% % Example: within each A, run slope ~ cond * S
% fprintf('\n--- Optional regressions: within each A, slope ~ cond * S ---\n');
% for ia = 1:nA
%     tA = tbl_slopeS_givenA(tbl_slopeS_givenA.iA == ia, :);
%     if height(tA) < 50, continue; end
%     try
%         fprintf('\nA=%g:\n', A(ia));
%         mdlA = fitlm(tA, 'value ~ cond * S');
%         disp(mdlA.Coefficients);
%     catch
%     end
% end
% 
% % ===================== helper =====================
% function tbl = local_makeSlopeTable_S_givenA(slopeBootS_givenA, Svals, A, condNames)
%     nCond = size(slopeBootS_givenA,1);
%     nA    = size(slopeBootS_givenA,2);
%     nS    = size(slopeBootS_givenA,3);
% 
%     rows = cell(0,7);
%     for ic = 1:nCond
%         for ia = 1:nA
%             for iS = 1:nS
%                 v = slopeBootS_givenA{ic,ia,iS};
%                 v = v(isfinite(v));
%                 if isempty(v), continue; end
%                 for j = 1:numel(v)
%                     rows(end+1,:) = {v(j), condNames{ic}, Svals(iS), A(ia), log(A(ia)), ia, iS}; %#ok<AGROW>
%                 end
%             end
%         end
%     end
% 
%     tbl = cell2table(rows, 'VariableNames', {'slope','cond','S','A','logA','iA','iS'});
%     tbl.cond = categorical(tbl.cond);
% end


%% ============================ LOCAL FUNCTIONS ============================
%
function tbl = local_makeLong(SRes, field, transformFn, condNames)
    Svals = arrayfun(@(z) z.S, SRes);
    A = SRes(1).A_list(:);
    nS = numel(SRes);
    nA = numel(A);
    nCond = size(SRes(1).muR,2);

    rows = {};
    for iS = 1:nS
        if ~isfield(SRes(iS), field), continue; end
        C = SRes(iS).(field); % cell(nA,nCond)
        for ia = 1:nA
            for ic = 1:nCond
                v = C{ia,ic};
                v = v(isfinite(v));
                if isempty(v), continue; end
                v = transformFn(v);
                for j=1:numel(v)
                    rows(end+1,:) = {v(j), condNames{ic}, Svals(iS), A(ia), log(A(ia)), iS, ia}; %#ok<AGROW>
                end
            end
        end
    end

    if isempty(rows)
        tbl = table();
        return;
    end

    tbl = cell2table(rows, 'VariableNames', {'value','cond','S','A','logA','iS','iA'});
    tbl.cond = categorical(tbl.cond);
end

function tbl = local_makeBounded(SRes, fieldPos, fieldNeg, condNames, outName)
    Svals = arrayfun(@(z) z.S, SRes);
    A = SRes(1).A_list(:);
    nS = numel(SRes);
    nA = numel(A);
    nCond = size(SRes(1).muR,2);

    if ~isfield(SRes(1), fieldPos) || ~isfield(SRes(1), fieldNeg)
        tbl = table(); return;
    end

    rows = {};
    for iS = 1:nS
        P = SRes(iS).(fieldPos);
        N = SRes(iS).(fieldNeg);
        for ia = 1:nA
            for ic = 1:nCond
                p = P{ia,ic}; n = N{ia,ic};
                p = p(isfinite(p)); n = n(isfinite(n));
                if isempty(p) || isempty(n), continue; end

                % Important: p and n are per-subject quantities in your preprocessing.
                % They should be aligned by subject count. If not, truncate to min length.
                L = min(numel(p), numel(n));
                p = p(1:L); n = n(1:L);

                denom = (p + abs(n));
                asy = (p - abs(n)) ./ denom;
                asy(~isfinite(asy)) = NaN;

                asy = asy(isfinite(asy));
                for j=1:numel(asy)
                    rows(end+1,:) = {asy(j), condNames{ic}, Svals(iS), A(ia), log(A(ia)), iS, ia}; %#ok<AGROW>
                end
            end
        end
    end

    if isempty(rows)
        tbl = table(); return;
    end

    tbl = cell2table(rows, 'VariableNames', {'value','cond','S','A','logA','iS','iA'});
    tbl.cond = categorical(tbl.cond);
    tbl.Properties.VariableNames{1} = outName; % rename first col
    tbl = renamevars(tbl, outName, 'value');   % keep unified name "value"
end

function local_summarizeByCond(tbl, metricName, condNames, NBOOT, ALPHA)
    if isempty(tbl) || height(tbl)==0
        fprintf('%s: table empty, skipping.\n', metricName);
        return;
    end

    fprintf('\n--- %s (pooled over S,A) ---\n', metricName);

    % Condition means + bootstrap CI
    for ic = 1:numel(condNames)
        v = tbl.value(tbl.cond == condNames{ic});
        v = v(isfinite(v));
        [m, ci] = bootMeanCI(v, NBOOT, ALPHA);
        fprintf('%s: mean=%.4g  CI[%.4g, %.4g]  n=%d\n', condNames{ic}, m, ci(1), ci(2), numel(v));
    end

    % Δ vs Control
    vC = tbl.value(tbl.cond == condNames{1});
    vC = vC(isfinite(vC));

    for ic = 2:numel(condNames)
        vT = tbl.value(tbl.cond == condNames{ic});
        vT = vT(isfinite(vT));

        [dMean, ciD] = bootDiffCI(vT, vC, NBOOT, ALPHA);
        pc = 100 * (mean(vT,'omitnan') - mean(vC,'omitnan')) / max(abs(mean(vC,'omitnan')), eps);
        d = cohend(vT, vC);

        fprintf('%s vs %s: Δmean=%.4g  CI[%.4g, %.4g]  %%change=%.2f%%  d=%.3g\n', ...
            condNames{ic}, condNames{1}, dMean, ciD(1), ciD(2), pc, d);
    end
end

function local_summarizeSlope(tbl, metricName, condNames, NBOOT, ALPHA)
    if isempty(tbl) || height(tbl)==0
        fprintf('%s: table empty, skipping.\n', metricName);
        return;
    end
    fprintf('\n--- %s: bootstrap distributions ---\n', metricName);

    for ic = 1:numel(condNames)
        v = tbl.slope(tbl.cond == condNames{ic});
        v = v(isfinite(v));
        [m, ci] = bootMeanCI(v, NBOOT, ALPHA);
        fprintf('%s: mean=%.4g  CI[%.4g, %.4g]  n=%d\n', condNames{ic}, m, ci(1), ci(2), numel(v));
    end
end

function local_regress(tbl, metricName, NBOOT, ALPHA, USE_LME)
    if isempty(tbl) || height(tbl)==0
        fprintf('\n[%s] regression: table empty, skipping.\n', metricName);
        return;
    end

    % Determine available predictors (slope tables may not have logA)
    hasLogA = any(strcmp(tbl.Properties.VariableNames,'logA'));
    hasS    = any(strcmp(tbl.Properties.VariableNames,'S'));
    hasCond = any(strcmp(tbl.Properties.VariableNames,'cond'));

    fprintf('\n--- Regression for %s ---\n', metricName);

    if ~hasCond
        fprintf('No condition column found; skipping.\n');
        return;
    end

    % Build formula
    if hasS && hasLogA
        form = 'value ~ cond * S + cond * logA + S * logA';
    elseif hasS
        form = 'value ~ cond * S';
    elseif hasLogA
        form = 'value ~ cond * logA';
    else
        form = 'value ~ cond';
    end

    % Fit point-estimate model
    try
        if USE_LME && exist('fitlme','file')==2 && all(ismember({'iS','iA'}, tbl.Properties.VariableNames))
            % mixed-effects: random intercepts for S-level and A-level index
            tbl.iS = categorical(tbl.iS);
            tbl.iA = categorical(tbl.iA);
            lmeForm = strrep(form,'value','value'); %#ok<NASGU>
            lmeForm = [form ' + (1|iS) + (1|iA)'];
            mdl = fitlme(tbl, lmeForm);
            fprintf('fitlme: %s\n', lmeForm);
            disp(mdl.Coefficients);
        else
            mdl = fitlm(tbl, form);
            fprintf('fitlm: %s\n', form);
            disp(mdl.Coefficients);
        end
    catch ME
        fprintf('Model fit failed: %s\n', ME.message);
        return;
    end

    % Bootstrap CIs for coefficients (clustered by (cond,S,A) cell)
    % We resample within each cell to preserve design structure.
    coefNames = mdl.CoefficientNames;
    B = NBOOT;
    C = numel(coefNames);
    bootCoef = nan(B,C);

    % cell id: condition x S x A
    if all(ismember({'cond','S','A'}, tbl.Properties.VariableNames))
        cellID = strcat(string(tbl.cond), "_S", string(tbl.S), "_A", string(tbl.A));
    elseif all(ismember({'cond','S'}, tbl.Properties.VariableNames))
        cellID = strcat(string(tbl.cond), "_S", string(tbl.S));
    elseif all(ismember({'cond','A'}, tbl.Properties.VariableNames))
        cellID = strcat(string(tbl.cond), "_A", string(tbl.A));
    else
        cellID = string(tbl.cond);
    end

    uID = unique(cellID);

    for b = 1:B
        idxAll = [];
        for k = 1:numel(uID)
            idx = find(cellID == uID(k));
            if isempty(idx), continue; end
            idxResamp = idx(randi(numel(idx), numel(idx), 1));
            idxAll = [idxAll; idxResamp]; %#ok<AGROW>
        end

        tbb = tbl(idxAll,:);
        try
            if USE_LME && exist('fitlme','file')==2 && all(ismember({'iS','iA'}, tbb.Properties.VariableNames))
                tbb.iS = categorical(tbb.iS); tbb.iA = categorical(tbb.iA);
                lmeForm = [form ' + (1|iS) + (1|iA)'];
                mb = fitlme(tbb, lmeForm);
                bootCoef(b,:) = mb.Coefficients.Estimate(:)';
            else
                mb = fitlm(tbb, form);
                bootCoef(b,:) = mb.Coefficients.Estimate(:)';
            end
        catch
            % leave NaNs for failed bootstrap draw
        end
    end

    % Print bootstrap CI for each coefficient
    fprintf('\nBootstrap %d%% CI for coefficients:\n', round(100*(1-ALPHA)));
    for j = 1:C
        v = bootCoef(:,j);
        v = v(isfinite(v));
        if isempty(v), continue; end
        ci = prctile(v, [100*ALPHA/2, 100*(1-ALPHA/2)]);
        fprintf('  %s: %.4g  CI[%.4g, %.4g]\n', coefNames{j}, mdl.Coefficients.Estimate(j), ci(1), ci(2));
    end
end

function tbl = local_makeSlopeTable_S(slopeBootS, Svals, condNames)
    % slopeBootS{ic,iS} = [NBOOT x 1]
    rows = {};
    nCond = size(slopeBootS,1);
    nS = size(slopeBootS,2);

    for ic = 1:nCond
        for iS = 1:nS
            v = slopeBootS{ic,iS};
            v = v(isfinite(v));
            if isempty(v), continue; end
            for j=1:numel(v)
                rows(end+1,:) = {v(j), condNames{ic}, Svals(iS), iS}; %#ok<AGROW>
            end
        end
    end

    tbl = cell2table(rows, 'VariableNames', {'slope','cond','S','iS'});
    tbl.cond = categorical(tbl.cond);
end

function tbl = local_makeSlopeTable_A(slopeBootA, A, condNames)
    % slopeBootA{ic,ia} = [NBOOT x 1]
    rows = {};
    nCond = size(slopeBootA,1);
    nA = size(slopeBootA,2);

    for ic = 1:nCond
        for ia = 1:nA
            v = slopeBootA{ic,ia};
            v = v(isfinite(v));
            if isempty(v), continue; end
            for j=1:numel(v)
                rows(end+1,:) = {v(j), condNames{ic}, A(ia), log(A(ia)), ia}; %#ok<AGROW>
            end
        end
    end

    tbl = cell2table(rows, 'VariableNames', {'slope','cond','A','logA','iA'});
    tbl.cond = categorical(tbl.cond);
end

function [m, ci] = bootMeanCI(x, B, ALPHA)
    x = x(isfinite(x));
    n = numel(x);
    if n==0, m=NaN; ci=[NaN NaN]; return; end
    m = mean(x,'omitnan');
    bs = nan(B,1);
    for b=1:B
        xb = x(randi(n,n,1));
        bs(b) = mean(xb,'omitnan');
    end
    ci = prctile(bs,[100*ALPHA/2, 100*(1-ALPHA/2)]);
end

function [dMean, ci] = bootDiffCI(x, y, B, ALPHA)
    x = x(isfinite(x)); y = y(isfinite(y));
    nx = numel(x); ny = numel(y);
    if nx==0 || ny==0, dMean=NaN; ci=[NaN NaN]; return; end
    dMean = mean(x,'omitnan') - mean(y,'omitnan');
    bs = nan(B,1);
    for b=1:B
        xb = x(randi(nx,nx,1));
        yb = y(randi(ny,ny,1));
        bs(b) = mean(xb,'omitnan') - mean(yb,'omitnan');
    end
    ci = prctile(bs,[100*ALPHA/2, 100*(1-ALPHA/2)]);
end

function d = cohend(x,y)
    x = x(isfinite(x)); y = y(isfinite(y));
    if isempty(x) || isempty(y), d=NaN; return; end
    mx = mean(x,'omitnan'); my = mean(y,'omitnan');
    sx = std(x,'omitnan');  sy = std(y,'omitnan');
    sp = sqrt((sx.^2 + sy.^2)/2);
    d = (mx - my) / max(sp, eps);
end


