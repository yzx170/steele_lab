% Identify distinct gait features in Feedback trials
% Methods: kinematics, kinetics, SVD, feature scores, weighted average
% Reference from baseline and emulation trials
% Outputs: GroupN

% Written by: Robin Yan 6/1/2020

clear; close all; clc


%% Initialization - Specify Participant Number

participants = {'P001','P002','P003','P004','P006','P007','P009','P010','P011','P012','P013','P014','P015'};
for jjj = 1:length(participants)
    participant = string(participants(jjj)); 
%     participant = 'P001';
    isSDA = 1; % 1: test SD above; 2: test SD below; 0: both
    pathInitialize = 'G:\Shared drives\Steele Lab\Projects\RT Synergies'; 
    basePath = strcat(pathInitialize,'\',participant,'\Matlab_Processed\');
    fileName = strcat(participant,'_Summary_File_GaitID.mat');
    % fileName = strcat(participant,'_Summary_File_FB.mat');
    load(fullfile(basePath,fileName));


    %% Generate Kinematic Matrices

    % Variables of Interests
    voi = [4 7 8]; % Hip flexion, knee and ankle
    % voi = 1:8; % first 8 kinematic parameters from GDI

    % Dominant Leg (1 - Right; 0 - Left)
    domLeg = master.participantInfo.domLeg;
    if domLeg == 1
        ALL_trials = 2:2:size(master.KIN,2); % all gait trials
    elseif domLeg == 0
        ALL_trials = (2:2:size(master.KIN,2))-1;
    else
        error('Cannot find dominant leg information!!!');
    end

    % Kinematics - specified from ALL_trials
    KIN = cell(length(voi),length(ALL_trials)); 
    for j = 1:length(voi)
        % extract kinematic data for both SD trials
        for jj = 1:length(ALL_trials)
        KIN{j,jj} = squeeze(master.KIN(ALL_trials(jj)).segmentData{1,1}(:,voi(j),:)); % 1: entire cycle; 2:stance only
        end
    end

    % Convert cell to matrix
    KIN = cell2mat(KIN);

    % Setup index for all trials
    clusterInd = zeros(1,size(KIN,2));
    len = zeros(1,length(ALL_trials));
    len(1) = size(master.KIN(ALL_trials(1)).segmentData{1,1},3);
    clusterInd(1:len(1)) = 1;
    for j = 2:length(ALL_trials)
        len(j) = size(master.KIN(ALL_trials(j)).segmentData{1,1},3);
        clusterInd((sum(len(1:j-1))+1):(sum(len(1:j)))) = j;
    end

    % Remove NaNs columns
    clusterInd(isnan(KIN(1,:))) = [];
    disp(strcat(num2str(sum(isnan(KIN(1,:)))),32,'NaNs removed.'))
    KIN(:,isnan(KIN(1,:))) = [];


    %% Feature Scores Calculation

    % Pre-run SVD and Feature Scores
    % Perform SVD 
    [U,~,~] = svd(KIN);

    % Feature Scores
    feature_scores = U'*KIN;

    % Calculate Feature Scores and VAF (raw kinematic data)
    targetVAF = 0.95;
    n = 0; % default from 0
    VAF = zeros(1,size(KIN,2));
    while min(VAF) < targetVAF % iterate on number of scores required for target VAF
        n = n+1; % number of feature scores
        D = zeros(size(U));
        for j = 1:size(KIN,2) % calculate VAF through all observations
            D(1:n,1:n) = diag(feature_scores(1:n,j));
            g_appr = sum(U*D,2);
            g = KIN(:,j);
            VAF(1,j) = (g_appr'*g_appr)/(g'*g);
        end
    end

    % Identify Outliers
    % Outliers with sliding windowed for all ns
    TF = zeros(n,length(VAF));
    for jj = 1:n % iterate through all modal numbers
        VAF = zeros(1,size(KIN,2));
        D = zeros(size(U));
        for j = 1:size(feature_scores,2) % iterate through all observations
            g = KIN(:,j);
            D(1:jj,1:jj) = diag(feature_scores(1:jj,j));
            g_appr = sum(U*D,2);
            VAF(1,j) = (g_appr'*g_appr)/(g'*g);
        end
        TF(jj,:) = isoutlier(VAF,'movmedian',10);
    end

    % Summation to determine outliners
    TF = sum(TF,1);
    [totalOccurance,Ind] = max(TF);
    if totalOccurance < n
        disp('Current outliers not present in all iterations! Manual check recommended!')
    end

    % Remove outliers from KIN
    KIN(:,Ind) = [];
    clusterInd(:,Ind) = [];
    disp(strcat(num2str(length(Ind)),32,'outliers in VAF calculation removed.'))

    % SVD and Feature Scores on Reduced Data
    % Perform SVD 
    [U,~,~] = svd(KIN);

    % Feature Scores
    feature_scores = U'*KIN;

    % Calculate Feature Scores and VAF (raw kinematic data)
    n = 0; % default from 0
    VAF = zeros(1,size(KIN,2));
    while min(VAF) < targetVAF % iterate on number of scores required for 0.95 VAF
        n = n+1; % number of feature scores
        D = zeros(size(U));
        for j = 1:size(KIN,2) % calculate VAF through all observations
            D(1:n,1:n) = diag(feature_scores(1:n,j));
            g_appr = sum(U*D,2);
            g = KIN(:,j);
            VAF(1,j) = (g_appr'*g_appr)/(g'*g);
        end
    end

    % Feature Scores with n modes
    feature_scores = feature_scores(1:n,:);

    % Determine the location of FB trials
    locSDA = find(contains({master.KIN(1:2:end).emulationType},'SDA'));
    locSDB = find(contains({master.KIN(1:2:end).emulationType},'SDB'));
    locBASE = find(contains({master.KIN(1:2:end).emulationType},'Baseline'));
    locREF = 1:max(clusterInd);
    locREF([locSDA locSDB]) = [];
    
    % Remove local cluster outliers in Reference FS
    outlierN = 0;
    for j = 1:length(locREF)
        % current cluster
        query = locREF(j);
        cluster = feature_scores(:,clusterInd == query);
        clusterInit = find(clusterInd == query,1);
        % locate outliers
        centroid = mean(cluster,2);
        dist = sqrt(sum((cluster-centroid).^2,1));
        outlierInd = find(isoutlier(dist,'ThresholdFactor',3));
        % remove outliers
        globalInd = clusterInit-1+outlierInd;
        feature_scores(:,globalInd) = [];
        clusterInd(:,globalInd) = [];
        VAF(:,globalInd) = [];
        outlierN = outlierN+length(outlierInd);
    end
    disp(strcat(num2str(outlierN),32,'outliers in local REF clusters removed.'))
    
    % Extract Feedback Trial FS
    if isSDA == 1
        test = feature_scores(:,ismember(clusterInd,locSDA)); % SDA
        testInit = find(clusterInd == locSDA(1),1);
    elseif isSDA == 2
        test = feature_scores(:,ismember(clusterInd,locSDB)); % SDB
        testInit = find(clusterInd == locSDB(1),1);
    else
        test = feature_scores(:,ismember(clusterInd,[locSDA locSDB])); % both
        testInit = find(clusterInd == locSDA(1),1);
    end

    % Diameter of local clusters in Reference FS
    DIA = zeros(1,length(locREF));
    for j = 1:length(DIA)
        query = locREF(j);
        baseline = feature_scores(:,clusterInd == query);
        DIA(j) = 2*max(sqrt(sum((baseline-mean(baseline,2)).^2,1)));
    end
    
    % Average diameter of baseline - for normalization
    BASE = zeros(1,length(locBASE));
    for j = 1:length(BASE)
        query = locBASE(j);
        baseline = feature_scores(:,clusterInd == query);
        BASE(j) = 2*max(sqrt(sum((baseline-mean(baseline,2)).^2,1)));
    end
    BASE = mean(BASE);
    
    % Threshold as max DIA, normalized
    thresh = max(DIA)/BASE;


     %% Generate Kinetic Matrices

    % Variables of Interests
    voi = [4 7 8]; % Hip flexion, knee and ankle
    % voi = 1:8; % first 8 kinematic parameters from GDI

    % Kinetic - specified from ALL_trials
    KNT = cell(length(voi),length(ALL_trials)); 
    for j = 1:length(voi)
        % extract kinetic data for both SD trials
        for jj = 1:length(ALL_trials)
        KNT{j,jj} = squeeze(master.KNT(ALL_trials(jj)).segmentData{1,2}(:,voi(j),:)); % 1: entire cycle; 2:stance only
        end
    end

    % Convert cell to matrix
    KNT = cell2mat(KNT);

    % Setup index for all trials
    clusterInd = zeros(1,size(KNT,2));
    len = zeros(1,length(ALL_trials));
    len(1) = size(master.KNT(ALL_trials(1)).segmentData{1,1},3);
    clusterInd(1:len(1)) = 1;
    for j = 2:length(ALL_trials)
        len(j) = size(master.KNT(ALL_trials(j)).segmentData{1,1},3);
        clusterInd((sum(len(1:j-1))+1):(sum(len(1:j)))) = j;
    end

    % Remove NaNs columns
    clusterInd(isnan(KNT(1,:))) = [];
    disp(strcat(num2str(sum(isnan(KNT(1,:)))),32,'NaNs removed.'))
    KNT(:,isnan(KNT(1,:))) = [];


    %% Feature Scores Calculation

    % Pre-run SVD and Feature Scores
    % Perform SVD 
    [U,~,~] = svd(KNT);

    % Feature Scores
    feature_scores = U'*KNT;

    % Calculate Feature Scores and VAF (raw kinematic data)
    targetVAF = 0.95;
    n = 0; % default from 0
    VAF_KNT = zeros(1,size(KNT,2));
    while min(VAF_KNT) < targetVAF % iterate on number of scores required for target VAF
        n = n+1; % number of feature scores
        D = zeros(size(U));
        for j = 1:size(KNT,2) % calculate VAF through all observations
            D(1:n,1:n) = diag(feature_scores(1:n,j));
            g_appr = sum(U*D,2);
            g = KNT(:,j);
            VAF_KNT(1,j) = (g_appr'*g_appr)/(g'*g);
        end
    end

    % Identify Outliers
    % Outliers with sliding windowed for all ns
    TF = zeros(n,length(VAF_KNT));
    for jj = 1:n % iterate through all modal numbers
        VAF_KNT = zeros(1,size(KNT,2));
        D = zeros(size(U));
        for j = 1:size(feature_scores,2) % iterate through all observations
            g = KNT(:,j);
            D(1:jj,1:jj) = diag(feature_scores(1:jj,j));
            g_appr = sum(U*D,2);
            VAF_KNT(1,j) = (g_appr'*g_appr)/(g'*g);
        end
        TF(jj,:) = isoutlier(VAF_KNT,'movmedian',10);
    end

    % Summation to determine outliners
    TF = sum(TF,1);
    [totalOccurance,Ind] = max(TF);
    if totalOccurance < n
        disp('Current outliers not present in all iterations! Manual check recommended!')
    end

    % Remove outliers from KNT
    KNT(:,Ind) = [];
    clusterInd(:,Ind) = [];
    disp(strcat(num2str(length(Ind)),32,'outliers in VAF calculation removed.'))

    % SVD and Feature Scores on Reduced Data
    % Perform SVD 
    [U,~,~] = svd(KNT);

    % Feature Scores
    feature_scores = U'*KNT;

    % Calculate Feature Scores and VAF (raw kinematic data)
    n = 0; % default from 0
    VAF_KNT = zeros(1,size(KNT,2));
    while min(VAF_KNT) < targetVAF % iterate on number of scores required for 0.95 VAF
        n = n+1; % number of feature scores
        D = zeros(size(U));
        for j = 1:size(KNT,2) % calculate VAF through all observations
            D(1:n,1:n) = diag(feature_scores(1:n,j));
            g_appr = sum(U*D,2);
            g = KNT(:,j);
            VAF_KNT(1,j) = (g_appr'*g_appr)/(g'*g);
        end
    end

    % Feature Scores with n modes
    feature_scores = feature_scores(1:n,:);

    % Determine the location of FB trials
    locSDA = find(contains({master.KNT(1:2:end).emulationType},'SDA'));
    locSDB = find(contains({master.KNT(1:2:end).emulationType},'SDB'));
    locBASE = find(contains({master.KNT(1:2:end).emulationType},'Baseline'));
    locREF = 1:max(clusterInd);
    locREF([locSDA locSDB]) = [];

    % Remove local cluster outliers in Reference FS
    outlierN = 0;
    for j = locREF(1:end)
        % current cluster
        cluster = feature_scores(:,clusterInd == j);
        clusterInit = find(clusterInd == j,1);
        % locate outliers
        centroid = mean(cluster,2);
        dist = sqrt(sum((cluster-centroid).^2,1));
        outlierInd = find(isoutlier(dist,'ThresholdFactor',3));
        % remove outliers
        globalInd = clusterInit-1+outlierInd;
        feature_scores(:,globalInd) = [];
        clusterInd(:,globalInd) = [];
        VAF_KNT(:,globalInd) = [];
        outlierN = outlierN+length(outlierInd);
    end
    disp(strcat(num2str(outlierN),32,'outliers in local REF clusters removed.'))
    
    % Extract Feedback Trial FS
    if isSDA == 1
        test_KNT = feature_scores(:,ismember(clusterInd,locSDA)); % SDA
        testInit_KNT = find(clusterInd == locSDA(1),1);
    elseif isSDA == 2
        test_KNT = feature_scores(:,ismember(clusterInd,locSDB)); % SDB
        testInit_KNT = find(clusterInd == locSDB(1),1);
    else
        test_KNT = feature_scores(:,ismember(clusterInd,[locSDA locSDB])); % both
        testInit_KNT = find(clusterInd == locSDA(1),1);
    end

    % Diameter of local clusters in Reference FS
    DIA_KNT = zeros(1,length(locREF));
    for j = 1:length(DIA_KNT)
        jj = locREF(j);
        baseline = feature_scores(:,clusterInd == jj);
        DIA_KNT(j) = 2*max(sqrt(sum((baseline-mean(baseline,2)).^2,1)));
    end
    
    % Average diameter of baseline - for normalization
    BASE_KNT = zeros(1,length(locBASE));
    for j = 1:length(BASE_KNT)
        query = locBASE(j);
        baseline = feature_scores(:,clusterInd == query);
        BASE_KNT(j) = 2*max(sqrt(sum((baseline-mean(baseline,2)).^2,1)));
    end
    BASE_KNT = mean(BASE_KNT);
    
    % Threshold as max DIA, normalized
    thresh_KNT = max(DIA_KNT)/BASE_KNT;


    %% Sequential Grouping based on linkage distance - Max DIA

    % Check if KIN can be combined with KNT
    if size(test,2) > size(test_KNT,2)
        disp('Matrix dimension inconsistent between KIN and KNT data: Use only KIN')
        % Initialization
        GroupN = zeros(1,size(test,2)); % preallocate group number (all zeros)
        cache = test(:,1);
        cache_list = 1;
        N = 1;
        n = 1;
        % Iteration
        for j = 2:size(test,2)
            current = test(:,j);
            dist = sqrt(sum((current-cache).^2,1)); % linkage distance
            if max(dist) <= thresh % add to cache
                cache = [cache current]; %#ok<AGROW>
                cache_list = [cache_list j]; %#ok<AGROW>
                n = n+1;
            else % jump detected
                if n >= 5 % criterion for new met
                    GroupN(cache_list) = N; % assign new group
                    N = N+1; % for next group
                elseif N>=2  % not eligible for new group, but previous group exists
                    for jj = 1:length(cache_list) % check if any belongs to previous group
                        link = max(sqrt(sum((cache(:,jj)-test(:,GroupN == (N-1))).^2,1)));
                        if link < thresh
                            GroupN(cache_list(jj)) = N-1; % assign to last group
                        end
                    end
                end
                % reset cache
                cache = current;
                cache_list = j;
                n = 1;
            end   
        end
        % Check for last cache
        if n >= 5 % criterion for new met
            GroupN(cache_list) = N; % assign new group
        elseif N>=2  % not eligible for new group, but previous group exists
            for jj = 1:length(cache_list) % check if any belongs to previous group
                link = min(sqrt(sum((cache(:,jj)-test(:,GroupN == (N-1))).^2,1)));
                if link < thresh
                    GroupN(cache_list(jj)) = N-1; % assign to last group
                end
            end
        end
    elseif size(test,2) < size(test_KNT,2)
        disp('Matrix dimension inconsistent between KIN and KNT data: Use only KNT')
        % Initialization
        GroupN = zeros(1,size(test_KNT,2)); % preallocate group number (all zeros)
        cache = test_KNT(:,1);
        cache_list = 1;
        N = 1;
        n = 1;
        % Iteration
        for j = 2:size(test_KNT,2)
            current = test_KNT(:,j);
            dist = sqrt(sum((current-cache).^2,1)); % linkage distance
            if max(dist) <= thresh_KNT % add to cache
                cache = [cache current]; %#ok<AGROW>
                cache_list = [cache_list j]; %#ok<AGROW>
                n = n+1;
            else % jump detected
                if n >= 5 % criterion for new met
                    GroupN(cache_list) = N; % assign new group
                    N = N+1; % for next group
                elseif N>=2  % not eligible for new group, but previous group exists
                    for jj = 1:length(cache_list) % check if any belongs to previous group
                        link = max(sqrt(sum((cache(:,jj)-test_KNT(:,GroupN == (N-1))).^2,1)));
                        if link < KNT_thresh
                            GroupN(cache_list(jj)) = N-1; % assign to last group
                        end
                    end
                end
                % reset cache
                cache = current;
                cache_list = j;
                n = 1;
            end   
        end
        % Check for last cache
        if n >= 5 % criterion for new met
            GroupN(cache_list) = N; % assign new group
        elseif N>=2  % not eligible for new group, but previous group exists
            for jj = 1:length(cache_list) % check if any belongs to previous group
                link = min(sqrt(sum((cache(:,jj)-test_KNT(:,GroupN == (N-1))).^2,1)));
                if link < KNT_thresh
                    GroupN(cache_list(jj)) = N-1; % assign to last group
                end
            end
        end
    else
        % Initialization
        GroupN = zeros(1,size(test,2)); % preallocate group number (all zeros)
        cache = test(:,1);
        cache_KNT = test_KNT(:,1);
        cache_list = 1;
        N = 1;
        n = 1;
        % Iteration
        for j = 2:size(test,2)
            current = test(:,j);
            current_KNT = test_KNT(:,j);
            % Weight - exponentially scaled
            current_VAF_KIN = VAF(testInit-1+j);
            current_VAF_KNT = VAF_KNT(testInit_KNT-1+j);
            modifier = 10;
            KIN_weight = (exp(current_VAF_KIN*modifier)-1)/(exp(modifier)-1);
            KNT_weight = (exp(current_VAF_KNT*modifier)-1)/(exp(modifier)-1);
            % Linkage distance, normalized
            dist = sqrt(sum((current-cache).^2,1))/BASE; 
            dist_KNT = sqrt(sum((current_KNT-cache_KNT).^2,1))/BASE_KNT;
            dif = max(dist)-thresh;
            dif_KNT = max(dist_KNT)-thresh_KNT;
            crit = mean(KIN_weight*dif+KNT_weight*dif_KNT);
            if crit <= 0 % add to cache
                cache = [cache current]; %#ok<AGROW>
                cache_KNT = [cache_KNT current_KNT];  %#ok<AGROW>
                cache_list = [cache_list j]; %#ok<AGROW>
                n = n+1;
            else % jump detected
                if n >= 5 % criterion for new met
                    GroupN(cache_list) = N; % assign new group
                    N = N+1; % for next group
                elseif N>=2  % not eligible for new group, but previous group exists
                    for jj = 1:length(cache_list) % check if any belongs to previous group
                        link = max(sqrt(sum((cache(:,jj)-test(:,GroupN == (N-1))).^2,1)))/BASE;
                        link_KNT = max(sqrt(sum((cache_KNT(:,jj)-test_KNT(:,GroupN == (N-1))).^2,1)))/BASE_KNT;
                        dif = link-thresh;
                        dif_KNT = link_KNT-thresh_KNT;
                        crit = mean(KIN_weight*dif+KNT_weight*dif_KNT);
                        if crit <= 0
                            GroupN(cache_list(jj)) = N-1; % assign to last group
                        end
                    end
                end
                % reset cache
                cache = current;
                cache_KNT = current_KNT;
                cache_list = j;
                n = 1;
            end   
        end
        % Check for last cache
        if n >= 5 % criterion for new met
            GroupN(cache_list) = N; % assign new group
        elseif N>=2  % not eligible for new group, but previous group exists
            for jj = 1:length(cache_list) % check if any belongs to previous group
                % Weight - exponentially scaled
                current_VAF_KIN = VAF(testInit-1+cache_list(jj));
                current_VAF_KNT = VAF_KNT(testInit_KNT-1+cache_list(jj));
                KIN_weight = (exp(current_VAF_KIN*modifier)-1)/(exp(modifier)-1);
                KNT_weight = (exp(current_VAF_KNT*modifier)-1)/(exp(modifier)-1);
                link = max(sqrt(sum((cache(:,jj)-test(:,GroupN == (N-1))).^2,1)))/BASE;
                link_KNT = max(sqrt(sum((cache_KNT(:,jj)-test_KNT(:,GroupN == (N-1))).^2,1)))/BASE_KNT;
                dif = link-thresh;
                dif_KNT = link_KNT-thresh_KNT;
                crit = mean(KIN_weight*dif+KNT_weight*dif_KNT);
                if crit <= 0
                    GroupN(cache_list(jj)) = N-1; % assign to last group
                end
            end
        end


        %% Visualization and Outputs
        figure(1)
        set(gcf,'Position',[100 100 1000 900])
        subplot(2,1,1)
        try
            gscatter(test(1,:),test(2,:),GroupN)
            title(strcat('KIN data clustered:',num2str(length(GroupN)),32,'DIA at',32,num2str(thresh)))
        catch
            scatter(test(1,:),test(2,:))
            title('KIN data')
        end
        subplot(2,1,2)
        try
            gscatter(test_KNT(1,:),test_KNT(2,:),GroupN)
            title(strcat('KNT data clustered:',num2str(length(GroupN)),32,'DIA at',32,num2str(thresh_KNT)))
        catch
            scatter(test_KNT(1,:),test_KNT(2,:))
            title('KNT data')
        end
        saveas(gcf,strcat(participant,'.svg'))
        
        
    end
end
