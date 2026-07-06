classdef CIDKTNoise < Algorithm

    %------------------------------- Copyright --------------------------------
    % Copyright (c) Yanchi Li. You are free to use the MToP for research
    % purposes. All publications which use this platform should acknowledge
    % the use of "MToP" or "MTO-Platform" and cite as "Y. Li, W. Gong, F. Ming,
    % T. Zhang, S. Li, and Q. Gu, MToP: A MATLAB Optimization Platform for
    % Evolutionary Multitasking, 2023, arXiv:2312.08134"
    %--------------------------------------------------------------------------

    properties (SetAccess = private)
        MuC = 2
        MuM = 5
        KTN = 5
        rho0 = 0.5
        Gap = 5
        delta0 = 0.5
        ParaMin = 0.05
        ParaMax = 0.95
        vartheta = 0.35
        theta  = 0.5;
        d_star = 10;
    end

    methods
        function Parameter = getParameter(Algo)
            Parameter = {'MuC: Simulated Binary Crossover', num2str(Algo.MuC), ...
                'MuM: Polynomial Mutation', num2str(Algo.MuM), ...
                'KTN: Knowledge Transfer Tasks Num', num2str(Algo.KTN), ...
                'rho0: Initial rho', num2str(Algo.rho0), ...
                'Gap: Parameter update gap', num2str(Algo.Gap), ...
                'delta0: Initial delta', num2str(Algo.delta0), ...
                'ParaMin: Lower bound of parameter', num2str(Algo.ParaMin), ...
                'ParaMax: Upper bound of parameter', num2str(Algo.ParaMax), ...
                'vartheta: Threshold', num2str(Algo.vartheta), ...
                'd_star: The number of dominant curvature directions', num2str(Algo.d_star), ...
                'theta: Strength of knowledge transfer', num2str(Algo.theta)};
        end

        function Algo = setParameter(Algo, Parameter)
            i = 1;
            Algo.MuC      = str2double(Parameter{i}); i = i + 1;
            Algo.MuM      = str2double(Parameter{i}); i = i + 1;
            Algo.KTN      = str2double(Parameter{i}); i = i + 1;
            Algo.rho0   = str2double(Parameter{i}); i = i + 1;
            Algo.Gap    =  str2double(Parameter{i});  i = i + 1;
            Algo.delta0   = str2double(Parameter{i}); i = i + 1;
            Algo.ParaMin = str2double(Parameter{i}); i = i + 1;
            Algo.ParaMax = str2double(Parameter{i}); i = i + 1;
            Algo.vartheta = str2double(Parameter{i}); i = i + 1;
            Algo.d_star = str2double(Parameter{i}); i = i + 1;
            Algo.theta   = str2double(Parameter{i});
        end

        function run(Algo, Prob)
            % Initialization
            population = Initialization(Algo, Prob, Individual);
            rho = Algo.rho0 * ones(1, Prob.T);
            sigma = 0.01;
            for t = 1:Prob.T
                [~, ~, c, theta] = Algo.EstimateNoisyCurvatureFromPopulation(population{t}, sigma);
                fprintf('Task %d: c = %.4f, theta = %.4f rad, theta = %.2f deg\n', ...
                    t, c, theta, theta * 180 / pi);
            end

            Ng = zeros(1, Prob.T);
            Nb = zeros(1, Prob.T);
            Nm = zeros(1, Prob.T);
            Rg = zeros(1, Prob.T);   % rank contribution of intratask offspring
            Rb = zeros(1, Prob.T);   % rank contribution of SSEI offspring
            Rm = zeros(1, Prob.T);   % rank contribution of MGMCE offspring

            delta = Algo.delta0 * ones(1, Prob.T);
            [v_cur, matrix_R] = Algo.CalcTaskSimilarityHessian(population, Algo.d_star);
            while Algo.notTerminated(Prob, population)
  
                if mod(Algo.Gen,Algo.Gap) == 0
                    [v_cur, matrix_R] = Algo.CalcTaskSimilarityHessian(population, Algo.d_star);
                end

                % --------- Generation ---------
                [offspring, transferFlag] = Algo.Generation(Prob, population, matrix_R, v_cur, rho, delta);

                % --------- Evaluation + Environmental Selection ---------
                for t = 1:Prob.T

                    prev_best(t) = min([population{t}.Objs]); 
                    offspring{t} = Algo.Evaluation(offspring{t}, Prob, t);

                    oldN = length(population{t});
                    population{t} = [population{t}, offspring{t}];

                    [~, rank] = sort([population{t}.Objs]);
                    survivor_idx = rank(1:Prob.N);
                for pos = 1:Prob.N
                      idx = survivor_idx(pos);

                      % Only offspring individuals have indices larger than oldN
                    if idx > oldN
                          k = idx - oldN;

                          % Rank contribution: the better the rank, the larger the contribution
                          rank_score = (Prob.N - pos + 1) / Prob.N;

                          if transferFlag{t}(k) == 0
                              Ng(t) = Ng(t) + 1;
                              Rg(t) = Rg(t) + rank_score;
                          elseif transferFlag{t}(k) == 1
                              Nb(t) = Nb(t) + 1;
                              Rb(t) = Rb(t) + rank_score;
                          elseif transferFlag{t}(k) == 2
                              Nm(t) = Nm(t) + 1;
                              Rm(t) = Rm(t) + rank_score;
                          end
                    end
                end

                    population{t} = population{t}(survivor_idx);
                end


                % --------- Update rho/delta every Gap generations ---------
                if mod(Algo.Gen, Algo.Gap) == 0

                    for t = 1:Prob.T
                        rho_obs = (Nb(t) ) / (Nb(t) + Nm(t));
                        rho(t) = (1 - 0.2) * rho(t) + 0.2 * rho_obs;
                        rho(t) = min(max(rho(t), Algo.ParaMin), Algo.ParaMax);
                    end

                     for t = 1:Prob.T
                        aux_idx = [1:t-1, t+1:Prob.T];
                        valid_idx = aux_idx(matrix_R(t, aux_idx) > 0);
                        score_aux = matrix_R(t, valid_idx);
                        p_t = score_aux / (sum(score_aux) + 1e-12);
                        S_t = -sum(p_t .* log(p_t + 1e-12)) / log(numel(valid_idx));


                        den_delta = Rg(t) + Rb(t) + Rm(t);

                        if den_delta > 1e-12
                            delta_obs = Rg(t) / den_delta;
                        else
                            delta_obs = delta(t);
                        end

                        delta(t) = (1 - 0.5) * S_t + 0.5 * delta_obs;

                        delta(t) = min(max(delta(t), Algo.ParaMin), Algo.ParaMax);

     
                    end
                    Ng(:) = 0;
                    Nb(:) = 0;
                    Nm(:) = 0;
                    
                end

            end
        end

        function [offspring, transferFlag] = Generation(Algo, Prob, population, matrix_R, v_cur, rho, delta)
            offspring    = cell(1, length(population));
            transferFlag = cell(1, length(population));

            for t = 1:Prob.T
                shiftDir_t(t) = v_cur(t);
            end

            for t = 1:length(population)
                popSize = length(population{t});
                indorder = randperm(popSize);
                transferFlag{t} = zeros(1, popSize);

                % The auxiliary task for the SSEI strategy
                [~, rt] = max(matrix_R(t, :));
                sourcePop = population{rt};
                [~, idx_s] = min(sourcePop.Objs);
                best_s = sourcePop(idx_s).Dec;


                % Auxiliary tasks and curvature information for the MGMCE strategy**
                knowledge_task_num = Algo.KTN;
                [~, index] = sort(matrix_R(t, :), 'descend');
                knowledge_task_num = min(knowledge_task_num, numel(index));
                index(index == t) = [];
                ass_tasks = index(1:knowledge_task_num);
                shiftDirs = cell(1, knowledge_task_num);
                for j = 1:knowledge_task_num
                    rt = ass_tasks(j);
                    srcPop = population{rt};
                    [~, idx_s] = min([srcPop.Objs]);
                    centerDecs(j, :) = srcPop(idx_s).Dec;
                    shiftDirs(j) = v_cur(rt);
                end


                % The mean, covariance, and elite scale of the target population elite solutions
                targetPop = population{t};
                [~, sort_idx] = sort([targetPop.Objs]);
                elite_num = max(2, ceil(0.2 * numel(targetPop)));
                elitePop = targetPop(sort_idx(1:elite_num));
                eliteDecs = vertcat(elitePop.Dec);
                dim_t = size(eliteDecs, 2);

                mu_e = size(eliteDecs, 1);
                weights = log(mu_e + 0.5) - log(1:mu_e);
                weights = weights / (sum(weights) + 1e-12);
                mu_t = weights * eliteDecs; 
                Y = eliteDecs - mu_t;
                Sigma_elite = Y' * diag(weights) * Y; 
                dist = sqrt(sum(Y.^2, 2));      
                elite_scale = weights * dist;  


                % The dominant direction of the target population elite set
                X = eliteDecs - mu_t;
                [~, ~, V] = svd(X, 'econ');
                mainDir = V(:, 1)';
                [~, i_best] = min([population{t}.Objs]);
                archive = population{t}(i_best).Dec;
                v_euc = best_s - archive;
                v_euc = v_euc / (norm(v_euc) + 1e-12);
                temp_archive = archive;

                
                count = 1;
                for i = 1:ceil(popSize / 2)
                    p1 = indorder(i);
                    p2 = indorder(i + fix(popSize / 2));
                    offspring{t}(count)     = population{t}(p1);
                    offspring{t}(count + 1) = population{t}(p2);

                    if rand() < delta(t)
                        transferFlag{t}(count)     = 0;
                        transferFlag{t}(count + 1) = 0;

                        [offspring{t}(count).Dec, offspring{t}(count + 1).Dec] = ...
                            GA_Crossover(population{t}(p1).Dec, population{t}(p2).Dec, Algo.MuC);

                        offspring{t}(count).Dec     = GA_Mutation(offspring{t}(count).Dec, Algo.MuM);
                        offspring{t}(count + 1).Dec = GA_Mutation(offspring{t}(count + 1).Dec, Algo.MuM);

                    else

                        if rand() < rho(t)

                            % SSEI strategy
                            transferFlag{t}(count)     = 1;
                            transferFlag{t}(count + 1) = 1;

                            v_kt = 0.5 * v_euc + 0.5 * shiftDir_t{t};
                            v_kt = v_kt / (norm(v_kt) + 1e-12);
                            u1 = temp_archive + 0.5 * v_kt;

                            v_euc2 = best_s - u1;
                            v_euc2 = v_euc2 / (norm(v_euc2) + 1e-12);
                            v_kt2 = 0.5 * v_euc2 + 0.5 * shiftDir_t{t};
                            v_kt2 = v_kt2 / (norm(v_kt2) + 1e-12);
                            u2 = u1 + 0.5 * v_kt2;

                            offspring{t}(count).Dec = u1;
                            offspring{t}(count + 1).Dec = u2;

                            temp_archive = u2;
                            v_euc = best_s - temp_archive;
                            v_euc = v_euc / (norm(v_euc) + 1e-12);


                        else
                            % MGMCE strategy
                            transferFlag{t}(count)     = 2;
                            transferFlag{t}(count + 1) = 2;

                            parent1 = population{t}(p1).Dec;
                            D_parent = pdist2(parent1, centerDecs);
                            [~, cluster_id] = min(D_parent, [], 2);
                            shiftDir = shiftDirs{cluster_id};
                            curvDir = shiftDir / (norm(shiftDir) + 1e-12);                   
                            dir_sim = abs(dot(curvDir, mainDir));
                            mu_shift = mu_t + Algo.theta * Algo.AlignDec(shiftDir, dim_t);
                            Sigma_new1 =  Sigma_elite + Algo.theta * (elite_scale^2);


                            parent2 = population{t}(p2).Dec;
                            D_parent2 = pdist2(parent2, centerDecs);
                            [~, cluster_id2] = min(D_parent2, [], 2);
                            shiftDir2 = shiftDirs{cluster_id2};
                            curvDir2 = shiftDir2 / (norm(shiftDir2) + 1e-12);
                            dir_sim2 = abs(dot(curvDir2, mainDir));
                            mu_shift2 = mu_t + Algo.theta * Algo.AlignDec(shiftDir2, dim_t);
                            Sigma_new2 = Sigma_elite + Algo.theta * (elite_scale^2);


                            if dir_sim > Algo.vartheta
                                offspring{t}(count).Dec = mvnrnd(mu_shift, Sigma_elite);
                            else
                                offspring{t}(count).Dec = mvnrnd(mu_t, Sigma_new1);
                            end

                            if dir_sim2 > Algo.vartheta
                                offspring{t}(count + 1).Dec = mvnrnd(mu_shift2, Sigma_elite);
                            else
                                offspring{t}(count + 1).Dec = mvnrnd(mu_t, Sigma_new2);
                            end


                        end
                    end

                    for x = count:count + 1
                        offspring{t}(x).Dec(offspring{t}(x).Dec > 1) = 1;
                        offspring{t}(x).Dec(offspring{t}(x).Dec < 0) = 0;
                    end

                    count = count + 2;
                end
            end
        end

        function [v_cur,R] = CalcTaskSimilarityHessian(Algo, population, d_star)
            T = length(population);
            R = zeros(T, T);
            Hs = cell(1, T);

            for t = 1:T
                pop_t = population{t};
                Hs{t} = Algo.EstimateHessianFromPopulation(pop_t);
                targetDim = length(pop_t(1).Dec);
                v_cur{t} = Algo.DominantNegativeCurvatureDirection(Hs{t}, targetDim);

            end

            for i = 1:T
                for j = 1:T
                    if i == j
                        R(i,j) = 0;
                    else
                        R(i,j) = Algo.HessianSimilarity(Hs{i}, Hs{j}, d_star);
                    end
                end
            end


        end

        function H = EstimateHessianFromPopulation(Algo, pop)
            Decs = vertcat(pop.Dec);
            Objs = vertcat(pop.Objs);

            if size(Objs, 2) > 1
                Objs = sum(Objs, 2);
            end
            Objs = Objs(:);
            d = size(Decs, 2);
            nPop = size(Decs, 1);
            [~, idx_best] = min(Objs);
            center = Decs(idx_best, :);
            Xall = Decs - center;
            nUse = min(nPop, max(65, 4 * d));
            [~, order] = sort(sum(Xall.^2, 2));
            order = order(1:nUse);

            X = Xall(order, :);
            y = Objs(order) - Objs(idx_best);
            y = y(:);

            dist = sqrt(sum(X.^2, 2));
            valid = dist > 1e-10;
            X = X(valid, :);
            y = y(valid);

            if size(X, 1) < 10
                H = eye(d) * 1e-8;
                return;
            end
            k = min([10, d, size(X, 1) - 1]);
            [~, ~, Vpca] = svd(X, 'econ');
            U = Vpca(:, 1:k);

            Z = X * U;
            Phi = Z;
            quadPairs = [];

            for a = 1:k
                for b = a:k
                    quadPairs = [quadPairs; a, b]; 
                    Phi(:, end + 1) = Z(:, a) .* Z(:, b);
                end
            end

            reg = 1e-5;
            beta = (Phi' * Phi + reg * eye(size(Phi, 2))) \ (Phi' * y);

            Hz = zeros(k, k);
            offset = k;

            for q = 1:size(quadPairs, 1)
                a = quadPairs(q, 1);
                b = quadPairs(q, 2);
                coef = beta(offset + q);

                if a == b
                    Hz(a, b) = 2 * coef;
                else
                    Hz(a, b) = coef;
                    Hz(b, a) = coef;
                end
            end
            Hz = (Hz + Hz') / 2;
            H = U * Hz * U';
            H = (H + H') / 2;
        end

        function sim = HessianSimilarity(Algo, Hi, Hj, d_star)
            Hi = (Hi + Hi') / 2;
            Hj = (Hj + Hj') / 2;
            [Vi, Di] = eig(Hi);
            [Vj, Dj] = eig(Hj);
            ei = diag(Di);
            ej = diag(Dj);
            [~, oi] = sort(abs(ei), 'descend');
            [~, oj] = sort(abs(ej), 'descend');
            ri = min([d_star, length(oi), size(Vi, 2)]);
            rj = min([d_star, length(oj), size(Vj, 2)]);

            if ri == 0 || rj == 0
                sim = 0;
                return;
            end

            Vi = Vi(:, oi(1:ri));
            Vj = Vj(:, oj(1:rj));
            ei = ei(oi(1:ri));
            ej = ej(oj(1:rj));

            weights = abs(ei);
            if sum(weights) <= 1e-12
                weights = ones(size(weights)) / numel(weights);
            else
                weights = weights / sum(weights);
            end

            align = zeros(ri, 1);
            for a = 1:ri
                vi = Algo.AlignDec(Vi(:, a), max(size(Vi, 1), size(Vj, 1)));
                bestAlign = 0;
                for b = 1:rj
                    vj = Algo.AlignDec(Vj(:, b), length(vi));
                    sameCurvature = sign(ei(a)) == sign(ej(b)) || abs(ei(a)) <= 1e-12 || abs(ej(b)) <= 1e-12;
                    if sameCurvature
                        bestAlign = max(bestAlign, abs(vi * vj') / (norm(vi) * norm(vj) + 1e-12));
                    end
                end
                align(a) = bestAlign;
            end

            sim = sum(weights(:) .* align(:));
            if ~isfinite(sim)
                sim = 0;
            end
            sim = min(max(sim, 0), 1);
        end

        
        function [H_noise, v_noise, c, theta] = EstimateNoisyCurvatureFromPopulation(Algo, pop, sigma)
            % Estimate Hessian and dominant curvature direction after adding noise
            % sigma: noise level, recommended values: 0.001, 0.005, 0.01, 0.05

            if nargin < 3
                sigma = 0.01;
            end

            Decs = vertcat(pop.Dec);
            Objs = vertcat(pop.Objs);

            if size(Objs, 2) > 1
                Objs = sum(Objs, 2);
            end
            Objs = Objs(:);

            d = size(Decs, 2);
            nPop = size(Decs, 1);

            % Original Hessian and original dominant curvature direction
            H_origin = Algo.EstimateHessianFromPopulation(pop);
            v_origin = Algo.DominantNegativeCurvatureDirection(H_origin, d);

            % Use the current best solution as the local center
            [~, idx_best] = min(Objs);
            center = Decs(idx_best, :);
            Xall = Decs - center;

            % Select neighboring samples according to the original population
            nUse = min(nPop, max(65, 4 * d));
            [~, order] = sort(sum(Xall.^2, 2));
            order = order(1:nUse);

            X = Xall(order, :);
            y = Objs(order) - Objs(idx_best);
            y = y(:);

            % Add Gaussian noise to neighboring perturbations
            noise = sigma * randn(size(X));
            X_noise = X + noise;

            % Remove samples with extremely small perturbations
            dist = sqrt(sum(X_noise.^2, 2));
            valid = dist > 1e-10;
            X_noise = X_noise(valid, :);
            y = y(valid);

            if size(X_noise, 1) < 10
                H_noise = eye(d) * 1e-8;
                v_noise = Algo.DominantNegativeCurvatureDirection(H_noise, d);
                c = abs(dot(v_origin, v_noise)) / ...
                    (norm(v_origin) * norm(v_noise) + 1e-12);
                theta = acos(min(max(c, 0), 1));
                return;
            end

            % Local PCA subspace
            k = min([10, d, size(X_noise, 1) - 1]);
            [~, ~, Vpca] = svd(X_noise, 'econ');
            U = Vpca(:, 1:k);

            Z = X_noise * U;

            % Quadratic regression in the low-dimensional subspace
            Phi = Z;
            quadPairs = [];

            for a = 1:k
                for b = a:k
                    quadPairs = [quadPairs; a, b]; %#ok<AGROW>
                    Phi(:, end + 1) = Z(:, a) .* Z(:, b); %#ok<AGROW>
                end
            end

            reg = 1e-5;
            beta = (Phi' * Phi + reg * eye(size(Phi, 2))) \ (Phi' * y);

            Hz = zeros(k, k);
            offset = k;

            for q = 1:size(quadPairs, 1)
                a = quadPairs(q, 1);
                b = quadPairs(q, 2);
                coef = beta(offset + q);

                if a == b
                    Hz(a, b) = 2 * coef;
                else
                    Hz(a, b) = coef;
                    Hz(b, a) = coef;
                end
            end

            Hz = (Hz + Hz') / 2;

            % Map the low-dimensional Hessian back to the original space
            H_noise = U * Hz * U';
            H_noise = (H_noise + H_noise') / 2;

            % Dominant curvature direction after noise perturbation
            v_noise = Algo.DominantNegativeCurvatureDirection(H_noise, d);

            % Directional consistency
            c = abs(dot(v_origin, v_noise)) / ...
                (norm(v_origin) * norm(v_noise) + 1e-12);
            c = min(max(c, 0), 1);

            % Angular deviation
            theta = acos(c);
        end


        function v = DominantNegativeCurvatureDirection(Algo, H, targetDim)
            if isempty(H) || any(isnan(H(:))) || any(isinf(H(:)))
                v = randn(1, targetDim);
                v = v / (norm(v) + 1e-12);
                return;
            end

            [V, D] = eig((H + H') / 2);
            eigvals = diag(D);
            neg_idx = find(eigvals < 0);

            if ~isempty(neg_idx)
                [~, local_idx] = max(abs(eigvals(neg_idx)));
                idx = neg_idx(local_idx);
            else
                [~, idx] = min(eigvals);
            end

            v = V(:, idx)';
            v = Algo.AlignDec(v, targetDim);
            v = v / (norm(v) + 1e-12);
        end

        function x = AlignDec(Algo, x, targetDim)
            x = x(:)';
            if length(x) > targetDim
                x = x(1:targetDim);
            elseif length(x) < targetDim
                x = [x, zeros(1, targetDim - length(x))];
            end
        end

    end
end