function [xy,par_est] = BayesOpt(target_fun,covariance_fun,acquis_fun,domain,fixed_par0,acquis_param,n_init,n_it,n_grid,plot_fun,log_color,acq_scale)
% BayesOpt: Function for Gaussian Process optimization in dD
%
% [xy, par_est] = BayesOpt(target_fun,covariance_fun,acquis_fun,domain,fixed_par0,acquis_param,n_init,n_it,n_grid,plot_fun,log_color,acq_scale)
%
% target_fnc - Objective function taking 1 d-dimensional input parameter.
% covariance_fun - The covariance function to be used for the Gaussian
%                       process. Either Matern or Gaussian
% acquis_fun - The acquisition function to be used. Should be one of:
%               LB: Lower Bound
%               EI: Expected Improvement
%               PI: Probability of Improvement
% acquis_param - Acquisition parameter (beta)
% domain - Domain of the function as a 2-by-d vector with min and max 
%          values for each dimension of the input.
% n_init - Number of initial points taken as
%          x_init = domain(1,:) + (domain(2,:)-domain(1,:)).*rand(n_init,size(domain,2));
% n_grid - Number of grid points in each dimension. Used for plotting in 1D
%           and 2D, and always for Kriging reconstructions.
% n_it - Number of iterations.

% nu - Fixed nu in the Matern covariance. If covariance_function is
%      Gaussian, then this can be set to anything.
% sigma2_eps - Evaluation noise (typically assumed to be small e.g. 1e-6)
% fixed_par0 = [ rho_x rho_y . . . sigma^2 nu sigma_ep^2 rho_cov] 
% parameters inputed as 0 are estimated while non-zeor values are fixed

% plotF - Plot optimization progress.
% log_color- shows color plots with log scale
% acq_scale- scales the acquisition function for visualization



% log_color - Plot function (and variance) values in log-scale in 2D
%
% Returns:
%  xy - matrix with all evaluation points and function values.
%  par_est - final parameter estimate for the Gaussian Process.

% TODO: what does the code do if nu is not fixed (not priority, we will only look at fixed nu)
% TODO: catching singular matrices (not prioriy, if you encounter issues
% increase the nugget)
% TODO: wasteful evaluations (more for discussion): sometimes the algorithm converges before the
% number of evaluations is finished. In that case it is just evaluating the
% same point and the algorithm should then stop (right now it does not)
% Note: regular initial points can be missguiding if it is in resonance
% with the function
% TODO: improve code efficiency for higher dimentions (common issue for
% Bayesian optimization). One proposal is to replace the grid search (which
% scales exponentially with dimenion) with randomized sampling with fixed
% number of points
% TODO: handel singular matrices. (this occures especially when the nr of
% initial points is too small)
%% domain dimension
d = length(domain(1,:));
if length(fixed_par0) ~= d+4
        error('fixed_par0 should have (4+ nr of dim) entries.');
end
rng(2)
%% Pick Covariance Function
switch covariance_fun
    case 'Matern'
        cov_func = @matern_covariance;
        par_est = [];
        % For matern we estimate sigma2 and anisotropy rho
        par_fixed = fixed_par0; %[zeros(1,d) 0 nu sigma2_eps 1]; 
    case 'Gaussian'
        cov_func = @gaussian_covariance;
        par_est = [];
        % For Gaussian we estimate sigma2 and anisotropy
        par_fixed = fixed_par0; %[zeros(1,d) 0 nu sigma2_eps 1]; 
end

%% Pick acquisition function
switch acquis_fun
    case 'LB'
        a = @LB;
        a_grid = @LB_grid;
    case 'EI'
        a = @EI;
        a_grid = @EI_grid;
    case 'PI'
        a = @PI;
        a_grid = @PI_grid;
end
%% Initial evals and grid points
% Initial points and grid
%sampling initial points based on domain
x_init = domain(1,:) + (domain(2,:)-domain(1,:)).*rand(n_init,size(domain,2));
% sample a grid for eval and plotting
X_grid = generate_grid(domain,n_grid);
%compute value of loss-function for initial x-values
xy = zeros(n_init+n_it,d+1);
j=n_init; % number of evaluated points
% matrix to save points and evalutations
xy(1:n_init,:) = [x_init target_fun(x_init)];

%% Initial Plots

if plot_fun & (d==1)
    figure;
    ax = axes;
    plot(ax,X_grid,target_fun(X_grid))
    hold on
    scatter(ax,x_init(:,1),target_fun(x_init(:,1)),'filled','r','*')
    hold off
    title(ax, '1');
end
% we only do this for test functions in 2D
if plot_fun & (d==2)
    %compute value of loss-function on grid for illustration
    f_grid = target_fun(X_grid);
    %illustrate the function in 2D
    subplot(221)
    %imagesc( reshape(f_grid,[n_grid n_grid]))
    imagesc(X_grid(:,1),X_grid(:,2),reshape(f_grid,[n_grid n_grid]))
    hold on
    scatter(x_init(:,1),x_init(:,2),4,'filled','w','o')
    colorbar
    if log_color; set(gca,'ColorScale','log'); end
    title('Test Function')
    hold off
end


for k=1:n_it
    %fit GP with fixed nu and sigma2_eps
    par_est = covest_ml(cov_func,xy(1:j,:), par_fixed, par_est);

    %reconstruction on grid: only need for plotting and initial minim guess
    [y_rec,y_var,S_kk,iS_y] = reconstruct(cov_func,xy(1:j,:), X_grid, par_est);
    % find the minimum of the acquisition funciton on the grid (this step
    % help us avoid local minima in the optimizaiton of the acquisition
    % funciton
    % acquisition evauated on a grid
    y_acq = a_grid(y_rec,y_var,min(xy(1:j,end)),acquis_param);

    [~,Imin]=min(y_acq);

    x_min0 = X_grid(Imin,:);

    

    %find minima, using grid minima as starting point; off grid
    x_min = fminsearch(@(x) a(cov_func,x,domain,acquis_param,xy(1:j,:),par_est,S_kk,iS_y,d), x_min0);

    if d==1
            if plot_fun
            plot(ax,X_grid, target_fun(X_grid),'-b', ...
            xy(1:j,1),xy(1:j,end),'*r', ...
            X_grid,y_rec,'-k', ...
            X_grid,y_rec+[-2 2].*sqrt(y_var),':k',...
            X_grid,acq_scale*y_acq,'--r')
            xline(x_min,'Color', [1, 0.5, 0])
            title(k)
            if plot_fun==1
                pause(0.4)
            else
            pause
            end
            end
    elseif d==2
    if plot_fun
        subplot(222) 
        imagesc(X_grid(:,1),X_grid(:,2),reshape(y_rec,[n_grid n_grid]))
        title('Reconstruction')
        axis image;
        colorbar;
        if log_color; set(gca,'ColorScale','log'); end

        subplot(223)
        imagesc(X_grid(:,1),X_grid(:,2),reshape(acq_scale.*y_acq,[n_grid n_grid]))
        hold on
        scatter(x_min(1),x_min(2),30,'filled','r','o')
        title('Acquisition Function')
        colorbar;
        if log_color; set(gca,'ColorScale','log'); end
        hold off
        axis image;

        subplot(224) 
        imagesc(X_grid(:,1),X_grid(:,2),reshape(sqrt(y_var),[n_grid n_grid]))
        hold on
        scatter(xy(1:j,1),xy(1:j,2),20,'w','o')
        hold off
        colorbar;
        if log_color; set(gca,'ColorScale','log'); end
        title(['Uncertainty (Iteration: ', num2str(k), ')']);
        axis image;
        if plot_fun==1
            pause(0.2)
        else
          pause
        end
    end
    end

    %updated function evaluations
    j = j+1;
    xy(j,:) = [x_min target_fun(x_min)];
end
% plot the estimated minimum
if d==2
    subplot(221)
    [m,I] = min(xy(:,d+1));
    x_min_est = (xy(I,1:d));
    %imagesc( reshape(f_grid,[n_grid n_grid]))
    imagesc(X_grid(:,1),X_grid(:,2),reshape(f_grid,[n_grid n_grid]))
    hold on
    scatter(x_init(:,1),x_init(:,2),10,'filled','w','o')
    scatter(x_min_est(2),x_min_est(1),40,'filled','w','o')
    colorbar
    axis image;
    if log_color; set(gca,'ColorScale','log'); end
    title('Test Function')
    hold off

end
%title('')
%legend('Objective funciton','Evaluations','GP','95 % CI','','Acquisition function','Next evaluation')

% FUNCTIONS
%% Distance matrix given rho
function D=distance_matrix(u,v,rho)
D = zeros(size(u,1),size(v,1)); %allocate zero matrix

for k=1:size(u,2) %loop over dimensions, matlab is index 1 to k

    D = D + ((u(:,k)- v(:,k)')/rho(k)).^2;
end
D = sqrt(D);


%% Matern covariance
function r=matern_covariance(dist,sigma2,nu,rho_cov)
%compute distances with effect from nu
dist_nu = sqrt(2*nu)*dist;


%deal with 0 values
dpos = (dist>0);
r = zeros(size(dist));
% cov for distance = 0 should be sigma2
r(~dpos) = sigma2;

% Special cases of nu={0.5, 1.5, and 2.5}
switch nu
  case 0.5
    d = dist(dpos)./rho_cov;
    B = sigma2.*exp(-d);
  case 1.5
    d = sqrt(3)*dist(dpos)./rho_cov;
    B = sigma2.*(1+d).*exp(-d);
  case 2.5
    d = sqrt(5)*dist(dpos)./rho_cov;
    B = sigma2.*(1+d+(d.^2)/3).*exp(-d);
    otherwise
    % not sure where this expression comes from. add /rho_cov to it
    % Less sensitive to large distances:
    % TODO: add rho_cov to this expression (kappa)
    B = log(besselk(nu,dist_nu(dpos)));
    B = exp(log(sigma2)-gammaln(nu)-(nu-1)*log(2) + ...
      nu.*log(dist_nu(dpos)) + B);
    %sanity check, large nu values will need replacement for small dunique
    %values
    if any(isinf(B))
      dist = dist(dpos)./rho_cov;
      B(isinf(B)) = sigma2*exp(-0.5*dist(isinf(B)).^2);
    end
end
r(dpos) = B;

%%
function r=gaussian_covariance(dist,sigma2,nu,rho_cov)

%deal with 0 values
dpos = (dist>0);
r = zeros(size(dist));
% cov for distance = 0 should be sigma2
r(~dpos) = sigma2;
r(dpos) = sigma2.*exp(-2.*((dist(dpos)./rho_cov).^2));


%% Maximum Likelihood Estimation
function par_est = covest_ml(cov_func,xy, par_fixed, par_start)
d = length(xy(1,:))-1;
%initial parameters
if isempty(par_start)
  par_start = zeros(size(par_fixed));
  %rough estimate of field variance
  par_start(d+1) = var(xy(:,d+1));
  %and nugget
  par_start(d+3) = par_start(d+1)/5;
  %rough estimate of range
  D = distance_matrix(xy(:,1:d),xy(:,1:d),ones(1,d));
  par_start(1:d) = (max(max(D))/5)*ones(1,d);
  par_start(d+4) = 1;% axis anisotropy takes care of this factor
  % if there was no axis anisotropy parameter use mean(D(:))/2
else
  par_start = reshape(par_start,1,[]);
end

%fixed parameters? if none then estimate all
if isempty(par_fixed)
  par_fixed = zeros(size(par_start));
end

%define loss function that accounts for fixed parameters
neg_log_like = @(par) covest_ml_loss(cov_func,xy, exp(par), par_fixed);
%minimize loss function (negative log-likelihood)
par = fminsearch(neg_log_like, log(par_start(par_fixed==0)));

%extract estimated parameters
par_fixed(par_fixed==0) = exp(par);
par_est=par_fixed;


%% loss function for the covariance function
function nlogf = covest_ml_loss(cov_func,xy, par, par0)
d = length(xy(1,:))-1;
%merge parameters
par0(par0==0) = par;
par=par0;
%extract parameters
rho = par(1:d);
sigma2 = par(d+1);
nu = par(d+2);
sigma2_eps = par(d+3);
rho2_gauss = par(d+4);
%compute distance matrix
D = distance_matrix(xy(:,1:d),xy(:,1:d),rho);
%compute covariance function
Sigma_yy = cov_func(D,sigma2,nu,rho2_gauss) + sigma2_eps*eye(size(D));
%compute Choleskey factor
[R,p] = chol(Sigma_yy);
%Choleskey fail -> return large value
if p~=0
	nlogf = realmax/2;
	return; 
end
nlogf = sum(log(diag(R))) + 0.5*sum((xy(:,d+1)'/R).^2);


%% Function computing reconstructions
function [y_rec,y_var,S_kk,iS_y] = reconstruct(cov_func,xy,x0,par,S_kk, iS_y)
d = length(xy(1,:))-1;
if nargin<5 || isempty(S_kk)
    S_kk = distance_matrix(xy(:,1:d),xy(:,1:d),par(1:d));% distance matrix
    S_kk = cov_func(S_kk, par(d+1), par(d+2),par(d+4)); % cov matrix S_kk
    S_kk = S_kk + par(d+3)*eye(size(S_kk)); % add the nuget
end
if nargin<6 || isempty(iS_y)
    iS_y = S_kk\xy(:,d+1);
end

S_uk = distance_matrix(x0,xy(:,1:d),par(1:d));
S_uk = cov_func(S_uk, par(d+1), par(d+2),par(d+4));

y_rec = S_uk * iS_y; %S_uk inv(S_kk) (y_k-mu_k)
y_var = par(d+1)+par(d+3) - sum((S_uk/S_kk).*S_uk,2);
y_var = max(y_var,0);

%% LB: lower bound acquisition function
function l=LB(cov_func,x, x_bnd, beta, xy, par, S_kk, iS_y,d)
  %check bounds
  for k=1:d

      if x(k)<x_bnd(1,1) || x_bnd(2,1)<x(k)
          l = realmax/2;
          return
      end
  end
  [y_rec,y_var] = reconstruct(cov_func,xy,x,par,S_kk,iS_y);
  l = y_rec-beta*sqrt(y_var);

%%
function l=LB_grid(y_rec,y_var, fmin,beta)
  l = y_rec-beta.*sqrt(y_var);
%% EI: expected improvement acquisition function
function l =EI(cov_func,x, x_bnd, beta, xy, par, S_kk, iS_y,d)

  %check bounds
  for k=1:d
      if x(k)<x_bnd(1,1) || x_bnd(2,1)<x(k)
          l = realmax/2;
          return
      end
  end
  [y_rec,y_var] = reconstruct(cov_func,xy,x,par,S_kk,iS_y);
  y_sig = sqrt(y_var);
  impr = (min(xy(:,end))- y_rec-beta);% improvement
  z= impr/y_sig;
  % multiply the expected improvement by (-1) to get a minimization problem
  l = -(impr*normcdf(z)+ y_sig * normpdf(z));
%% 
function l =EI_grid(y_rec,y_var,fmin,beta)
      y_sig = sqrt(y_var);
      impr = (fmin- y_rec-beta);% improvement
      z= impr./y_sig;
      % multiply the expected improvement by (-1) to get a minimization problem
      l = -(impr.*normcdf(z)+ y_sig .* normpdf(z));

%% PI: probability of improvement acquisition function
function l =PI(cov_func,x, x_bnd, beta, xy, par, S_kk, iS_y,d)

  %check bounds
  for k=1:d
      if x(k)<x_bnd(1,1) || x_bnd(2,1)<x(k)
          l = realmax/2;
          return
      end
  end
  [y_rec,y_var] = reconstruct(cov_func,xy,x,par,S_kk,iS_y);
  z = (min(xy(:,end))- y_rec-beta)./sqrt(y_var);% improvement
  % multiply the PI by (-1) to get a minimization problem
  l = -(normcdf(z));

%% PI: only a grid, given y_rec, y_var
function l =PI_grid(y_rec,y_var,fmin,beta)
  z = (fmin- y_rec-beta)./sqrt(y_var);% improvement
  % multiply the PI by (-1) to get a minimization problem
  l = -(normcdf(z));
