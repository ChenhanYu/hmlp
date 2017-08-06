%% 
%%  @brief  These matlab scripts create 18 spd matrices.
%%
%%  @author George Biros (The Univeristy of Texas at Austin)
%%



function [vals,gradvals] = calc2_functions(o, type, varargin)
%function vals = calc3_functions(o, type)
%
% vals: scalar or vector field values
% gradvals  = gradient (analytic)
%
% o:  calc2 object
% type: type of fuctions
%    - gaussian,  varargin, 'mean',[x,y,z]; 'sigma', scalar_sigma
%    - trigonomzetric
vals=[]; gradvals = [];

X=o.regulargrid;
x = o.gvc(X,1);
y = o.gvc(X,2);

switch type
%/* ************************************************** */
  case 'trigonometric'
    freq = [1 0];
    for k=1:2:length(varargin), % overwrite default parameter
      eval([varargin{k},'=varargin{',int2str(k+1),'};']);
    end;

    K = o.const_vec(freq);
    vals = exp ( 1i * o.inner(X, K ) );
    gradvals = 1i*o.scalevec(K,vals);

%/* ************************************************** */
  case 'gaussian'
    mean = [0,0];
    sigma = pi/4;
    for k=1:2:length(varargin), % overwrite default parameter
      eval([varargin{k},'=varargin{',int2str(k+1),'};']);
    end;
    
    C = o.const_vec(mean);
    r = X-C;
    vals = 1/sigma * exp( - 1/sigma^2 * o.inner(r,r).^16 );
    gradvals =  o.scalevec( r, -2/sigma^2 * vals);
    
  otherwise
    error('type not found');
end

