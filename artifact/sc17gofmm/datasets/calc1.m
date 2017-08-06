%% 
%%  @brief  These matlab scripts create 18 spd matrices.
%%
%%  @author George Biros (The Univeristy of Texas at Austin)
%%



classdef calc1 < calc 
% SETUP of algebraic and derivative operations for 2D periodic fields using
% equispaced points
% derivatives of periodic functions deifned in [-pi,pi]^2
% support for scalar, vector, and tensor functions. 
% 
% use doc calc2 for a summary of the documentations

methods

%/******************************************************/
function xo=mygpuArray(o,xin)
% this is used to convert all calculations in GPU
% as of 2013/05/08, I've tried, but not speed ups have been observed. 
 xo=xin;     
end

%/******************************************************/
function o = calc1(N)
% constructor for parent class
  o@calc(N);
end

%/******************************************************/
function setupIK(o)
% setup differentiation kernel for in 2D
Nx=o.N(1);

kx=o.dk(Nx)';

tmp{1}=o.mygpuArray(1i*kx); 

o.IK=o.crv(tmp);
end


%/* ************************************************** */
 function val = S(o,x)
% convert matlab array X to scalar field
% val = S(o,x)
  val=x(:); 
end

%/* ************************************************** */
function val = V(o,x)
% convert matlab array to vector field
 val =  x
end

%/* ************************************************** */
function v = crv(o,Vs)
% creates vector field
% v = crv(o,Vs)
%
% input:
% Vs should be a struct with three components correspoding to each
%   vector components
% Vs{1} is the x-component, Vs{2} the y-component etc.

    v = Vs{1};
end

%/******************************************************/
function T = crT(o, Vs)
% create tensor  fireld
% T = crT(o,Vs)
%
% input:
% Vs should be a struct with two vectors correspoding to each
%   tensor column
% Vs{1} is the first 'column', Vs{2} the second 'column'
%   'column' is a vector field
  T=Vs{1};
end


%/******************************************************/
function T = trans(o, Tin)
% transpose of a tensor
  T=Tin;
end


%/* ************************************************** */
function vj = gvc(o,v,j)
% returns the j the component of vector v
%
%  vj = o.gvc(v,j)
   vj = v;
end

%/* ************************************************** */
function Tj = gTc(o,T,j)
% returns the jth column of the tensor T
%
Tj = T;
end


%/* ************************************************** */
function vo = svc(o,v,j,f)
% sets  the j the component of vector v to field f
%
  vo = f;
end

%/* ************************************************** */
function Tj = sTc(o,T,j,v)
% sets the jth column of the tensor field T to vector field f
%
Tj = v;
end


%/* ************************************************** */
function [curlV] = curl(o, v)
% return the curl of a vector 'v'
% [curlV] = curl(o, v)
    curlv = 0*v;
    warning('undefined');
end

%/******************************************************/
function [gv] = gradt(o,f)
gv = 0*f;
warning('undefined');
end
    

%/* ************************************************** */
function [DivT] = div_T(o,T)
% return the divergence of a tensor T
% [DivT] = div_T(o,T)
    DivT=0;
    warning('undefined');
end    

%/* ************************************************** */
function X=regulargrid(o)
% return the coordinates of all the points of a regular grid in [-pi, pi].
%
% X=regulargrid(o)
% X is a calc1 vector object.
%
% x= -pi: hx : pi-hx,  hx = 2*pi/Nx

  H=2*pi./o.N;
  X=[-pi:H:pi-H]';
end


%/* ************************************************** */
function tensor = outer(o, vec1, vec2)
  tensor = vec1.*vec2;
end

%/* ************************************************** */
function f=ones(o)
% create a  scalar field = 1
  f=ones(o.N(1),1);
end

%/* ************************************************** */
function filter = create_box_filter(o,percent)
% create a frequency filter in wich only the frequencies that are less thout round(percent*N/2) are left
%  where N is the number of points per dimension
  filter = o.cutoff(o.N(1),percent);
end
    

%/* ************************************************** */
function use_hou_filtering(o)
% this routine is used to filter all first derivatives using Thomas Hou's smooth filtering of
% high frequencies.
% only curl, grad, div will be  filtered. Laplacian and invlaplacian will _not_ be filtered
  hf  = @(n) exp(-36*(2* o.dk(n)/n).^36).*o.dk(n);
  
  o.IK=1i*hf(o.N(1));
end


end %methods
end %class


  
    

