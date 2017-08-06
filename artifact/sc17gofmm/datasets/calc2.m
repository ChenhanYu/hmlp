%% 
%%  @brief  These matlab scripts create 18 spd matrices.
%%
%%  @author George Biros (The Univeristy of Texas at Austin)
%%


classdef calc2 < calc 
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
% xo=gpuArray(xin);  % uncomment this line to enable GPU
end

%/******************************************************/
function o = calc2(N)
% constructor for parent class
  o@calc(N);
end

%/******************************************************/
function setupIK(o)
% setup differentiation kernel for in 2D
Nx=o.N(1); Ny=o.N(2);

[kx,ky]=(meshgrid(o.dk(Nx), o.dk(Ny)));

tmp{1}=o.mygpuArray(1i*kx); 
tmp{2}=o.mygpuArray(1i*ky); 

o.IK=o.crv(tmp);
end


%/* ************************************************** */
 function val = S(o,x)
% convert matlab array X to scalar field
% val = S(o,x)
  val=reshape(x,[o.N(2),o.N(1)]); % column  to vector
end

%/* ************************************************** */
function val = V(o,x)
% convert matlab array to vector field
 val =  reshape(x,[o.N(2),o.N(1),o.dim]); % column to scalar
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

    v = o.mygpuArray(zeros([size(Vs{1}), o.dim]));
    for j=1:o.dim
        v(:,:,j) = Vs{j};
    end
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

  T = o.mygpuArray(zeros([size(Vs{1}), o.dim]));
  for j=1:o.dim
    T(:,:,:,j) = Vs{j};
  end
end


%/******************************************************/
function T = trans(o, Tin)
% transpose of a tensor
  T=0*Tin;
  for i=1:o.dim
    for j=1:o.dim
      T(:,:,i,j)=Tin(:,:,:,j,i);
    end
  end
end


%/* ************************************************** */
function vj = gvc(o,v,j)
% returns the j the component of vector v
%
%  vj = o.gvc(v,j)
    vj = v(:,:,j);
end

%/* ************************************************** */
function Tj = gTc(o,T,j)
% returns the jth column of the tensor T
%
Tj =  T(:,:,:,j);
end


%/* ************************************************** */
function vo = svc(o,v,j,f)
% sets  the j the component of vector v to field f
%
  vo = v;
  vo(:,:,j) = f;
end

%/* ************************************************** */
function Tj = sTc(o,T,j,v)
% sets the jth column of the tensor field T to vector field f
%
Tj = T;
Tj(:,:,:,j) = v;
end


%/* ************************************************** */
function [curlV] = curl(o, v)
% return the curl of a vector 'v'
% [curlV] = curl(o, v)
    curlv = 0*v;
    
    gv_x = o.grad( o.gvc(v,1));
    gv_y = o.grad( o.gvc(v,2));

    curlV=o.gvc(gv_y, 1) - o.gvc( gv_x, 2);
end

%/******************************************************/
function [gv] = gradt(o,f)
% return the [-df/dy   df /dx ], where f is a scalar field
gv = o.grad(f);
tmp=gv;
gv = -o.svc(gv,1,o.gvc(tmp,2));
gv =  o.svc(gv,2,o.gvc(tmp,1));
end
    

%/* ************************************************** */
function [DivT] = div_T(o,T)
% return the divergence of a tensor T
% [DivT] = div_T(o,T)
    DivT=0*o.const_vec;
    for j=1:o.dim
        Tj = reshape(T(:,:,j,:), [size(T,1),size(T,2),o.dim]);
        DivT(:,:,j) = o.div(Tj);
    end
end    

%/* ************************************************** */
function X=regulargrid(o)
% return the coordinates of all the points of a regular grid in [-pi, pi].
%
% X=regulargrid(o)
% X is a calc2 vector object.
%
% x= -pi: hx : pi-hx,  hx = 2*pi/Nx
% y= -pi: hy : pi-hy,  hy = 2*pi/Ny
% generated with matlab's meshgrid 

  H=2*pi./o.N;
  for j=1:o.dim, X{j}=[-pi:H(j):pi-H(j)]; end
  [x,y] = meshgrid(X{1},X{2});

  tmp{1}=o.mygpuArray(x); 
  tmp{2}=o.mygpuArray(y); 

  X = o.crv( tmp );
end


%/* ************************************************** */
function tensor = outer(o, vec1, vec2)
% pointwise outer product of two tensor fields
% tensor = outer(o, vec1, vec2)
  tensor = zeros( size(vec1,1), size(vec1,2), size(vec1,3), o.dim, o.dim);
  for j=1:o.dim
      for k=1:o.dim
         tensor(:,:,j,k) = o.gvc(vec1, j).*o.gvc(vec2, k);
      end
  end
end

%/* ************************************************** */
function f=ones(o)
% create a  scalar field = 1
  f=o.mygpuArray(ones([o.N(2); o.N(1);]'));
end

%/* ************************************************** */
function filter = create_box_filter(o,percent)
% create a frequency filter in wich only the frequencies that are less thout round(percent*N/2) are left
%  where N is the number of points per dimension
 [hx,hy]=meshgrid( o.cutoff(o.N(1),percent), o.cutoff(o.N(2),percent));
 filter = hx.*hy;
end
    

%/* ************************************************** */
function use_hou_filtering(o)
% this routine is used to filter all first derivatives using Thomas Hou's smooth filtering of
% high frequencies.
% only curl, grad, div will be  filtered. Laplacian and invlaplacian will _not_ be filtered
  hf  = @(n) exp(-36*(2* o.dk(n)/n).^36).*o.dk(n);

  [hx,hy]=meshgrid( hf(o.N(1)),hf(o.N(2)));

  tmp{1}=1i*hx; tmp{2}=1i*hy;
  o.IK=o.crv(tmp);
end

%/* ************************************************** */
function use_twothirds_filtering(o)
% this routine is used to filter all first derivatives filtering out the last 1/3 of the spectrum
% only curl, grad, div will be  filtered. Laplacian and invlaplacian will _not_ be filtered
  ts =@(n) (abs( o.dk(n) ) < floor(n/2 * 2/3)).* o.dk(n);

  [hx,hy]=meshgrid( ts(o.N(1)),ts(o.N(2)));
  tmp{1}=1i*hx; tmp{2}=1i*hy; 
  o.IK=o.crv(tmp);
end


end %methods
end %class


  
    

