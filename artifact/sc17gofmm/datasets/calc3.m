%% 
%%  @brief  These matlab scripts create 18 spd matrices.
%%
%%  @author George Biros (The Univeristy of Texas at Austin)
%%



classdef calc3 < calc 
% SETUP of algebraic and derivative operations for 3D periodic fields using
% equispaced points
% derivatives of periodic functions deifned in Omega : =[-pi,pi]^3
% support for scalar, vector, and tensor functions. 
% 
% see: doc calc3 and doc calc

methods

function xo=mygpuArray(o,xin)
 xo=xin;
% xo=gpuArray(xin); 
end

%/******************************************************/
function o = calc3(N)
% constructor for derivative in [-pi,pi]
  o@calc(N);
end

%/******************************************************/
function setupIK(o)
Nx=o.N(1); Ny=o.N(2); Nz=o.N(3);

[kx,ky,kz]=(meshgrid(o.dk(Nx), o.dk(Ny), o.dk(Nz)));

%tmp{1}=1i*kx; tmp{2}=1i*ky; tmp{3}=1i*kz;
tmp{1}=o.mygpuArray(1i*kx); 
tmp{2}=o.mygpuArray(1i*ky); 
tmp{3}=o.mygpuArray(1i*kz);

o.IK=o.crv(tmp);
end


%/* ************************************************** */
 function val = S(o,x)
% convert matlab array X to scalar field
% val = S(o,x)
  val=reshape(x,[o.N(2),o.N(1),o.N(3)]); % column  to vector
end

%/* ************************************************** */
function val = V(o,x)
% convert matlab array to vector field
 val =  reshape(x,[o.N(2),o.N(1),o.N(3),o.dim]); % column to scalar
end

%/* ************************************************** */
function v = crv(o,Vs)
% creates vector with proper dimensions.
% v = crv(o,Vs)
%
% input:
% Vs should be a struct with three components correspoding to each
%   vector components
% Vs{1} is the x-component, Vs{2} the y-component etc.

    v = o.mygpuArray(zeros([size(Vs{1}), o.dim]));
    for j=1:o.dim
        v(:,:,:,j) = Vs{j};
    end
end

%/******************************************************/
function T = crT(o, Vs)
% create tensor 
% T = crT(o,Vs)
%
% input:
% Vs should be a struct with three vectors correspoding to each
%   tensor columns
% Vs{1} is the onent, Vs{2} the y-component etc.

  T = o.mygpuArray(zeros([size(Vs{1}), o.dim]));
  for j=1:o.dim
    T(:,:,:,:,j) = Vs{j};
  end
end


%/******************************************************/
function T = trans(o, Tin)
% transpose of a tensor
  T=0*Tin;
  for i=1:o.dim
    for j=1:o.dim
      T(:,:,:,i,j)=Tin(:,:,:,j,i);
    end
  end
end


%/* ************************************************** */
function vj = gvc(o,v,j)
% returns  the j the component of vector v
%
%  vj = o.gvc(v,j)
    vj = v(:,:,:,j);
end

%/* ************************************************** */
function Tj = gTc(o,T,j)
% returns the jth column of the tensor field T
%
Tj =  T(:,:,:,:,j);
end


%/* ************************************************** */
function vo = svc(o,v,j,f)
% sets  the j the component of vector v to field f
%
  vo = v;
  vo(:,:,:,j) = f;
end

%/* ************************************************** */
function Tj = sTc(o,T,j,v)
% sets the jth column of the tensor field T to vector field f
%
Tj = T;
Tj(:,:,:,:,j) = v;
end


%/* ************************************************** */
function [curlV] = curl(o, v)
% return the curl of a vector 'v'
% [curlV] = curl(o, v)
    curlv = 0*v;
    
    gv_x = o.grad( o.gvc(v,1));
    gv_y = o.grad( o.gvc(v,2));
    gv_z = o.grad( o.gvc(v,3));

    tmp{1}=o.gvc(gv_z, 2) - o.gvc( gv_y, 3);
    tmp{2}=o.gvc(gv_x, 3) - o.gvc( gv_z, 1);
    tmp{3}=o.gvc(gv_y, 1) - o.gvc( gv_x, 2);
    
    curlV = o.crv( tmp );
end
    

%/* ************************************************** */
function [DivT] = div_T(o,T)
% return the divergence of a tensor T
% [DivT] = div_T(o,T)
    DivT=0*o.const_vec;
    for j=1:o.dim
        Tj = reshape(T(:,:,:,j,:), [size(T,1),size(T,2),size(T,3),o.dim]);
        DivT(:,:,:,j) = o.div(Tj);
    end
end    

%/* ************************************************** */
function X=regulargrid(o)
% return the coordinates of all the points of a regular grid in [-pi, pi].
%
% X=regulargrid(o)
% X is a calc3 vector object.
%
% x= -pi: hx : pi-hx,  hx = 2*pi/Nx
% y= -pi: hy : pi-hy,  hy = 2*pi/Ny
% z= -pi: hz : pi-hz,  hz = 2*pi/Nz
% generated with matlab's meshgrid 

  H=2*pi./o.N;
  for j=1:o.dim, X{j}=[-pi:H(j):pi-H(j)]; end
  [x,y,z] = meshgrid(X{1},X{2},X{3});

  tmp{1}=o.mygpuArray(x); 
  tmp{2}=o.mygpuArray(y); 
  tmp{3}=o.mygpuArray(z);

  X = o.crv( tmp );
end


%/* ************************************************** */
function tensor = outer(o, vec1, vec2)
% pointwise outer product of two tensor fields
% tensor = outer(o, vec1, vec2)
  tensor = zeros( size(vec1,1), size(vec1,2), size(vec1,3), o.dim, o.dim);
  for j=1:o.dim
      for k=1:o.dim
         tensor(:,:,:,j,k) = o.gvc(vec1, j).*o.gvc(vec2, k);
      end
  end
end

%/* ************************************************** */
function f=ones(o)
% return  scalar field = 1
  f=o.mygpuArray(ones([o.N(2); o.N(1); o.N(3)]'));
end

%/* ************************************************** */
function filter = create_box_filter(o,percent)
% create a frequency filter in wich only the frequencies that are less thout round(percent*N/2) are left
%  where N is the number of points per dimension
 [hx,hy,hz]=meshgrid( o.cutoff(o.N(1),percent), o.cutoff(o.N(2),percent), o.cutoff(o.N(3),percent));
 filter = hx.*hy.*hz;
end
    

%/* ************************************************** */
function use_hou_filtering(o)
% this routine is used to filter all first derivatives using Thomas Hou's smooth filtering of
% high frequencies.
% functions curl, grad, div are changed. laplacian and invlaplacian are _not_  changed
  hf  = @(n) exp(-36*(2* o.dk(n)/n).^36).*o.dk(n);

  [hx,hy,hz]=meshgrid( hf(o.N(1)),hf(o.N(2)),hf(o.N(3)) );

  tmp{1}=1i*hx; tmp{2}=1i*hy; tmp{3}=1i*hz;
  o.IK=o.crv(tmp);
end

%/* ************************************************** */
function use_twothirds_filtering(o)
% this routine is used to filter all first derivatives filtering out the last 1/3 of the spectrum
% functions curl, grad, div are changed. laplacian and invlaplacian are _not_  changed

  ts =@(n) (abs( o.dk(n) ) < floor(n/2 * 2/3)).* o.dk(n);

  [hx,hy,hz]=meshgrid( ts(o.N(1)),ts(o.N(2)),ts(o.N(3)) );
  tmp{1}=1i*hx; tmp{2}=1i*hy; tmp{3}=1i*hz;
  o.IK=o.crv(tmp);
end

%/* ************************************************** */
function [xy,xz,yz] = get_planes(o,scalar,zplane,yplane,xplane)
%function [xy,xz,yz] = get_2D_plane(o,scalar,zplane,yplane,zplane)
% xy: scalar in z_index=zplane (integer) [default, midplane]
% xz: scalar in y_index=yplane (integer) [default, midplane]
% yz: scalar in x_index=xplane (integer) [default, midplane]

  if nargin<3
    zplane = ceil(size(scalar,3)/2);
    yplane = ceil(size(scalar,2)/2);
    xplane = ceil(size(scalar,1)/2);
  end
  xy =  scalar(:,:,zplane);
  xz = reshape(scalar(:,yplane,:),[size(scalar,1),size(scalar,3)]);
  yz = reshape(scalar(xplane,:,:),[size(scalar,2),size(scalar,3)]);
end


end %methods
end %class


  
    

