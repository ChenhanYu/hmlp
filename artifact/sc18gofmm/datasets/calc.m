%% 
%%  @brief  These matlab scripts create 18 spd matrices.
%%
%%  @author George Biros (The Univeristy of Texas at Austin)
%%



classdef (Abstract) calc < handle
% SETUP of algebraic and derivative operations for D periodic fields using
% equispaced points
% derivatives of periodic functions defined in Omega : = [-pi,pi]^D
% support for scalar, vector, and tensor fields on a lattice of points in Omega
% 
% see: doc calc
  properties(GetAccess = 'public', SetAccess = 'public')
    use_real; % flag to indicate that all functions are real (and not complex)
  end

  properties(GetAccess = 'public', SetAccess = 'public')
    dim; % dimension of the problem
    N;   % d-dimensional array, where N(i) is the number of lattice points in dimension i
    IK;  % differentiation scaling for Fourier differentiation (frequency or K-space)
    L_IK; % laplacian, in K-space
    invL_IK;  % inverse laplacian in K-space
  end    
  
  properties(GetAccess = 'public', SetAccess = 'private', Constant=true)
    forward_fft = true; % boolean enum for fft_vec
    inverse_fft = false; % boolean enum for fft_vec
    %dk  = @(n) [0:n/2-1 0 -n/2+1:-1];  % diagonal  k-space derivative operator
    dk  = @(n) [0:n/2-1 -n/2 -n/2+1:-1];  % diagonal  k-space derivative operator
  end

methods (Abstract)
  setupIK(o)
  % setup derivatives matrix  

  S(o,x)
  % val = S(o,x) : convert matlab array X to scalar field

  V(o,x)
  % val = V(o,x): convert matlab array to vector field

  crv(o,Vs)
  % v = crv(o,Vs): creates vector field; Vs{1}..Vs{d} are the scalar components

  crT(o,Ts) 
  % T = crT(o.Ts): create tensor field from vectors 

  trans(o,T)
  %  return the transpose of a vector field
  
  gvc(o,v,j)
  % vj = gvc(o,v,j): returns  the j the component of vector field v
  
  gTc(o,T,j)
  % Tj = gTc(o,T,j): returns the jth column of a tensor field T

  curl(o, v)
  % [curlV] = curl(o, v): return the curl of a vector 'v'

  div_T(o,T)
  % [DivT] = div_T(o,T) return the divergence of a tensor T

  regulargrid(o)
  % X=regulargrid(o): return the coordinates of all the points of a regular grid in [-pi, pi].

  outer(o, vec1, vec2)
  %tensor = outer(o, vec1, vec2):  pointwise outer product of two tensor fields

  create_box_filter(o, percent)
  % create a frequency box cutoff filter
  
  %create_smooth_filter(o, scalar_function)
  % create a freqquency
  
  ones(o)
  % f=ones(o): return  scalar field = 1
end

%/******************************************************/
%/******************************************************/
methods
    
%/* ************************************************** */
function o = calc(N)
% constructor for derivative in [-pi,pi]
o.N=N(:);
assert(all(~mod(o.N,2))); % make sure all dimensions are even.

o.dim=length(N);
o.setupIK();
o.setupLIK();
end


%/* ************************************************** */
function flt = cutoff(o,n,percent)
% cutoff filter in frequency domain, percent: number of frequencies to keep.
flt = abs( o.dk(n) ) < round(n/2*percent);
end


%/******************************************************/
% setup Laplacian operators
function setupLIK(o)
o.L_IK = o.gvc(o.IK,1)*0;
for j=1:o.dim 
    tmp= o.gvc(o.IK,j).^2;
    o.L_IK=o.L_IK+tmp;
end
o.invL_IK=1./o.L_IK;
o.invL_IK( abs(o.invL_IK)==Inf ) = 0;
end


%/* ************************************************** */
function vals = C(o,x)
% convert tensor 'x' to column vector
% vals = C(o,x)
 vals=x(:); 
end

%/* ************************************************** */
function out = fft_filter(o, in, flt)
% function out = fft_filter( in, flt)    
% filter spectrally a field (scalar, vector, tensor) given a filter 'flt'
%     the 'flt' can be create with create_box_filter() or
%     create_smooth_filter()
field_type = length(size(in));

switch field_type
    case o.dim   % scalar field
        out = ifftn(fftn(in).*flt);

    case o.dim+1 % vector field
        for j=1:o.dim
            tmp{j} = ifftn( fftn( o.gvc(in,j) ) .* flt);
        end
        out = o.crv( tmp);

    case o.dim+2 % tensor field
        for j=1:o.dim
            tmp{j} = o.filter( o.gTc(in,j), flt);
        end
        out = o.crT( tmp);       
    otherwise
        error('invalid field: not a scalar, vector or tensor');
end

if o.use_real, out = real(out); end
end


%/* ************************************************** */
function [GradF] = grad(o,f)
% return the gradient of a scalar function 'f'
% [GradF] = grad(o,f)
  fk=fftn(f);
  for j=1:o.dim
    tmp{j} = ifftn( o.gvc( o.IK, j) .*fk );
  end
  GradF = o.crv( tmp );

  if o.use_real,  GradF = real(GradF); end
end

%/******************************************************/
function [GradV] = grad_V(o, V)
% return the gradient of a vector function 'V'
% [GradV] = grad_V(o, V)
  for j=1:o.dim
    tmp{j} = o.grad( o.gvc(V,j) );
  end
  GradV = o.crT( tmp );
  GradV = o.trans(GradV);
end

%/* ************************************************** */
function [divV]  = div(o,v)
% return the divergence of a vector 'v'
% [divV]  = div(o,v)
   divV = 0*o.ones;
   
   for j=1:o.dim
       divV_kj = fftn( o.gvc(v,j) ) .* o.gvc(o.IK,j);
       divV  = divV + ifftn( divV_kj );
   end

   if o.use_real, divV=real(divV); end
end

   

%/* ************************************************** */
function lapf = laplacian(o, f)
% return the  laplacian of a scalar
% lapf = -laplacian(o, f), so that -laplacian is SPD
    lapf = 0*f;
    fk = fftn(f);
    lapf = ifftn( o.L_IK .* fk );
    if o.use_real, lapf=real(lapf); end;
end

%/* ************************************************** */
function invlapf = inv_laplacian(o,f)
% apply the inverse laplacian on a scalar 
% invlapf = inv_laplacian(o,f)
    invlapf = 0*f;
    fk = fftn(f);
    invlapf = ifftn( o.invL_IK .* fk );
    if o.use_real, invlapf=real(invlapf); end;
end

%/* ************************************************** */
function out = helmholtz(o,f,kappa)
% return the Helmholtz of a scalar
    helm = 0*f;
    fk = fftn(f);
    out = ifftn( (o.L_IK + kappa) .* fk );
end

%/* ************************************************** */
function out = inv_helmholtz(o,f,kappa)
% apply the inverse laplacian on a scalar 
% out = inv_helmholtz(o,f,kappa)
    invlapf = 0*f;
    fk = fftn(f);
    out = ifftn( 1./(o.L_IK+kappa) .* fk );
end


%/* ************************************************** */
function lapv = laplacian_vec(o,v)
% return the laplacian of a vector
% lapv = laplacian_vec(o,v)

    lapv = 0*v;
    vk = fft_vec(o,v,o.forward_fft);
    lapv = o.scalevec(vk,o.L_IK);
    lapv = fft_vec(o,lapv,o.inverse_fft);
    if o.use_real, lapv=real(lapv); end;
end

%/* ************************************************** */
function invlapv = inv_laplacian_vec(o,v)
% apply the inverse laplacian on a vector
%  invlapv = inv_laplacian_vec(o,v)

    invlapv = 0*v;
    vk = fft_vec(o,v,o.forward_fft);
    invlapv = o.scalevec(vk,o.invL_IK);
    invlapv = fft_vec(o,invlapv,o.inverse_fft);    
    if o.use_real, lapv=real(invlapv); end;
end

%/* /* ************************************************** */
function vk = fft_vec(o, v, forward)
% apply ffts to each componenet of a vector
%
% vk = fft_vec(o, v, forward)
% if forward = o.fft_forward, then applies the forward transform
% if forward = o.fft_inverse, then applies the inverse transform
    vk = 0*v;
    if forward == o.forward_fft;
        lfft=@(x) fftn(x);
    else
        lfft=@(x) ifftn(x);
    end
    for j=1:o.dim, tmp{j}=lfft(o.gvc( v,j)); end
    vk = o.crv( tmp );
end


%/* ************************************************** */
function scalar = inner(o,vec1,vec2)
% pointwise inner product of two vector fields
% scalar = inner(o,vec1,vec2)
% for complex vectors, it is the proper inner product with conjugate
    scalar = dot(vec1,vec2,o.dim+1);
end

%/* ************************************************** */
function vector_out = scalevec(o, vector_in, scalar)
% scale a vector field by a scalar field:  vout = vin * scalar
% vector_out = scalevec(o, vector_in, scalar)
  vector_out=0*vector_in;
  for j=1:o.dim
    tmp{j}=scalar.*o.gvc(vector_in,j);
  end
  vector_out = o.crv(  tmp );

end
  

%/* ************************************************** */
function v=const_vec(o,vals)
% function v=const_vec(o,vals)
% return constant vector field
% vals = [v1, v2,..., vdim] , default = [1,1,...,1]
  if nargin<2, vals = ones(o.dim,1); end
  for d=1:o.dim
    tmp{d} = vals(d)*o.ones;
  end
  v = o.crv( tmp );
end

%/******************************************************/
function v = hodge_proj(o, vin)
% Hodge decomposition that takes a field 'vin' and returns a divergence-free field 
% v=vin-grad phi, where the grad phi is computed by a Laplace solve.
v = vin - o.grad(o.inv_laplacian(o.div(vin)));  
end

end %methods
end %class


  
    

