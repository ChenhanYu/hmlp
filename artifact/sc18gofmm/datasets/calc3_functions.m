%% 
%%  @brief  These matlab scripts create 18 spd matrices.
%%
%%  @author George Biros (The Univeristy of Texas at Austin)
%%


function [vals,gradvals] = calc3_functions(o, type, varargin)
%function vals = calc3_functions(o, type)
%
% vals: scalar or vector field values
% gradvals  = gradient (analytic)
%
% o:  calc3 object
% type: type of fuctions
%    - gaussian,  varargin, 'mean',[x,y,z]; 'sigma', scalar_sigma
%    - trigonometric,  varagin='freq',[Kx,Ky,Kz]
%
%    - trigonometric_vector
%    - bell_markus
%    - taylor vortex
%    - vorticity singularity

vals=[]; gradvals = [];

X=o.regulargrid;
x = o.gvc(X,1);
y = o.gvc(X,2);
z = o.gvc(X,3);


switch type
%/* ************************************************** */
  case 'trigonometric'
    freq = [1 0 0];
    for k=1:2:length(varargin), % overwrite default parameter
      eval([varargin{k},'=varargin{',int2str(k+1),'};']);
    end;

    K = o.const_vec(freq);
    vals = exp ( 1i * o.inner(X, K ) );
    gradvals = 1i*o.scalevec(K,vals);

%/* ************************************************** */
  case 'gaussian'
    mean = [0,0,0];
    sigma = pi/4;
    for k=1:2:length(varargin), % overwrite default parameter
      eval([varargin{k},'=varargin{',int2str(k+1),'};']);
    end;
    
    C = o.const_vec(mean);
    r = X-C;
    vals = 1/sigma * exp( - 1/sigma^2 * o.inner(r,r) );
    gradvals =  o.scalevec( r, -2/sigma^2 * vals);
    
%/* ************************************************** */
  case 'trigonometric_vector'
    U{1} = sin(x).*( cos(3*y).*cos(z) - cos(y).*cos(3*z) );
    U{2} = sin(y).*( cos(3*z).*cos(x) - cos(z).*cos(3*x) );
    U{3} = sin(z).*( cos(3*x).*cos(y) - cos(x).*cos(3*y) );
    
    vals = o.crv(U);

%/* ************************************************** */
  case 'two_blobs'
   w{1} =  calc3_functions(o,'gaussian','sigma',pi/6); %Nx>=96 for good accuracy
    w{2} =  -w{1}.*calc3_functions(o,'trigonometric');
    w{3} = 2*w{1};
    om = o.curl(o.crv(w)); 
    w{1} =  -calc3_functions(o,'gaussian','mean',pi*[-0.2,-0.3,0.1],'sigma',pi/6); 
   w{2} =  w{1}.*calc3_functions(o,'trigonometric');
    w{3} = 1.2*w{1};
    om = om + o.curl(o.crv(w));
    vals = om;
    
%/* ************************************************** */
  case 'bell_markus'
    ro = 0.15; de =0.0333; ep=0.05; be = 15;
    x = x/2/pi;
    y = y/2/pi;
    z = z/2/pi;
    U{1} = tanh( (ro - sqrt(z.*z + y.*y))/de);
    U{2} = 0*x;
    U{3} = ep * exp( -be*(x.*x + y.*y) );

    vals = o.crv(U);

%/* ************************************************** */
  case 'kerr'
    z0 = 1;
    [omx1,omy1,omz1]=kerr_vortex_tube(x,y,z, z0);
    [omx2,omy2,omz2]=kerr_vortex_tube(x,y,z,-z0);

    om{1} =  omx1 - omx2;
    om{2} =  omy1 - omy2;
    om{3} =  omz1 - omz2;

    vals = o.crv(om);

%/* ************************************************** */
  case 'two_vortex_tubes'
    xc1=-0.866;
    xc2=-xc1;
    yc = 0;
    A=0.2;
    a1=pi/3;
    a2=2*pi/3;
    om0 = 26.0;
    [omx1,omy1,omz1]=vortex_tube(x,y,z,xc1,yc,A,a1,om0);
    [omx2,omy2,omz2]=vortex_tube(x,y,z,xc2,yc,A,a2,om0);
    
    om{1} =  omx1 + omx2;
    om{2} =  omy1 + omy2;
    om{3} =  omz1 + omz2;

    vals = o.crv(om);

%/******************************************************/
  case 'euler_exact'
    time = 0;
    rhs_only = 0;
    solution = 'trig0';
    for k=1:2:length(varargin), % overwrite default parameter
      eval([varargin{k},'=varargin{',int2str(k+1),'};']);
    end;
    switch solution
      case 'trig0'
        [om,u,rhs]=euler_exact_trig0(x,y,z,time);
      case 'trig1'
        [om,u,rhs]=euler_exact_trig1(x,y,z,time);
      case 'gauss'
        [om,u,rhs]=euler_exact_gauss(x,y,z,time);
      case 'gaussp'
        [om,u,rhs]=euler_exact_gaussp(x,y,z,time);
      otherwise
        error('invalid exact euler solution tag');
    end
    if rhs_only
      vals = o.crv(rhs);
    else
      vals{1} = o.crv(om);
      vals{2} = o.crv(u);
      vals{3} = o.crv(rhs);
    end

%/* ************************************************** */
  otherwise
    error('type not found');
end

end

%/* ************************************************** */
function [omx,omy,omz] = vortex_tube(x,y,z,xc,yc,A,a,om0)


Lx = 3;
Ly = 2;
Lz = 1;
x = Lx*x;
y = Ly*y;
z = Lz*z;

%%
r_cutoff = 2/3;
K=1/2*exp(2)*log(2);
omr_1 = @(r) (1 - exp(-K./r.*exp(1./(r-1))) ).* (r<1);
omr_f = @(r) omr_1(r/r_cutoff);

%%
tbx = xc + A*cos(a)*(1+cos(z));
tby = yc + A*sin(a)*(1+cos(z));

r = sqrt( (x-tbx).^2 + (y-tby).^2 );

omr = om0 * omr_f(r);

omx = omr.*(-A*cos(a).*sin(z));
omy = omr.*(-A*sin(a).*sin(z));
omz = omr;
end

%/* ************************************************** */
function [omx,omy,omz] = kerr_vortex_tube(x,y,z,z0)
  % domain size
  Lx = 4;
  Ly = 4;
  Lz = 2;
  % resize x,y,z defined in [-pi, pi]
  x = (Lx*(x)); 
  y = (Ly*(y));
  z = (Lz*(z));
  
% parameters from from Hou and Li 2006 paper
% notice that pi's in Lx,Ly,Lz and their formulas cancel out so
% I've simplified them here. 
  R = 0.75;
  dy1 = 0.5;
  dy2 = 0.4;
  dx  =-1.6;
  dz  =  0;
  x0 = 0;
  
% definition of vortex hou trajectory for positive z
  y2 = y + pi*Ly*dy2*sin(y/Ly);
  s  = y2 +pi*Ly*dy1*sin(y2/Ly);
  xs = x0 +  dx*cos(s/Ly);
  zs = z0 +  dz*cos(s/Ly);

% Gaussian smoothing
  r = sqrt( (x-xs).^2 + (z-zs).^2)/R;
  TR = (r.^2).*(r<=1);
  FR = -TR./(1-TR) + TR.^2 .* (1 + TR + TR.^2);
  OmR = exp(FR).*(r<=1);
  
%  save('r.mat','r', 'FR', 'OmR');
  
  omx = -OmR .* dx/Lx .* (1+pi*dy2*cos(y/Ly)) .* (1+pi*dy1*cos(y2/Ly)) .* sin(s/Lx);
  omy = OmR;
  omz = -OmR .* dz/Lz .* (1+pi*dy2*cos(y/Ly)) .* (1+pi*dy1*cos(y2/Ly)) .* sin(s/Lz);
  
end  


%/******************************************************/
function [om,u,rhs]=euler_exact_trig0(xx,yy,zz,tt)
u{1}=0*xx;
u{2}=-2*sin(2*zz);
u{3}=0*xx;

om{1}= 4*cos(2*zz);
om{2} = 0*xx;
om{3} = 0*xx;

rhs{1} = 0*xx;
rhs{2} = 0*xx;
rhs{3} = 0*xx;
end


function [om,u,rhs]=euler_exact_trig1(xx,yy,zz,tt)
a = 2; b=1;
u{1}=0*xx;
u{2}=-a*sin(a*zz)*cos(b*2*pi*tt);
u{3}=0*xx;

om{1}= a^2*cos(a*zz)*cos(b*2*pi*tt);
om{2} = 0*xx;
om{3} = 0*xx;

rhs{1} = -a^2*b*2*pi*sin(b*2*pi*tt)*cos(a*zz);
rhs{2} = 0*xx;
rhs{3} = 0*xx;
end


function [om, u, rhs]=euler_exact_gauss(xx,yy,zz,tt)
b=0; sigma = 0.24;

u{1} = 0.*xx;
u{2} = -(2.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2)/sigma^2).*cos(2.*pi.*b*tt))/sigma^3;
u{3} = (2.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2)/sigma^2).*cos(2.*pi.*b*tt))/sigma^3;
 
om{1}= -(4.*exp(-(xx.^2 + yy.^2 + zz.^2)/sigma^2).*cos(2.*pi.*b*tt).*(- sigma^2 + yy.^2 + zz.^2))/sigma^5;
om{2}=                    (4.*xx.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2)/sigma^2).*cos(2.*pi.*b*tt))/sigma^5;
om{3}=                      (4.*xx.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2)/sigma^2).*cos(2.*pi.*b*tt))/sigma^5;
 
rhs{1}=                                                      (8.*pi.*b.*exp(-(xx.^2 + yy.^2 + zz.^2)/sigma^2).*sin(2.*pi.*b*tt).*(- sigma^2 + yy.^2 + zz.^2))/sigma^5;
rhs{2}= -(8.*xx.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2))/sigma^2).*(2.*zz.*cos(2.*pi.*b*tt).^2 + pi.*b.*sigma^3.*yy.*exp((xx.^2 + yy.^2 + zz.^2)/sigma^2).*sin(2.*pi.*b*tt)))/sigma^8;
rhs{3}=  (8.*xx.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2))/sigma^2).*(2.*yy.*cos(2.*pi.*b*tt).^2 - pi.*b.*sigma^3.*zz.*exp((xx.^2 + yy.^2 + zz.^2)/sigma^2).*sin(2.*pi.*b*tt)))/sigma^8;
end



function [om, u, rhs]=euler_exact_gaussp(xx,yy,zz,tt)
b=1; sigma = 0.4; 
u{1}=0.*xx;
u{2}= -(4.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma.^4).*cos(2.*pi.*b.*tt).*(xx.^2 + yy.^2 + zz.^2))/sigma.^5;
u{3}=  (4.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma.^4).*cos(2.*pi.*b.*tt).*(xx.^2 + yy.^2 + zz.^2))/sigma.^5;
 
 
om{1}= (8.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma.^4).*cos(2.*pi.*b.*tt).*(xx.^2 + 2.*yy.^2 + 2.*zz.^2))/sigma.^5 - (16.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma.^4).*cos(2.*pi.*b.*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2).^2)/sigma.^9;
om{2}=                                           (8.*xx.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma.^4).*cos(2.*pi.*b.*tt).*(- sigma.^4 + 2.*xx.^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma.^9;
om{3}=                                          (8.*xx.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma.^4).*cos(2.*pi.*b.*tt).*(- sigma.^4 + 2.*xx.^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma.^9;


 
rhs{1} = (4.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + yy.^2 + zz.^2).*((32.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^9 - (32.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt))/sigma^5 + (32.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + 2.*yy.^2 + 2.*zz.^2).*(xx.^2 + yy.^2 + zz.^2))/sigma^9 + (64.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2))/sigma^9 - (64.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2).^3)/sigma^13))/sigma^5 - (4.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + yy.^2 + zz.^2).*((32.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^9 - (32.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt))/sigma^5 + (32.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + 2.*yy.^2 + 2.*zz.^2).*(xx.^2 + yy.^2 + zz.^2))/sigma^9 + (64.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2))/sigma^9 - (64.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2).^3)/sigma^13))/sigma^5 - (16.*pi.*b.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*sin(2.*pi.*b*tt).*(xx.^2 + 2.*yy.^2 + 2.*zz.^2))/sigma^5 + (32.*pi.*b.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*sin(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^9;

rhs{2}= (64.*xx.*zz.^3.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^14 + (8.*xx.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*((8.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + 2.*yy.^2 + 2.*zz.^2))/sigma^5 - (16.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^9))/sigma^5 + (4.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + yy.^2 + zz.^2).*((16.*xx.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt))/sigma^5 - (32.*xx.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + 2.*yy.^2 + 2.*zz.^2).*(xx.^2 + yy.^2 + zz.^2))/sigma^9 - (64.*xx.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2))/sigma^9 + (64.*xx.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2).^3)/sigma^13))/sigma^5 - (256.*xx.*zz.^3.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(xx.^2 + yy.^2 + zz.^2).^2.*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^18 + (32.*xx.*yy.^2.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(xx.^2 + yy.^2 + zz.^2).*(8.*xx.^2.*zz + 8.*yy.^2.*zz + 8.*zz.^3))/sigma^14 + (32.*xx.*zz.^2.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(xx.^2 + yy.^2 + zz.^2).*(8.*xx.^2.*zz + 8.*yy.^2.*zz + 8.*zz.^3))/sigma^14 + (64.*xx.*yy.^2.*zz.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^14 - (16.*xx.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*((8.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + 2.*yy.^2 + 2.*zz.^2))/sigma^5 - (16.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^9).*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^9 + (64.*xx.*zz.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(xx.^2 + yy.^2 + zz.^2).*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^14 - (16.*pi.*b.*xx.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*sin(2.*pi.*b*tt).*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^9 - (256.*xx.*yy.^2.*zz.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(xx.^2 + yy.^2 + zz.^2).^2.*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^18;

rhs{3}= (256.*xx.*yy.^3.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(xx.^2 + yy.^2 + zz.^2).^2.*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^18 - (8.*xx.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*((8.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + 2.*yy.^2 + 2.*zz.^2))/sigma^5 - (16.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^9))/sigma^5 - (4.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + yy.^2 + zz.^2).*((16.*xx.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt))/sigma^5 - (32.*xx.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + 2.*yy.^2 + 2.*zz.^2).*(xx.^2 + yy.^2 + zz.^2))/sigma^9 - (64.*xx.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2))/sigma^9 + (64.*xx.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2).^3)/sigma^13))/sigma^5 - (64.*xx.*yy.^3.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^14 - (32.*xx.*yy.^2.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(xx.^2 + yy.^2 + zz.^2).*(8.*xx.^2.*yy + 8.*yy.^3 + 8.*yy.*zz.^2))/sigma^14 - (32.*xx.*zz.^2.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(xx.^2 + yy.^2 + zz.^2).*(8.*xx.^2.*yy + 8.*yy.^3 + 8.*yy.*zz.^2))/sigma^14 - (64.*xx.*yy.*zz.^2.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^14 + (16.*xx.*yy.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*((8.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(xx.^2 + 2.*yy.^2 + 2.*zz.^2))/sigma^5 - (16.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*cos(2.*pi.*b*tt).*(yy.^2 + zz.^2).*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^9).*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^9 - (64.*xx.*yy.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(xx.^2 + yy.^2 + zz.^2).*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^14 - (16.*pi.*b.*xx.*zz.*exp(-(xx.^2 + yy.^2 + zz.^2).^2/sigma^4).*sin(2.*pi.*b*tt).*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^9 + (256.*xx.*yy.*zz.^2.*exp(-(2.*(xx.^2 + yy.^2 + zz.^2).^2)/sigma^4).*cos(2.*pi.*b*tt).^2.*(xx.^2 + yy.^2 + zz.^2).^2.*(2.*xx.^4 - sigma^4 + 4.*xx.^2.*yy.^2 + 4.*xx.^2.*zz.^2 + 2.*yy.^4 + 4.*yy.^2.*zz.^2 + 2.*zz.^4))/sigma^18;

end 


 

  
