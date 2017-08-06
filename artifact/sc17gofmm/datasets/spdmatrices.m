%% 
%%  @brief  These matlab scripts create 18 spd matrices.
%%
%%  @author George Biros (The Univeristy of Texas at Austin)
%%


clear all; clear globals; clc;
%addpath('~biros/projects/matlab/lib');
% nohup matlab -nosplash -nodesktop -r 'spdmatrices;exit;' >& output192 &

 % 2d problems will have n^2 points on a regular grid;
 % 3d problems will have n^3 points on a regular grid
 % kernel problems will have n^2 random points. 
n=64;

dim=6;  % dimensions for  kernel matrices
precision='single';
savefile=true;   % if false, matrices will not be saved on disk
dokernels=true;  % if false, no kernel matrices are computed
savepoints=true; % if false, no kernel points will be save
dographs=~true;  % if false, no graphs will be created; this requires certain graph files are in the path. search for "dographs" in the m-fiel

%----------------------  do not edit below this line. 
N=n^2;
fprintf('The total number of points will be %d\n', N);
fprintf('All matrices are be dense and stored\n');
fprintf('For some matrices, script can be modified to have matrix-free versions\n');
KA={}; cnt=1;
t=linspace(0,1,N);
X1D= [t(:)];
t=linspace(0,1,n);
[gx,gy]    = meshgrid(t,t);  
X2D= [gx(:)';gy(:)'];
n3=  2^ceil( 2./3. * log2(n) );
t3=linspace(0,1,n3);
[gx,gy,gz] = meshgrid(t3,t3,t3);
X3D= [gx(:)';gy(:)';gz(:)'];

%
X=randn(N,dim); % for kernel matrices

if savepoints
    filename=sprintf('X3DN%d.points.bin', length(X3D));
    f=fopen(filename,'wb');
    fwrite(f,single(X3D(:)), 'single');
    fclose(f);
    filename=sprintf('X2DN%d.points.bin', N);
    f=fopen(filename, 'wb'); 
    fwrite(f,single(X2D(:)),'single'); 
    fclose(f); 
    filename=sprintf('X1DN%d.points.bin', N);
    f=fopen(filename, 'wb'); 
    fwrite(f,single(X1D(:)),'single'); 
    fclose(f); 
    filename=sprintf('XKEN%d.points.bin', N);
    f=fopen(filename, 'wb'); 
    X=X';
    fwrite(f,single(X(:)),'single'); 
    fclose(f); 
    X=X';
end
%clear('gx','gy','gz','X2D','X3D','X1D');
XKEN=X;

%eigevalue solver options:
eigopt.issym = true;
eigopt.isreal= true;
eigopt.tol = 1e-2;



%% 2D Poisson problem
fprintf('MATRIX %d: SPARSE: forward 2D Poisson problem operator\n',cnt);
J=gallery('poisson', n)/n/n;
s=eigs(J,1,'LM',eigopt);
J=J/s;
if ~savefile, KA{cnt}=J;  end;
matvec=@(x) single( J*x);
filename=sprintf('K%02dN%d.bin', cnt, N);
if savefile,savelargematrix(matvec,N,filename,precision);end 
%%
cnt=cnt+1;
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);


%%
fprintf('MATRIX %d: inverse squared 2D Poisson problem operator\n',cnt);
R=chol(J);
matvec1=@(x) R\(R'\full(x));
matvec =@(x) matvec1(matvec1(x));
s=eigs(matvec,N,1,'LM',eigopt);
matvec = @(x) single(matvec1(matvec1(x))/s);
filename=sprintf('K%02dN%d.bin', cnt, N);
if savefile,savelargematrix(matvec,N,filename,precision);end
if ~savefile, KA{cnt}=matvec(speye(N)); end;
cnt=cnt+1;
clear('R'); clear('J');
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);

%% Helmohltz in 2D
fprintf('MATRIX %d: inverse squared 2D Helmholtz problem operator\n',cnt);   % MATRIX 3
frequency = n/10;
omega=2*pi*frequency;
J=gallery('poisson',n)/n/n;
N=size(J,1);
I=speye(N);
J= omega^2*I - J;
fprintf('Factorizing matrix..\n'); [L,U]=lu(J); fprintf('Done\n');
matvec1=@(x) U\(L\full(x));
matvec = @(x) matvec1(matvec1(x));
s=eigs(matvec,N,1,'LM',eigopt);
matvec =@(x) single(matvec1(matvec1(x))/s);
filename=sprintf('K%02dN%d.bin', cnt, N);
if savefile, savelargematrix(matvec,N,filename,precision); end
if ~savefile, KA{cnt}=matvec(I); end;
cnt=cnt+1;  
clear('J'); clear('L'); clear('U'); clear('I');
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);

%% distance based functions; green's functions and kernels/ 
if dokernels
    fprintf('MATRIX %d--%d Computing various kernel functions\n',cnt,cnt+6);

    R=pdist2(X,X);
    Rnnz = nonzeros(triu(R));
    h=median(Rnnz);
    hmin=min(Rnnz); hmax=max(Rnnz); % get h statistics.
    clear('Rnnz');
    
    % different kernel functions (but I'm not using all of them)
    gaus=@(R)     exp(-R.^2/2);   
    orns=@(R)     exp(-R);
    quad=@(R,c)   sqrt(c.^2+R.^2);
    iqua=@(R,c)   1./quad(R,c);   
    poly=@(R,c,p) (R+c).^p;
    lapl=@(R)     1./ (R.^(2-dim));
    rlap=@(R) 1./(R.^(2-dim)+hmin);  % regularized Laplacian

    jj=1;
    fun{jj} = @(R) gaus(R/(0.7*h+0.3*hmax)); jj=jj+1;  %matrix 4 
    fun{jj} = @(R) gaus(R/h);  jj=jj+1;
    fun{jj} = @(R) gaus(R/(.1*h+.9*hmin)); jj=jj+1; % matrix 6
    fun{jj} = @(R) lapl(R);jj=jj+1;
    %Kl =lapl(R); Kl(isnan(Kl))=0; Kl=Kl+2*diag(Kl*ones(N,1)); % need to fix 1/0 singularity of self interactions.
    fun{jj} = @(R) quad(R,sqrt(h)); jj=jj+1; % Kq=Kq+2*diag(Kq*ones(N,1));           %matrix 8 . 
    fun{jj} = @(R) iqua(R,h); jj=jj+1;
    fun{jj} = @(X) poly(X*X',h,2);   %Kp=Kp+2*diag(Kp*ones(N,1));  % matrix 10
    
    for j=1:jj
        if j<jj, K= fun{j}(R); else, clear('R'); K=fun{j}(X); end;
        % fix diagonal so matrices are spd
        diagidx = linspace(1,numel(K),length(K));
        K(diagidx)=0;
        one = ones(length(K),1);
        v=K*one;
        K(diagidx)=2*v;
        clear('diagidx');
        s=eigs(K,1,'LM',eigopt);
        
        if savefile
            filename=sprintf('K%02dN%d.bin',cnt,length(K));
            matvec=@(x)K*x/s;
            savelargematrix(matvec,length(K),filename,precision);
        else
            KA{cnt}=K/s; 
        end
       cnt = cnt+1;
       clear('K'); clear('one'); clear('v');
    end
    clear('fun','matvec','matvec1','X');
else
   cnt = cnt+7;
end
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);

%% 1D variable coefficients Poisson;
fprintf('MATRIX %d: inverse squared 1D variable coefficient Poisson problem operator\n',cnt);  % MATRIX 11
e = ones(N+1,1);
Dn2c = N*spdiags([-e,e],[0,1],N,N+1); 
Dc2n = N*spdiags([-e,e],[-1,0],N+1,N); 
Ac2n = 1/2*spdiags([e,e],[-1,0],N+1,N);
coeff = 1+(abs(sin(10*2*pi*linspace(0,1,N)'))>0.7)*1000;
coeff = Ac2n*coeff;
J=Dn2c*spdiags(ones(N+1,1).*coeff,0,N+1,N+1)*Dc2n;
R=chol(-J);
matvec1=@(x) R\(R'\full(x));
matvec =@(x) matvec1(matvec1(x));
s=eigs(matvec,N,1,'LM',eigopt);
matvec =@(x) single(matvec1(matvec1(x))/s);
filename=sprintf('K%02dN%d.bin', cnt, N);
if savefile, savelargematrix(matvec,N,filename,precision); end;
if ~savefile, KA{cnt}=matvec(speye(N)); end;
clear('Dn2c','Dc2n','Ac2n','coeff','J','R');
cnt=cnt+1;
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);

%% 2d variable coeffiecients
N=n^2;
fprintf('MATRIX %d: inverse squared 2D variable coefficient Poisson problem operator\n',cnt);  % MATRIX 12
e = ones(n,1);
D1 = n*spdiags([-e,e],[-1,0],n,n);
contrast=1E-4;
diffusion_coefficient=ones(n,n);  
rows=randi(n-2,n/2,1);  cols=randi(n-2,n/2,1); 
diffusion_coefficient(rows+1,2:end-1)=contrast; 
diffusion_coefficient(2:end-1,cols+1)=contrast; 
%surf(diffusion_coefficient); view(2); shading interp;

Dx=kron(speye(n),D1);
Dy=kron(D1,speye(n));

DIV=[Dx Dy];
GRAD=DIV';
diffcoff = diffusion_coefficient(:);
diffcoff = spdiags(diffcoff,0,length(diffcoff),length(diffcoff));
J=DIV*blkdiag(diffcoff,diffcoff)*GRAD;
R=chol(J);
matvec1=@(x) R\(R'\full(x));
matvec =@(x) R\(R'\matvec1(x));
s=eigs(matvec,N,1,'LM',eigopt);
matvec =@(x) single(R\(R'\matvec1(x))/s);
filename=sprintf('K%02dN%d.bin', cnt, N);
if savefile, savelargematrix(matvec,N,filename,precision); else, KA{cnt}=matvec(speye(N)); end;
cnt=cnt+1;
clear('R');
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);
%%
% 2D advection diffusion
fprintf('MATRIX %d: inverse squared 2D variable coefficient advection-diffusion problem operator\n',cnt);  %MATRIX 13
t=linspace(0,1,n); [x,y]=meshgrid(t,t); 
vx= cos(3*2*pi*x+3*2*pi*y); 
vy = sin(10*2*pi*x+3*2*pi*y); 
vx = spdiags(vx(:),0,n^2,n^2);
vy = spdiags(vy(:),0,n^2,n^2);
J=J+1e3*((Dx-Dx')*vx+(Dy-Dy')*vy);
[L,U]=lu(J);
matvec1=@(x) U\(L\full(x));
matvec =@(x) L'\(U'\matvec1(x));
s=eigs(matvec,N,1,'LM',eigopt);
matvec =@(x) single(L'\(U'\matvec1(x))/s);
filename=sprintf('K%02dN%d.bin', cnt, N);
if savefile, savelargematrix(matvec,N,filename,precision); else, KA{cnt}=matvec(speye(N)); end;
cnt=cnt+1;
clear('L'); clear('U'); 
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);

%% 2D advection diffusion
fprintf('MATRIX %d: inverse squared 2D variable coefficient reaction-advection-diffusion problem operator\n',cnt);  %MATRIX 14
t=linspace(0,1,n); [x,y]=meshgrid(t,t); 
rho= cos(3*2*pi*x+3*2*pi*y).^2; 
clear('x','y','t');
J=J+10*spdiags(rho(:),0,n^2,n^2);
[L,U]=lu(J);
matvec1=@(x) U\(L\full(x));
matvec =@(x) L'\(U'\matvec1(x));
s=eigs(matvec,N,1,'LM',eigopt);
matvec =@(x) single(L'\(U'\matvec1(x))/s);
filename=sprintf('K%02dN%d.bin', cnt, N);
if savefile, savelargematrix(matvec,N,filename,precision); else, KA{cnt}=matvec(speye(N)); end;
cnt=cnt+1;
clear('L','U','J','rho','vx','vy','Dx','Dy','DIV','GRAD','D1','diffcoff','diffusion_coefficient');
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);


%%


%%
addpath('~/projects/matlab/proj/euler');
 %% reaction-advection-diffusion - spectral
fprintf('MATRIX %d: Spectral 2D reaction diffusion advection: squared forward\n', cnt);
dim=2;
N=n^2;
o = calc2([n,n]);
X=o.regulargrid; 
v=o.crv({sin(16*o.gvc(X,2)).^2, cos(2*o.gvc(X,1)).^2});
o.use_real = true;
diffusion_coefficient = 1E-7;
Jfun = @(cin)cin-diffusion_coefficient*o.laplacian(cin) + o.inner( o.grad(cin), v );
Jtfun = @(cin) cin - diffusion_coefficient*o.laplacian(cin) - o.div(o.scalevec(v, cin));
matvec =@(x) o.C( Jtfun( Jfun( o.S(full(x)))));
s=eigs(matvec,N,1,'LM',eigopt);
matvec =@(x) single(o.C( Jtfun( Jfun( o.S(full(x)))))/s);
filename=sprintf('K%02dN%d.bin', cnt, N);
%
if savefile
    savelargematrix(matvec,N,filename,precision,1);
else
    Is=speye(N); K=zeros(N); 
    for kk=1:N; K(:,kk)=matvec(Is(:,kk)); end;
    KA{cnt}=K; clear('K'); 
end
cnt=cnt+1;
clear('J','v');
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);

%% variable coefficient poisson, fractured media
fprintf('MATRIX %d: Spectral 2D fractured media Poisson: forward\n',cnt);   % MATRIX 16
N= n^2;
o=calc2([n,n]);  o.use_real=true; X=o.regulargrid;
contrast = 10^-10;
diffusion_coefficient=ones(n);  
rows=randi(n-2,n/2,1);  cols=randi(n-   2,n/2,1); 
diffusion_coefficient(rows+1,2:end-1)=contrast; 
diffusion_coefficient(2:end-1,cols+1)=contrast; 
%surf(diffusion_coefficient); view(2); shading interp;
Jfun = @(u) u- o.div(o.scalevec( o.grad(u), diffusion_coefficient));
matvec = @(x) o.C(Jfun(o.S(full(x))));
s=eigs(matvec,N,1,'LM',eigopt);
matvec = @(x) single(o.C(Jfun(o.S(full(x))))/s);
filename=sprintf('K%02dN%d.bin', cnt, N);
if savefile
    savelargematrix(matvec,N,filename,precision,1);
else
    Is=speye(N); K=zeros(N); 
    for kk=1:N; K(:,kk)=matvec(Is(:,kk)); end;
    KA{cnt}=K; clear('K','Is'); 
end
cnt = cnt+1;
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);

%% 3D laplacian
fprintf('MATRIX %d: 3D constant coefficient Laplacian: forward\n',cnt);
N=n3^3;
o=calc3([n3,n3,n3]); o.use_real=true;
Jfun = @(u) u -o.laplacian(u);
matvec = @(x) o.C(Jfun(o.S(full(x))));
s=eigs(matvec,N,1,'LM',eigopt);
matvec = @(x) single(o.C(Jfun(o.S(full(x))))/s);
filename=sprintf('K%02dN%d.bin', cnt, N);
if savefile
    savelargematrix(matvec,N,filename,precision,1);
else
    Is=speye(N); K=zeros(N); 
    for kk=1:N; K(:,kk)=matvec(Is(:,kk)); end;
    KA{cnt}=K; clear('K','Is'); 
end
cnt = cnt+1;
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);
%% 3d constant coeffiecients Dirichlent
fprintf('MATRIX %d: inverse squared 3D constant coefficient Laplacian\n',cnt);
e = ones(n3,1);
D1 = n3*spdiags([-e,e],[-1,0],n3,n3);
D2x=kron(speye(n3),D1);
D2y=kron(D1,speye(n3));
Dx = kron(speye(n3),D2x);
Dy = kron(speye(n3),D2y);
Dz = kron(D1,kron(speye(n3),speye(n3)));
DIV=[Dx Dy Dz];
GRAD=DIV';
J=DIV*GRAD;
fprintf('Factorizing matrix..\n'); R=chol(J); fprintf('Done\n');
matvec1=@(x) R\(R'\full(x));
matvec =@(x) R\(R'\matvec1(x));
s=eigs(matvec,N,1,'LM',eigopt);
matvec =@(x) single(R\(R'\matvec1(x))/s);
filename=sprintf('K%02dN%d.bin', cnt, N);
if savefile, savelargematrix(matvec,N,filename,precision); else, KA{cnt}=matvec(speye(N)); end;
cnt=cnt+1;
clear('R','J','Dx','Dy','Dz','D1','DIV','GRAD');
mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);
%%
if dographs
% files taken from http://yifanhu.net/GALLERY/GRAPHS/search.html
graphs = { 'c-49.mat', 'rgg_n_2_16_s0.mat', 'denormal.mat', 'conf6_0-8x8-30.mat'};
ngr = length(graphs);
fprintf('MATRIX %d--%d: graph Laplcacian inverse\n', cnt,cnt+ngr-1);

for gr=1:ngr
    fprintf('\n\t processing file %s\n',graphs{gr});
    load(graphs{gr});
    N=length(Problem.A);
    fprintf('Factorizing...\n'); R=cholesky_graphlaplacian(Problem.A); fprintf('Done\n');
    matvec1= @(x) R\(R'\full(x));
    s=eigs(matvec1,N,1,'LM',eigopt);
    matvec = @(x) single(matvec1(x)/s);
    filename=sprintf('K%02dN%d.bin', cnt, N);
    if savefile, savelargematrix(matvec,N,filename,precision); else, KA{cnt}=matvec(speye(N)); end; 
    cnt=cnt+1;
    mem=whos; fprintf('Memory %1.2EGB\n', sum([mem.bytes])/1E9);
    clear('R', 'Problem');
end
end

    
%%
fprintf('Done. Computed %d matrices\n', cnt-1);

%% check spd
if ~savefile
    for j=1:cnt-1
        if isempty(KA{j}), continue; end;
        K=KA{j}; 
        if any(diag(K)<=0), fprintf('\n\t ERROR: found non-positive diagonal entry\n'); end;
        esym=norm(K-K','fro')/norm(K,'fro');
        evs=eig(K);
        epos=sum(evs<0);
        maxevs=max(evs);
        fprintf('Matrix %d',j);
        fprintf(' 2-norm=%1.2E ', maxevs);
        fprintf('\t Symmetry error=%1.2E ',esym);
        fprintf('\t Found %d negative eigenvalues\n',epos);
        if epos>0
            negevs=evs( find(evs<0) );
            num_signi_negevs =  sum ( abs(negevs)/maxevs > eps(maxevs)*1000 );
            fprintf('\t\t Found %d significant negative eigenvalue(s)\n', num_signi_negevs);
        end
    end
end

















