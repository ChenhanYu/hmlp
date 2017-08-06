%% 
%%  @brief  These matlab scripts create 18 spd matrices.
%%
%%  @author George Biros (The Univeristy of Texas at Austin)
%%


function savelargematrix(matvec,N,filename,precision,blocksize,usemap)
% this will fail for complex matrices;
if nargin<1, selftest; return; end
if nargin<5, blocksize=[]; end;
if nargin<6, usemap=false; end;

if isempty(blocksize), blocksize=1000; end;

if ~usemap,  savematrix(matvec,N,filename,precision,blocksize); return; end;

matsize=[N,N];
filesize=0;
fid=fopen(filename,'w+');
max_chunk_size=1e9;
fprintf('Creating file\n')
while filesize<prod(matsize)
  fprintf('\t...still working\n');
  towrite=min(prod(matsize)-filesize,max_chunk_size);
  filesize=filesize+fwrite(fid,zeros(towrite,1,precision),precision);
end
fclose(fid);
fprintf('Done creating file\n');

fprintf('Mapping file to memory...');
fm = memmapfile(filename,'Format',{'single',[N,N],'A'},'Writable',true);
fprintf('done\n');
sparseI=speye(N);
col=0;

fprintf('Adding columns to the matrix\n')
while col<N
  nc = min(N-col,blocksize);
  idx = col+[1:nc];
  col = col+nc;
  rhs = full(sparseI(:,idx));
  K=matvec(rhs);
  fm.Data.A(:,idx) = K;
  fprintf('\t added %05d columns so far\n', col);
end
fprintf('Done\n');

function selftest
n=16;
N=n^2;
J=gallery('poisson',n)/n/n;
filename=sprintf('kmat.bin');
fprintf('Factorizing input matrix...')
R=chol(J);  %R'*R=J
matvec_with_invJ= @(x) single(R\(R'\x));
fprintf('done\n');
savelargematrix(matvec_with_invJ, N, filename, 'single');


function savematrix(matvec,N,filename,precision,blocksize)
fid = fopen(filename,'w')
sparseI=speye(N);
col=0;
fprintf('Adding columns to the matrix\n')
K=zeros(N,blocksize);
while col<N
  nc = min(N-col,blocksize);
  idx = col+[1:nc];
  col = col+nc;
  rhs = full(sparseI(:,idx));
  K=matvec(rhs);
  fwrite(fid,K(:),'single');
  fprintf('\t added %05d columns so far\n', col);
end
fprintf('Done\n');

