addpath('function_code','utils','results','4th layer hidstate(circlecut)_(1f1000f500f100f)')

fname=sprintf('rbm_circlecut_5thlayer_(1f1000f500f100f12ws189ws1ws1ws1ws)_alloy_w1_b30_trans_ntx1_gr1_pb0.5_pl0_iter_20000');
load(sprintf('%s.mat',fname));


params.optgpu = 0;
spacing = 1;
ws=params.ws;
rs=params.rs;
kcd=params.kcd;
txtype = params.txtype;
grid = params.grid;
numrot=params.numrot;
% params.numrot = 1;
numch=params.numch;
% numch=1;
% numchannels=params.numch;
weight=gpu2cpu_struct(weight);


W=weight.vishid;

hbias_vec=weight.hidbias;
dataname='alloy_scale';
% dataname='image';
% Tlist = get_txmat(params.txtype, params.rs, params.ws, params.grid, params.numrot, params.numch);
params.numtx = 1;

for ii = 1:100
fname=sprintf('hidstates4th_circlecut_(1f1000f500f100f12ws189ws1ws1ws)_%d',ii);
load([fname '.mat'],'hidstate')

image2=hidstate;
image2=permute(image2,[3 1 2]);
image2=reshape(image2,[sqrt(size(image2,1)) sqrt(size(image2,1)) size(image2,2)]);

image_reconstruct = crbm_5thlayer(image2, patch, W,weight,  params,ii); % remove rbm1.pars and set the value0.2 inside the function 10/15/2015
recon(:,ii)=image_reconstruct(:);
end

