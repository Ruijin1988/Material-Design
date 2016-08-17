for ii = 1:80
fname = sprintf('hidstates5th_WB_nowh(p2p2)_imresize_(1f1000f500f100f30f12ws189ws1s1ws)_%d',ii);
f1=load(sprintf('%s.mat', fname));   
temp = double([f1.hidstate;]);
temp = permute(temp,[3,2,1]);
xtr(:,ii) = temp(:)';
% xtr(:,ii) = abs(1-temp(:)');
% f1=load([CIFAR_DIR '/filter8_ws12.mat']);
end
fname = sprintf('hidstates5th_WB_nowh(p2p2)_imresize_(1f1000f500f100f30f12ws189ws1s1ws)');
save(sprintf('%s.mat',fname),'xtr', '-v7.3');